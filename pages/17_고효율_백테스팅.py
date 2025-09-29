import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- 상수 정의 ---
TARGET_PERIOD = 7 # 예측할 미래 일수
TRAIN_DAYS = 365 # 훈련에 사용할 기간 (1년으로 단축)

# --- LightGBM 모델 하이퍼파라미터 (과적합 방지 최적화) ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1200,       # 훈련 기간 단축에 맞춰 Estimator 증가
    'learning_rate': 0.02,      # 과적합 방지를 위해 학습률 미세 조정
    'feature_fraction': 0.75,   # 피처 일부만 사용 (과적합 방지)
    'bagging_fraction': 0.75,   # 데이터 일부만 사용 (과적합 방지)
    'bagging_freq': 1,
    'num_leaves': 31,           # 트리의 복잡도 제한 (과적합 방지)
    'max_depth': 8,             # 트리의 깊이 제한
    'lambda_l1': 0.2,           # L1 규제 강화
    'lambda_l2': 0.2,           # L2 규제 강화
    'verbose': -1,              # 로그 출력 끔
    'n_jobs': -1,
    'seed': 42
}

# --- 1. 피처 엔지니어링 함수 ---
def create_features(df):
    """
    LightGBM 모델 훈련을 위한 시계열 피처를 생성합니다.
    """
    df = df.copy()

    # 1. 시간 기반 피처 (주기성 반영)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear
    
    # 2. 지연 피처 (Lag Features)
    lags = [1, 3, 7, 14, 30]
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
    # 3. 이동 평균 및 볼륨 지표
    windows = [5, 20, 60]
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Vol_{window}'] = df['Close'].rolling(window=window).std()

    # 4. 상대적인 변화율 (차분 피처)
    df['Daily_Change'] = df['Close'].pct_change()
    
    # 5. 타겟 변수 (미래 1일 후의 종가)
    df['Target'] = df['Close'].shift(-1) 

    # 결측치 제거 (Lag Features 때문에 발생하는 초기 결측치)
    df = df.dropna()
    
    return df

# --- 2. 데이터 로드 함수 ---
@st.cache_data(ttl=60*60*4) # 캐시 유지 시간을 4시간으로 설정하여 로딩 속도 개선
def load_data(ticker):
    """
    YFinance를 사용하여 주가 데이터를 로드하고 컬럼 이름을 정리합니다.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=TRAIN_DAYS + 100)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        
        # yfinance 멀티 인덱스 문제 방지 및 컬럼 이름 정리
        # 컬럼 이름이 튜플일 경우 문자열로 변환한 후 정리합니다. (오류 해결 핵심)
        def sanitize_column_name(col):
            if isinstance(col, tuple):
                # 튜플인 경우, 요소들을 합쳐서 문자열로 만듭니다. (예: ('Close', 'AAPL') -> 'Close_AAPL')
                name = '_'.join(map(str, col))
            else:
                name = str(col)
            
            # 특수 문자 정리
            return name.replace(' ', '_').replace('.', '').replace(',', '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_')

        data.columns = [sanitize_column_name(col) for col in data.columns.to_list()]
        
        return data.dropna()
    except Exception as e:
        st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# --- 3. 모델 훈련 및 검증 함수 ---
def train_and_validate_model(data_features):
    """
    시계열 분할을 이용해 LightGBM 모델을 훈련 및 검증하고 결과를 반환합니다.
    """
    
    # Feature와 Target 분리
    X = data_features.drop('Target', axis=1)
    y = data_features['Target']
    
    # --- LightGBM 오류 방지를 위한 피처 이름 정리 (Sanitization) ---
    # 데이터 로드 단계에서 컬럼을 정리했으므로 이 단계는 LightGBM이 요구하는 문자열 규칙만 강제합니다.
    sanitized_columns = [
        str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '')
        for col in X.columns
    ]
    X.columns = sanitized_columns # 컬럼 이름 업데이트
    # ------------------------------------------------------------------
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    # 시계열 교차 검증 (Time Series Split) 설정
    tscv = TimeSeriesSplit(n_splits=3)
    
    rmse_scores = []
    
    st.markdown("##### 🚀 모델 훈련 및 시계열 검증 진행 중...")
    progress_bar = st.progress(0)
    
    final_model = None
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled_df)):
        X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # LightGBM 모델 훈련 (조기 종료 적용)
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=-1)] # 조기 종료 라운드 조정
        )
        
        # 검증 세트 예측 및 RMSE 계산
        val_predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        rmse_scores.append(rmse)
        
        progress_bar.progress((fold + 1) / 3)
        st.caption(f"Fold {fold+1} 검증 완료. RMSE: {rmse:.4f}")
        final_model = model # 마지막 모델 저장

    avg_rmse = np.mean(rmse_scores)
    st.success(f"✅ 모델 훈련 완료. 평균 검증 RMSE: {avg_rmse:.4f}")
    
    return final_model, scaler, X.columns

# --- 4. 미래 예측 함수 ---
def predict_future(model, scaler, last_data, feature_columns):
    """
    훈련된 모델을 사용하여 향후 7일간의 주가를 예측합니다.
    """
    
    future_dates = [last_data.index[-1] + datetime.timedelta(days=i) for i in range(1, TARGET_PERIOD + 1)]
    
    # 마지막 데이터를 기반으로 예측을 위한 초기 피처 생성
    current_data = create_features(last_data).iloc[-1].to_frame().T # 피처를 다시 생성하여 예측 시점의 데이터만 추출

    predictions = []
    
    # Walk-Forward 방식으로 7일 예측
    for date in future_dates:
        # 예측 날짜에 맞는 시간 기반 피처 업데이트
        # current_data는 이미 피처가 생성된 상태이므로, 'Close'와 'Volume'만 업데이트하고
        # 시간 기반 피처와 Lag 피처는 예측 시점의 정보를 반영합니다.
        
        # 주의: 이 시점에서는 실제 Close/Volume 컬럼이 존재해야 합니다.
        
        # 1. 새로운 예측 시점 데이터 생성
        new_row = pd.DataFrame(index=[date])
        # 이전 예측값을 현재 종가로 설정
        # predictions 리스트가 비어있다면 (첫 번째 예측), 실제 마지막 종가를 사용합니다.
        new_row['Close'] = predictions[-1] if predictions else last_data['Close'].iloc[-1]
        new_row['Volume'] = last_data['Volume'].iloc[-1] # 거래량은 단순하게 마지막 실제값 유지 (개선 가능)

        # 2. 피처를 다시 생성
        # 예측에 필요한 과거 데이터(last_data)의 최신 60일 데이터에 새로운 예측 날짜를 추가합니다.
        # last_data는 재귀적으로 예측값이 추가되어 성장하고 있으므로, 마지막 60일만 사용합니다.
        temp_df = last_data.iloc[-60:].copy()
        temp_df = pd.concat([temp_df, new_row])
        temp_df = create_features(temp_df).iloc[-1].to_frame().T
        
        # 3. 모델이 기대하는 피처만 추출 및 정리
        X_future = temp_df[feature_columns].fillna(0)
        X_future.columns = feature_columns # 컬럼 순서 및 이름 일치 강제

        # 4. 스케일링 적용
        X_future_scaled = scaler.transform(X_future)
        
        # 5. 예측
        next_price = model.predict(X_future_scaled)[0]
        predictions.append(next_price)
        
        # 6. 다음 예측을 위해 'last_data' 업데이트 (재귀적 예측을 위해)
        # 새로운 행을 생성하고 last_data에 추가합니다.
        last_data = pd.concat([last_data, pd.DataFrame({'Open': next_price, 'High': next_price, 'Low': next_price, 'Close': next_price, 'Volume': new_row['Volume'].iloc[0]}, index=[date])])
        
        # last_data의 컬럼 이름이 원본 yfinance 데이터의 컬럼 이름 구조를 유지하도록 보정
        last_data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if 'Adj Close' not in last_data.columns:
            last_data['Adj Close'] = last_data['Close']

        # 마지막 예측값을 current_data에 반영하여 다음 루프에 사용
        current_data = temp_df 
        current_data['Target'] = next_price # 다음 루프를 위해 Target에 예측값 저장

        
    return pd.Series(predictions, index=future_dates)


# --- Streamlit 메인 앱 ---
st.set_page_config(layout="wide", page_title="LGBM 주가 예측 시스템")

def app():
    st.title("💡 LightGBM 고효율 주가 예측 시스템 (Feat. yfinance)")
    st.markdown("이 시스템은 **LightGBM**과 **시계열 특화 피처 엔지니어링**을 사용하여 과적합을 방지하고 예측 성능을 극대화합니다.")
    st.markdown("---")

    # 1. 종목 선택 및 실행 버튼
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY']
    
    col1, col2, _ = st.columns([1, 1, 3])
    
    with col1:
        selected_ticker = st.selectbox("예측할 종목 선택", TICKERS, key='ticker_select')
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True) 
        run_button = st.button("모델 훈련 및 예측 실행", type="primary")

    st.markdown("---") 

    if run_button:
        with st.spinner(f"⏳ '{selected_ticker}' 데이터 로드 및 피처 생성 중 (훈련 기간: {TRAIN_DAYS}일)..."):
            
            # 2. 데이터 로드 및 피처 생성
            raw_data = load_data(selected_ticker)
            if raw_data is None:
                return

            data_features = create_features(raw_data)
            
            if data_features.empty:
                 st.error("피처 생성 후 데이터가 충분하지 않습니다. 훈련 기간을 늘리거나 다른 종목을 선택하세요.")
                 return

            # 데이터 분할
            train_data = data_features.iloc[:-1]
            
            st.subheader(f"🔍 종목 분석: {selected_ticker} (총 {len(train_data)}일 데이터)")
            
            # 3. 모델 훈련 및 검증
            model, scaler, feature_columns = train_and_validate_model(train_data)
            
            # 4. 미래 예측
            with st.spinner("🔮 미래 7일 예측 중 (Walk-Forward)..."):
                
                last_actual_close = raw_data['Close'].iloc[-1]
                
                # 예측을 위해 충분한 과거 데이터 확보
                # predict_future 내부에서 원본 데이터프레임의 구조를 유지하며 재귀적으로 확장
                last_data_for_prediction = raw_data.iloc[-60:].copy() 
                
                # 예측 함수 호출 시 Target 컬럼 제거
                # feature_columns는 이미 Target이 제거된 상태입니다.
                
                future_predictions_series = predict_future(
                    model, 
                    scaler, 
                    last_data_for_prediction, 
                    feature_columns 
                )
                
                st.subheader(f"📈 {selected_ticker} 주가 예측 결과")
                
                # 5. 결과 시각화
                
                # 과거 및 예측 데이터 병합
                past_prices = raw_data['Close'].iloc[-90:] # 최근 90일의 실제 가격 표시
                
                predicted_df = pd.DataFrame({
                    'Actual': past_prices,
                    'Predicted': np.nan 
                })
                
                future_df = pd.DataFrame({
                    'Actual': np.nan,
                    'Predicted': future_predictions_series
                })
                
                final_df = pd.concat([predicted_df, future_df])
                
                # 차트 생성
                st.line_chart(final_df)
                st.caption(f"마지막 실제 종가: ${last_actual_close:.2f}")

                # 예측 수치 테이블
                st.markdown("##### 🗓️ 향후 7일 예측 종가")
                st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format('${:.2f}'))


if __name__ == "__main__":
    app()
