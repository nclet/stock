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
TRAIN_DAYS = 365 * 3 # 훈련에 사용할 기간 (3년)

# --- LightGBM 모델 하이퍼파라미터 (과적합 방지 최적화) ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,    # 피처 일부만 사용 (과적합 방지)
    'bagging_fraction': 0.8,      # 데이터 일부만 사용 (과적합 방지)
    'bagging_freq': 1,
    'num_leaves': 31,           # 트리의 복잡도 제한 (과적합 방지)
    'max_depth': 7,             # 트리의 깊이 제한 (과적합 방지)
    'lambda_l1': 0.1,           # L1 규제
    'lambda_l2': 0.1,           # L2 규제
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
    # 1, 3, 7, 14, 30일 전 종가, 거래량 등을 피처로 추가
    lags = [1, 3, 7, 14, 30]
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
    # 3. 이동 평균 및 볼륨 지표
    windows = [5, 20, 60]
    for window in windows:
        # 이동 평균 (단기/장기 추세 반영)
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        # 가격 변동성 (표준편차)
        df[f'Vol_{window}'] = df['Close'].rolling(window=window).std()

    # 4. 상대적인 변화율 (차분 피처)
    df['Daily_Change'] = df['Close'].pct_change()
    
    # 5. 타겟 변수 (미래 1일 후의 종가)
    # 주의: 이 예시에서는 단순화하여 1일 후 종가를 예측하도록 설정했습니다.
    # 복잡한 모델에서는 수익률(Return)이나 방향(Binary Classification)을 타겟으로 설정합니다.
    df['Target'] = df['Close'].shift(-1) 

    # 결측치 제거 (Lag Features 때문에 발생하는 초기 결측치)
    df = df.dropna()
    
    return df

# --- 2. 데이터 로드 함수 ---
@st.cache_data
def load_data(ticker):
    """
    YFinance를 사용하여 주가 데이터를 로드합니다.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=TRAIN_DAYS + 100) # 피처 생성 여유분 포함
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
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
    # LightGBM은 일부 특수 문자(예: [, ], :, <, > 등)를 피처 이름에서 허용하지 않습니다.
    sanitized_columns = [
        col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '')
        for col in X.columns
    ]
    X.columns = sanitized_columns
    # ------------------------------------------------------------------
    
    # 스케일링 (선형 모델에는 필수, 부스팅 모델에도 성능 향상에 도움)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    # 시계열 교차 검증 (Time Series Split) 설정
    # 총 3개의 폴드(Fold)를 사용하여 시간 순서대로 검증
    tscv = TimeSeriesSplit(n_splits=3)
    
    rmse_scores = []
    
    st.markdown("##### 🚀 모델 훈련 및 시계열 검증 진행 중...")
    progress_bar = st.progress(0)
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled_df)):
        X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # LightGBM 모델 훈련 (조기 종료 적용)
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=-1)] # 조기 종료로 과적합 방지
        )
        
        # 검증 세트 예측 및 RMSE 계산
        val_predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        rmse_scores.append(rmse)
        
        progress_bar.progress((fold + 1) / 3)
        st.caption(f"Fold {fold+1} 검증 완료. RMSE: {rmse:.4f}")

    avg_rmse = np.mean(rmse_scores)
    st.success(f"✅ 모델 훈련 완료. 평균 검증 RMSE: {avg_rmse:.4f}")
    
    # 마지막 폴드의 모델과 스케일러, 정리된 피처 이름을 반환
    return model, scaler, X.columns

# --- 4. 미래 예측 함수 ---
def predict_future(model, scaler, last_data, feature_columns):
    """
    훈련된 모델을 사용하여 향후 7일간의 주가를 예측합니다.
    """
    
    future_dates = [last_data.index[-1] + datetime.timedelta(days=i) for i in range(1, TARGET_PERIOD + 1)]
    
    # 마지막 데이터를 기반으로 예측을 위한 초기 피처 생성
    current_data = last_data.iloc[-1].to_frame().T
    
    predictions = []
    
    # Walk-Forward 방식으로 7일 예측
    for date in future_dates:
        # 예측 날짜에 맞는 시간 기반 피처 업데이트
        current_data.index = [date]
        current_data['Year'] = date.year
        current_data['Month'] = date.month
        current_data['Day'] = date.day
        current_data['DayOfWeek'] = date.weekday()
        current_data['DayOfYear'] = date.timetuple().tm_yday
        
        # 지연 피처(Lag Features)는 이전 예측값 또는 실제값으로 업데이트되어야 함
        # 여기서는 단순화를 위해 이전 종가/거래량 피처를 마지막 실제값으로 고정하여 사용합니다.
        
        X_future = current_data[feature_columns].fillna(0) # 결측치는 임의로 0 처리 (실제 환경에서는 더 정교한 처리 필요)
        
        # 스케일링 적용
        X_future_scaled = scaler.transform(X_future)
        
        # 예측
        next_price = model.predict(X_future_scaled)[0]
        predictions.append(next_price)
        
        # 다음 예측을 위해 '현재 종가'를 예측값으로 업데이트 (재귀적 예측)
        current_data['Close'] = next_price
        
    return pd.Series(predictions, index=future_dates)


# --- Streamlit 메인 앱 ---
st.set_page_config(layout="wide", page_title="LGBM 주가 예측 시스템")

def app():
    st.title("💡 LightGBM 고효율 주가 예측 시스템 (Feat. yfinance)")
    st.markdown("이 시스템은 **LightGBM**과 **시계열 특화 피처 엔지니어링**을 사용하여 과적합을 방지하고 예측 성능을 극대화합니다.")
    st.markdown("---")

    # 1. 사이드바 설정
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY']
    st.sidebar.header("⚙️ 예측 설정")
    selected_ticker = st.sidebar.selectbox("예측할 종목 선택", TICKERS)
    
    run_button = st.sidebar.button("모델 훈련 및 예측 실행", type="primary")

    if run_button:
        with st.spinner(f"⏳ '{selected_ticker}' 데이터 로드 및 피처 생성 중..."):
            
            # 2. 데이터 로드 및 피처 생성
            raw_data = load_data(selected_ticker)
            if raw_data is None:
                st.error("데이터를 로드할 수 없습니다. 종목 코드나 데이터 가용성을 확인하세요.")
                return

            data_features = create_features(raw_data)
            
            if data_features.empty:
                 st.error("피처 생성 후 데이터가 충분하지 않습니다. 훈련 기간을 늘리거나 다른 종목을 선택하세요.")
                 return

            # 데이터 분할: 마지막 데이터는 예측을 위해 남겨둡니다.
            last_actual_data = data_features.iloc[-1].drop('Target')
            train_data = data_features.iloc[:-1]
            
            st.subheader(f"🔍 종목 분석: {selected_ticker} (총 {len(train_data)}일 데이터)")
            
            # 3. 모델 훈련 및 검증
            model, scaler, feature_columns = train_and_validate_model(train_data)
            
            # 4. 미래 예측
            with st.spinner("🔮 미래 7일 예측 중..."):
                # 예측에 사용할 마지막 실제 데이터 (Target 제거)
                last_actual_close = raw_data['Close'].iloc[-1]
                
                # 예측을 위한 마지막 실제 데이터 프레임 준비
                last_data_for_prediction = raw_data.iloc[-30:].copy() # Lag features 계산을 위해 충분한 과거 데이터 필요
                
                # 현재 시점의 피처를 생성하고 미래 예측 시작
                future_predictions_series = predict_future(
                    model, 
                    scaler, 
                    last_data_for_prediction, 
                    feature_columns.drop('Target', errors='ignore') # Target 피처 제거
                )
                
                st.subheader(f"📈 {selected_ticker} 주가 예측 결과")
                
                # 5. 결과 시각화
                
                # 과거 및 예측 데이터 병합
                past_prices = raw_data['Close'].iloc[-60:] # 최근 60일의 실제 가격만 표시
                
                # 예측 데이터프레임 생성
                predicted_df = pd.DataFrame({
                    'Actual': past_prices,
                    'Predicted': np.nan # 과거 시점은 예측값이 없음
                })
                
                # 예측 시점의 데이터 병합 (미래 예측은 'Predicted' 컬럼에만 값 추가)
                future_df = pd.DataFrame({
                    'Actual': np.nan,
                    'Predicted': future_predictions_series
                })
                
                # 시각화를 위해 모든 데이터를 합칩니다.
                final_df = pd.concat([predicted_df, future_df])
                
                # 차트 생성
                st.line_chart(final_df)
                st.caption(f"마지막 실제 종가: ${last_actual_close:.2f}")

                # 예측 수치 테이블
                st.markdown("##### 🗓️ 향후 7일 예측 종가 (Walk-Forward)")
                st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format('${:.2f}'))


if __name__ == "__main__":
    app()
