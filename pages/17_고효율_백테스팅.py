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
TARGET_PERIOD = 10 # 예측할 미래 일수 (7일 -> 10일로 변경)
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
def create_features(df, is_for_training=True):
    """
    LightGBM 모델 훈련을 위한 시계열 피처를 생성합니다.
    (is_for_training=False일 경우 Target 생성 및 Target 관련 dropna 방지)
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
    
    if is_for_training:
        # 5. 타겟 변수 (미래 1일 후의 종가) - 훈련 시에만 필요
        df['Target'] = df['Close'].shift(-1) 

    # 결측치 제거 (Lag Features, MA 때문에 발생하는 초기 결측치)
    # is_for_training=True일 경우 Target이 NaN인 마지막 행도 제거됩니다.
    df = df.dropna()
    
    return df

# --- 2. 데이터 로드 함수 ---
@st.cache_data(ttl=60*60*4) # 캐시 유지 시간을 4시간으로 설정하여 로딩 속도 개선
def load_data(ticker):
    """
    YFinance를 사용하여 주가 데이터를 로드하고 6개 핵심 컬럼 이름으로 정리합니다.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=TRAIN_DAYS + 100)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        
        # yfinance 멀티 인덱스 문제 방지 및 컬럼 이름 정리 함수
        def sanitize_column_name(col):
            if isinstance(col, tuple):
                name = '_'.join(map(str, col))
            else:
                name = str(col)
            
            # 특수 문자 정리
            return name.replace(' ', '_').replace('.', '').replace(',', '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_')

        data.columns = [sanitize_column_name(col) for col in data.columns.to_list()]
        
        # --- 핵심: 6개 코어 컬럼의 이름을 표준화합니다. ---
        # NOTE: CORE_COL_NAMES는 공백을 포함합니다. (예: 'Adj Close')
        new_columns = {}
        CORE_COL_NAMES = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        for col in data.columns:
            sanitized_col = col.upper() # 대소문자 구분 없이 처리
            
            # 컬럼 이름에 'Open', 'High' 등을 포함하는지 확인하고 표준화합니다.
            if 'OPEN' in sanitized_col:
                new_columns[col] = 'Open'
            elif 'HIGH' in sanitized_col:
                new_columns[col] = 'High'
            elif 'LOW' in sanitized_col:
                new_columns[col] = 'Low'
            elif 'VOLUME' in sanitized_col:
                new_columns[col] = 'Volume'
            # 'Adj Close'를 먼저 처리하여 일반 'Close'와 분리합니다.
            elif 'ADJ_CLOSE' in sanitized_col:
                new_columns[col] = 'Adj Close'
            elif 'CLOSE' in sanitized_col:
                new_columns[col] = 'Close'
        
        # 컬럼 이름 변경 적용
        data = data.rename(columns=new_columns)
        
        # 필수 컬럼이 모두 포함되었는지 확인
        if not all(col in data.columns for col in ['Close', 'Volume']):
            st.error(f"데이터에 'Close' 또는 'Volume' 컬럼이 없어 처리를 계속할 수 없습니다. (처리 후 컬럼: {data.columns.tolist()})")
            return None

        # Adj Close가 없으면 Close로 채워주는 폴백 로직
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        
        # 최종적으로 필요한 6개 컬럼만 남깁니다. (순서도 고정)
        final_cols = [col for col in CORE_COL_NAMES if col in data.columns]
        data = data[final_cols].copy()
        
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
    # NOTE: 훈련 시 피처 이름을 표준화하여(공백 -> 언더바) 저장합니다. 
    # 이 이름(feature_columns)이 예측 시 사용됩니다.
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
    훈련된 모델을 사용하여 향후 TARGET_PERIOD 일간의 주가를 예측합니다.
    """
    
    future_dates = [last_data.index[-1] + datetime.timedelta(days=i) for i in range(1, TARGET_PERIOD + 1)]
    
    predictions = []
    
    # Walk-Forward 방식으로 예측
    for date in future_dates:
        
        # 1. 새로운 예측 시점 데이터 생성
        new_row = pd.DataFrame(index=[date])
        # 이전 예측값을 현재 종가로 설정
        new_row['Close'] = predictions[-1] if predictions else last_data['Close'].iloc[-1]
        new_row['Volume'] = last_data['Volume'].iloc[-1] 
        
        # 나머지 가격 컬럼(Open, High, Low, Adj Close)도 채워줍니다.
        price_cols = ['Open', 'High', 'Low', 'Adj Close']
        for col in price_cols:
             new_row[col] = new_row['Close'].iloc[0]
             
        # 2. 피처를 다시 생성하기 위해 과거 데이터에 새로운 행 추가
        temp_df = last_data.iloc[-60:].copy()
        temp_df = pd.concat([temp_df, new_row])
        
        # 피처 생성 (is_for_training=False: Target 생성 건너뛰기)
        temp_df = create_features(temp_df, is_for_training=False)
        
        # --- 핵심 수정: 예측에 사용되는 피처 데이터의 컬럼 이름 표준화 ---
        # 훈련 시 feature_columns가 'Adj Close'를 'Adj_Close'로 표준화했으므로, 
        # 예측 데이터의 컬럼 이름도 일치시켜야 KeyError가 발생하지 않습니다.
        sanitized_temp_columns = [
            str(col).replace(' ', '_') for col in temp_df.columns
        ]
        temp_df.columns = sanitized_temp_columns
        # -----------------------------------------------------------
        
        # 마지막 행(예측 일자)의 피처만 추출
        X_future_data = temp_df.iloc[-1].to_frame().T
        
        # 3. 모델이 기대하는 피처만 추출 및 정리 
        # feature_columns는 이제 'Adj_Close'와 같은 표준화된 이름을 포함합니다.
        X_future = X_future_data[feature_columns].fillna(0)
        X_future.columns = feature_columns # 컬럼 순서 및 이름 일치 강제

        # 4. 스케일링 적용
        X_future_scaled = scaler.transform(X_future)
        
        # 5. 예측
        next_price = model.predict(X_future_scaled)[0]
        predictions.append(next_price)
        
        # 6. 다음 예측을 위해 'last_data' 업데이트 (재귀적 예측을 위해)
        last_data = pd.concat([last_data, new_row])
        
    return pd.Series(predictions, index=future_dates)


# --- Streamlit 메인 앱 ---
st.set_page_config(layout="wide", page_title="LGBM 주가 예측 시스템")

def app():
    st.title("💡 LightGBM 고효율 주가 예측 시스템 (Feat. yfinance)")
    st.markdown("이 시스템은 **LightGBM**과 **시계열 특화 피처 엔지니어링**을 사용하여 과적합을 방지하고 예측 성능을 극대화합니다.")
    st.markdown("---")

    # 1. 종목 선택 및 실행 버튼
    # 사용자가 직접 원하는 종목의 티커를 입력하도록 변경
    col1, col2, _ = st.columns([1, 1, 3])
    
    with col1:
        # 텍스트 입력 필드를 사용하여 모든 미국 상장 종목에 대한 유연성을 제공
        selected_ticker = st.text_input("예측할 미국 상장 종목 티커 입력 (예: AAPL, AMD, NVDA)", value='AAPL', key='ticker_input')
        # 입력된 티커를 대문자로 변환하여 사용
        selected_ticker = selected_ticker.upper().strip()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True) 
        run_button = st.button("모델 훈련 및 예측 실행", type="primary")

    st.markdown("---") 
    
    # 입력 검증
    if run_button and not selected_ticker:
        st.warning("예측할 종목의 티커를 입력해주세요.")
        return

    if run_button:
        with st.spinner(f"⏳ '{selected_ticker}' 데이터 로드 및 피처 생성 중 (훈련 기간: {TRAIN_DAYS}일)..."):
            
            # 2. 데이터 로드 및 피처 생성
            raw_data = load_data(selected_ticker)
            if raw_data is None:
                return

            # 훈련 데이터를 위한 피처 생성 (Target 포함)
            data_features = create_features(raw_data, is_for_training=True)
            
            if data_features.empty:
                 st.error("피처 생성 후 데이터가 충분하지 않습니다. 훈련 기간을 늘리거나 다른 종목을 선택하세요.")
                 return

            # 데이터 분할 (create_features의 dropna()로 인해 마지막 Target=NaN 행은 이미 제거됨)
            train_data = data_features
            
            st.subheader(f"🔍 종목 분석: {selected_ticker} (총 {len(train_data)}일 데이터)")
            
            # 3. 모델 훈련 및 검증
            model, scaler, feature_columns = train_and_validate_model(train_data)
            
            # 4. 미래 예측
            with st.spinner(f"🔮 미래 {TARGET_PERIOD}일 예측 중 (Walk-Forward)..."): # 예측 기간 변수 사용
                
                last_actual_close = raw_data['Close'].iloc[-1]
                
                # 예측을 위해 충분한 과거 데이터 확보
                # raw_data는 이미 6개 컬럼으로 정제되었으므로 그대로 사용합니다.
                last_data_for_prediction = raw_data.iloc[-100:].copy() 
                
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
                st.markdown(f"##### 🗓️ 향후 {TARGET_PERIOD}일 예측 종가") # 예측 기간 변수 사용
                st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format('${:.2f}'))


if __name__ == "__main__":
    app()



# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import datetime
# import lightgbm as lgb
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler

# # --- 상수 정의 ---
# TARGET_PERIOD = 7 # 예측할 미래 일수
# TRAIN_DAYS = 365 # 훈련에 사용할 기간 (1년으로 단축)

# # --- LightGBM 모델 하이퍼파라미터 (과적합 방지 최적화) ---
# LGBM_PARAMS = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'n_estimators': 1200,       # 훈련 기간 단축에 맞춰 Estimator 증가
#     'learning_rate': 0.02,      # 과적합 방지를 위해 학습률 미세 조정
#     'feature_fraction': 0.75,   # 피처 일부만 사용 (과적합 방지)
#     'bagging_fraction': 0.75,   # 데이터 일부만 사용 (과적합 방지)
#     'bagging_freq': 1,
#     'num_leaves': 31,           # 트리의 복잡도 제한 (과적합 방지)
#     'max_depth': 8,             # 트리의 깊이 제한
#     'lambda_l1': 0.2,           # L1 규제 강화
#     'lambda_l2': 0.2,           # L2 규제 강화
#     'verbose': -1,              # 로그 출력 끔
#     'n_jobs': -1,
#     'seed': 42
# }

# # --- 1. 피처 엔지니어링 함수 ---
# def create_features(df, is_for_training=True):
#     """
#     LightGBM 모델 훈련을 위한 시계열 피처를 생성합니다.
#     (is_for_training=False일 경우 Target 생성 및 Target 관련 dropna 방지)
#     """
#     df = df.copy()

#     # 1. 시간 기반 피처 (주기성 반영)
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['DayOfWeek'] = df.index.dayofweek
#     df['DayOfYear'] = df.index.dayofyear
    
#     # 2. 지연 피처 (Lag Features)
#     lags = [1, 3, 7, 14, 30]
#     for lag in lags:
#         df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
#         df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
#     # 3. 이동 평균 및 볼륨 지표
#     windows = [5, 20, 60]
#     for window in windows:
#         df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
#         df[f'Vol_{window}'] = df['Close'].rolling(window=window).std()

#     # 4. 상대적인 변화율 (차분 피처)
#     df['Daily_Change'] = df['Close'].pct_change()
    
#     if is_for_training:
#         # 5. 타겟 변수 (미래 1일 후의 종가) - 훈련 시에만 필요
#         df['Target'] = df['Close'].shift(-1) 

#     # 결측치 제거 (Lag Features, MA 때문에 발생하는 초기 결측치)
#     # is_for_training=True일 경우 Target이 NaN인 마지막 행도 제거됩니다.
#     df = df.dropna()
    
#     return df

# # --- 2. 데이터 로드 함수 ---
# @st.cache_data(ttl=60*60*4) # 캐시 유지 시간을 4시간으로 설정하여 로딩 속도 개선
# def load_data(ticker):
#     """
#     YFinance를 사용하여 주가 데이터를 로드하고 6개 핵심 컬럼 이름으로 정리합니다.
#     """
#     end_date = datetime.date.today()
#     start_date = end_date - datetime.timedelta(days=TRAIN_DAYS + 100)
    
#     try:
#         data = yf.download(ticker, start=start_date, end=end_date)
#         if data.empty:
#             return None
        
#         # yfinance 멀티 인덱스 문제 방지 및 컬럼 이름 정리 함수
#         def sanitize_column_name(col):
#             if isinstance(col, tuple):
#                 name = '_'.join(map(str, col))
#             else:
#                 name = str(col)
            
#             # 특수 문자 정리
#             return name.replace(' ', '_').replace('.', '').replace(',', '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_')

#         data.columns = [sanitize_column_name(col) for col in data.columns.to_list()]
        
#         # --- 핵심: 6개 코어 컬럼의 이름을 표준화합니다. ---
#         # NOTE: CORE_COL_NAMES는 공백을 포함합니다. (예: 'Adj Close')
#         new_columns = {}
#         CORE_COL_NAMES = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
#         for col in data.columns:
#             sanitized_col = col.upper() # 대소문자 구분 없이 처리
            
#             # 컬럼 이름에 'Open', 'High' 등을 포함하는지 확인하고 표준화합니다.
#             if 'OPEN' in sanitized_col:
#                 new_columns[col] = 'Open'
#             elif 'HIGH' in sanitized_col:
#                 new_columns[col] = 'High'
#             elif 'LOW' in sanitized_col:
#                 new_columns[col] = 'Low'
#             elif 'VOLUME' in sanitized_col:
#                 new_columns[col] = 'Volume'
#             # 'Adj Close'를 먼저 처리하여 일반 'Close'와 분리합니다.
#             elif 'ADJ_CLOSE' in sanitized_col:
#                 new_columns[col] = 'Adj Close'
#             elif 'CLOSE' in sanitized_col:
#                 new_columns[col] = 'Close'
        
#         # 컬럼 이름 변경 적용
#         data = data.rename(columns=new_columns)
        
#         # 필수 컬럼이 모두 포함되었는지 확인
#         if not all(col in data.columns for col in ['Close', 'Volume']):
#             st.error(f"데이터에 'Close' 또는 'Volume' 컬럼이 없어 처리를 계속할 수 없습니다. (처리 후 컬럼: {data.columns.tolist()})")
#             return None

#         # Adj Close가 없으면 Close로 채워주는 폴백 로직
#         if 'Adj Close' not in data.columns:
#             data['Adj Close'] = data['Close']
        
#         # 최종적으로 필요한 6개 컬럼만 남깁니다. (순서도 고정)
#         final_cols = [col for col in CORE_COL_NAMES if col in data.columns]
#         data = data[final_cols].copy()
        
#         return data.dropna()
#     except Exception as e:
#         st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
#         return None

# # --- 3. 모델 훈련 및 검증 함수 ---
# def train_and_validate_model(data_features):
#     """
#     시계열 분할을 이용해 LightGBM 모델을 훈련 및 검증하고 결과를 반환합니다.
#     """
    
#     # Feature와 Target 분리
#     X = data_features.drop('Target', axis=1)
#     y = data_features['Target']
    
#     # --- LightGBM 오류 방지를 위한 피처 이름 정리 (Sanitization) ---
#     # NOTE: 훈련 시 피처 이름을 표준화하여(공백 -> 언더바) 저장합니다. 
#     # 이 이름(feature_columns)이 예측 시 사용됩니다.
#     sanitized_columns = [
#         str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '')
#         for col in X.columns
#     ]
#     X.columns = sanitized_columns # 컬럼 이름 업데이트
#     # ------------------------------------------------------------------
    
#     # 스케일링
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
#     # 시계열 교차 검증 (Time Series Split) 설정
#     tscv = TimeSeriesSplit(n_splits=3)
    
#     rmse_scores = []
    
#     st.markdown("##### 🚀 모델 훈련 및 시계열 검증 진행 중...")
#     progress_bar = st.progress(0)
    
#     final_model = None
    
#     for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled_df)):
#         X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
#         y_train, y_val = y.iloc[train_index], y.iloc[val_index]

#         # LightGBM 모델 훈련 (조기 종료 적용)
#         model = lgb.LGBMRegressor(**LGBM_PARAMS)
#         model.fit(
#             X_train, y_train,
#             eval_set=[(X_val, y_val)],
#             eval_metric='rmse',
#             callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=-1)] # 조기 종료 라운드 조정
#         )
        
#         # 검증 세트 예측 및 RMSE 계산
#         val_predictions = model.predict(X_val)
#         rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
#         rmse_scores.append(rmse)
        
#         progress_bar.progress((fold + 1) / 3)
#         st.caption(f"Fold {fold+1} 검증 완료. RMSE: {rmse:.4f}")
#         final_model = model # 마지막 모델 저장

#     avg_rmse = np.mean(rmse_scores)
#     st.success(f"✅ 모델 훈련 완료. 평균 검증 RMSE: {avg_rmse:.4f}")
    
#     return final_model, scaler, X.columns

# # --- 4. 미래 예측 함수 ---
# def predict_future(model, scaler, last_data, feature_columns):
#     """
#     훈련된 모델을 사용하여 향후 7일간의 주가를 예측합니다.
#     """
    
#     future_dates = [last_data.index[-1] + datetime.timedelta(days=i) for i in range(1, TARGET_PERIOD + 1)]
    
#     predictions = []
    
#     # Walk-Forward 방식으로 7일 예측
#     for date in future_dates:
        
#         # 1. 새로운 예측 시점 데이터 생성
#         new_row = pd.DataFrame(index=[date])
#         # 이전 예측값을 현재 종가로 설정
#         new_row['Close'] = predictions[-1] if predictions else last_data['Close'].iloc[-1]
#         new_row['Volume'] = last_data['Volume'].iloc[-1] 
        
#         # 나머지 가격 컬럼(Open, High, Low, Adj Close)도 채워줍니다.
#         price_cols = ['Open', 'High', 'Low', 'Adj Close']
#         for col in price_cols:
#              new_row[col] = new_row['Close'].iloc[0]
             
#         # 2. 피처를 다시 생성하기 위해 과거 데이터에 새로운 행 추가
#         temp_df = last_data.iloc[-60:].copy()
#         temp_df = pd.concat([temp_df, new_row])
        
#         # 피처 생성 (is_for_training=False: Target 생성 건너뛰기)
#         temp_df = create_features(temp_df, is_for_training=False)
        
#         # --- 핵심 수정: 예측에 사용되는 피처 데이터의 컬럼 이름 표준화 ---
#         # 훈련 시 feature_columns가 'Adj Close'를 'Adj_Close'로 표준화했으므로, 
#         # 예측 데이터의 컬럼 이름도 일치시켜야 KeyError가 발생하지 않습니다.
#         sanitized_temp_columns = [
#             str(col).replace(' ', '_') for col in temp_df.columns
#         ]
#         temp_df.columns = sanitized_temp_columns
#         # -----------------------------------------------------------
        
#         # 마지막 행(예측 일자)의 피처만 추출
#         X_future_data = temp_df.iloc[-1].to_frame().T
        
#         # 3. 모델이 기대하는 피처만 추출 및 정리 
#         # feature_columns는 이제 'Adj_Close'와 같은 표준화된 이름을 포함합니다.
#         X_future = X_future_data[feature_columns].fillna(0)
#         X_future.columns = feature_columns # 컬럼 순서 및 이름 일치 강제

#         # 4. 스케일링 적용
#         X_future_scaled = scaler.transform(X_future)
        
#         # 5. 예측
#         next_price = model.predict(X_future_scaled)[0]
#         predictions.append(next_price)
        
#         # 6. 다음 예측을 위해 'last_data' 업데이트 (재귀적 예측을 위해)
#         last_data = pd.concat([last_data, new_row])
        
#     return pd.Series(predictions, index=future_dates)


# # --- Streamlit 메인 앱 ---
# st.set_page_config(layout="wide", page_title="LGBM 주가 예측 시스템")

# def app():
#     st.title("💡 LightGBM 고효율 주가 예측 시스템 (Feat. yfinance)")
#     st.markdown("이 시스템은 **LightGBM**과 **시계열 특화 피처 엔지니어링**을 사용하여 과적합을 방지하고 예측 성능을 극대화합니다.")
#     st.markdown("---")

#     # 1. 종목 선택 및 실행 버튼
#     TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY']
    
#     col1, col2, _ = st.columns([1, 1, 3])
    
#     with col1:
#         selected_ticker = st.selectbox("예측할 종목 선택", TICKERS, key='ticker_select')
    
#     with col2:
#         st.markdown("<br>", unsafe_allow_html=True) 
#         run_button = st.button("모델 훈련 및 예측 실행", type="primary")

#     st.markdown("---") 

#     if run_button:
#         with st.spinner(f"⏳ '{selected_ticker}' 데이터 로드 및 피처 생성 중 (훈련 기간: {TRAIN_DAYS}일)..."):
            
#             # 2. 데이터 로드 및 피처 생성
#             raw_data = load_data(selected_ticker)
#             if raw_data is None:
#                 return

#             # 훈련 데이터를 위한 피처 생성 (Target 포함)
#             data_features = create_features(raw_data, is_for_training=True)
            
#             if data_features.empty:
#                  st.error("피처 생성 후 데이터가 충분하지 않습니다. 훈련 기간을 늘리거나 다른 종목을 선택하세요.")
#                  return

#             # 데이터 분할 (create_features의 dropna()로 인해 마지막 Target=NaN 행은 이미 제거됨)
#             train_data = data_features
            
#             st.subheader(f"🔍 종목 분석: {selected_ticker} (총 {len(train_data)}일 데이터)")
            
#             # 3. 모델 훈련 및 검증
#             model, scaler, feature_columns = train_and_validate_model(train_data)
            
#             # 4. 미래 예측
#             with st.spinner("🔮 미래 7일 예측 중 (Walk-Forward)..."):
                
#                 last_actual_close = raw_data['Close'].iloc[-1]
                
#                 # 예측을 위해 충분한 과거 데이터 확보
#                 # raw_data는 이미 6개 컬럼으로 정제되었으므로 그대로 사용합니다.
#                 last_data_for_prediction = raw_data.iloc[-100:].copy() 
                
#                 # feature_columns는 이미 Target이 제거된 상태입니다.
                
#                 future_predictions_series = predict_future(
#                     model, 
#                     scaler, 
#                     last_data_for_prediction, 
#                     feature_columns 
#                 )
                
#                 st.subheader(f"📈 {selected_ticker} 주가 예측 결과")
                
#                 # 5. 결과 시각화
                
#                 # 과거 및 예측 데이터 병합
#                 past_prices = raw_data['Close'].iloc[-90:] # 최근 90일의 실제 가격 표시
                
#                 predicted_df = pd.DataFrame({
#                     'Actual': past_prices,
#                     'Predicted': np.nan 
#                 })
                
#                 future_df = pd.DataFrame({
#                     'Actual': np.nan,
#                     'Predicted': future_predictions_series
#                 })
                
#                 final_df = pd.concat([predicted_df, future_df])
                
#                 # 차트 생성
#                 st.line_chart(final_df)
#                 st.caption(f"마지막 실제 종가: ${last_actual_close:.2f}")

#                 # 예측 수치 테이블
#                 st.markdown("##### 🗓️ 향후 7일 예측 종가")
#                 st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format('${:.2f}'))


# if __name__ == "__main__":
#     app()
