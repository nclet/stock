import streamlit as st
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- 데이터 로딩 라이브러리 및 API ---
import FinanceDataReader as fdr
import pyupbit
import requests
from json.decoder import JSONDecodeError

# --- 상수 정의 ---
TARGET_PERIOD = 10 # 예측할 미래 일수
TRAIN_DAYS = 1825 # 훈련에 사용할 기간 (약 5년)

# 시장 매핑
MARKET_MAPPING = {
    "KRX": "한국 주식 (KRX)",
    "NASDAQ": "미국 증시 (NASDAQ)",
    "COIN": "코인 (Upbit)"
}

# --- LightGBM 모델 하이퍼파라미터 ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1200,
    'learning_rate': 0.02,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 1,
    'num_leaves': 31,
    'max_depth': 8,
    'lambda_l1': 0.2,
    'lambda_l2': 0.2,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# --------------------------
# 1. 멀티 마켓 종목 목록 로딩 함수
# --------------------------

@st.cache_data(ttl=60*60*24) # 24시간 캐시
def get_stock_listing(market_name):
    """FinanceDataReader에서 주식 종목 전체 목록을 가져옵니다 (KRX 또는 NASDAQ)."""
    if market_name == 'KRX':
        market_code = 'KRX'
    elif market_name == 'NASDAQ':
        market_code = 'NASDAQ'
    else:
        return pd.DataFrame()
        
    try:
        df = fdr.StockListing(market_code)
        
        if 'Code' not in df.columns and 'Symbol' in df.columns:
            df.rename(columns={'Symbol': 'Code'}, inplace=True)
            
        if 'Code' not in df.columns or df.empty:
            st.error(f"데이터에 'Code' 또는 'Symbol' 열이 없습니다. ({market_name})")
            return pd.DataFrame()

        df['Code'] = df['Code'].astype(str)
        # 종목명과 티커를 결합하여 레이블 생성
        name_col = 'Name' if 'Name' in df.columns else 'Symbol'
        df['label'] = df[name_col].astype(str) + ' (' + df['Code'] + ')'
        return df
    except Exception as e:
        st.error(f"{market_name} 종목 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=60*60*24) # 24시간 캐시
def get_coin_listing():
    """pyupbit에서 원화(KRW) 코인 목록을 가져오고 한글명을 매핑합니다."""
    try:
        url = "https://api.upbit.com/v1/market/all"
        response = requests.get(url, params={'isDetails': 'false'})
        response.raise_for_status()
        all_markets = response.json()
        
        krw_markets = [market for market in all_markets if market['market'].startswith('KRW-')]
        df_coin = pd.DataFrame(krw_markets)
        df_coin.rename(columns={'market': 'Code', 'korean_name': 'Name'}, inplace=True)
        
        # 레이블을 '한글명 (영문티커)' 형식으로 생성
        df_coin['label'] = df_coin['Name'].astype(str) + ' (' + df_coin['Code'].str.replace('KRW-', '') + ')'
        
        return df_coin
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Upbit API 연결 오류: {e}")
        return pd.DataFrame()
    except JSONDecodeError as e:
        st.error(f"❌ Upbit API 응답 파싱 오류: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"코인 리스트를 가져오는 중 예상치 못한 오류가 발생했습니다: {e}")
        return pd.DataFrame()


# --------------------------
# 2. 피처 엔지니어링 함수 (동일)
# --------------------------
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

# --------------------------
# 3. 멀티 마켓 데이터 로드 함수 (핵심 변경)
# --------------------------
@st.cache_data(ttl=60*60*4) # 캐시 유지 시간을 4시간으로 설정
def load_data(ticker, market):
    """
    선택된 시장에 따라 주식 또는 코인 데이터를 가져오고 표준화합니다.
    """
    end_date = datetime.date.today()
    # 충분한 과거 데이터 확보를 위해 훈련 기간 + 100일
    start_date = end_date - datetime.timedelta(days=TRAIN_DAYS + 100) 
    
    data = None
    
    try:
        if market in ['KRX', 'NASDAQ']:
            # FinanceDataReader 사용
            data = fdr.DataReader(ticker, start_date, end_date)
            data.index.name = 'Date'
            
            # fdr 데이터의 컬럼 이름을 표준화합니다.
            # (KRX는 'Close'와 'Adj Close'가 동일할 수 있음)
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                st.error(f"데이터에 'Close' 또는 'Volume' 컬럼이 없어 처리를 계속할 수 없습니다.")
                return None
            
            # LightGBM의 피처 생성을 위해 'Adj Close'를 필수적으로 확보합니다.
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
            
            # 최종 컬럼 순서 정리
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        elif market == 'COIN':
            # Pyupbit 사용
            # pyupbit는 count 기반이므로, 날짜 차이를 계산하여 count를 설정합니다.
            days_diff = (end_date - start_date).days
            # 일봉만 사용하므로 days_diff + 1
            count = days_diff + 1
            
            # ticker는 'KRW-BTC'와 같은 형태로 전달됩니다.
            df_coin = pyupbit.get_ohlcv(ticker=ticker, interval='day', count=count)
            
            if df_coin is None or df_coin.empty:
                st.warning(f"오류: [{ticker}] 코인에 대한 데이터를 찾을 수 없습니다.")
                return None
                
            # 컬럼 이름 표준화
            df_coin.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'value']
            df_coin.index.name = 'Date'
            
            # 코인은 Adjusted Close 개념이 없으므로 Close 값을 사용합니다.
            df_coin['Adj Close'] = df_coin['Close']
            data = df_coin[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        if data is None or data.empty:
            return None

        return data.dropna()
        
    except Exception as e:
        st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# --------------------------
# 4. 모델 훈련 및 예측 함수 (동일)
# --------------------------
def train_and_validate_model(data_features):
    """
    시계열 분할을 이용해 LightGBM 모델을 훈련 및 검증하고 결과를 반환합니다.
    """
    
    # Feature와 Target 분리
    X = data_features.drop('Target', axis=1)
    y = data_features['Target']
    
    # --- LightGBM 오류 방지를 위한 피처 이름 정리 (Sanitization) ---
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
        # 피처 생성을 위해 필요한 최소 60일 데이터에 새로운 행 추가
        temp_df = last_data.iloc[-60:].copy()
        temp_df = pd.concat([temp_df, new_row])
        
        # 피처 생성 (is_for_training=False: Target 생성 건너뛰기)
        temp_df = create_features(temp_df, is_for_training=False)
        
        # --- 핵심: 예측에 사용되는 피처 데이터의 컬럼 이름 표준화 ---
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


# --------------------------
# 5. Streamlit 메인 앱
# --------------------------
st.set_page_config(layout="wide", page_title="LGBM 멀티 자산 예측 시스템")

def app():
    st.title("💡 LightGBM 멀티 자산 예측 시스템")
    st.markdown("이 시스템은 **LightGBM**을 사용하여 **한국 주식, 미국 주식, 코인**의 주가/가격 흐름을 예측합니다.")
    st.markdown("---")

    # 1. 시장 및 종목 선택
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_market_name = st.selectbox(
            "📊 예측할 자산 시장 선택",
            list(MARKET_MAPPING.values()),
            key='market_select'
        )
    
    market_key = [k for k, v in MARKET_MAPPING.items() if v == selected_market_name][0]

    with col2:
        # 2. 종목 목록 동적 로딩
        stock_list_df = pd.DataFrame()
        default_ticker = ""

        if market_key == 'KRX':
            stock_list_df = get_stock_listing('KRX')
            default_ticker = '005930' # 삼성전자
            
        elif market_key == 'NASDAQ':
            stock_list_df = get_stock_listing('NASDAQ')
            default_ticker = 'AAPL' # 애플
            
        elif market_key == 'COIN':
            stock_list_df = get_coin_listing()
            default_ticker = 'KRW-BTC' # 비트코인
        
        if not stock_list_df.empty:
            # Dropdown에 표시할 레이블 리스트
            options = stock_list_df['label'].tolist()
            # 기본 선택값을 찾습니다.
            try:
                default_index = options.index(stock_list_df[stock_list_df['Code'] == default_ticker]['label'].iloc[0])
            except:
                default_index = 0
                
            selected_label = st.selectbox(
                f"🏷️ 예측할 {selected_market_name} 종목/코인 선택", 
                options, 
                index=default_index, 
                key='ticker_label_select'
            )
            
            # 레이블에서 실제 티커 코드를 추출합니다.
            selected_ticker = stock_list_df[stock_list_df['label'] == selected_label]['Code'].iloc[0].upper().strip()
            
        else:
            st.warning("선택한 시장의 종목 목록을 불러올 수 없습니다.")
            selected_ticker = ""
    
    st.markdown("---")
    
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 1, 3])
    with col_btn_left:
        run_button = st.button("모델 훈련 및 예측 실행", type="primary", use_container_width=True)

    # 입력 검증
    if run_button and not selected_ticker:
        st.warning("예측할 종목의 티커를 선택해주세요.")
        return

    if run_button:
        # 사용자가 선택한 시장의 실제 키를 전달
        current_market = market_key 
        
        with st.spinner(f"⏳ '{selected_ticker}' ({current_market}) 데이터 로드 및 피처 생성 중 (훈련 기간: {TRAIN_DAYS}일)..."):
            
            # 2. 데이터 로드 및 피처 생성
            raw_data = load_data(selected_ticker, current_market)
            if raw_data is None:
                return

            # 훈련 데이터를 위한 피처 생성 (Target 포함)
            data_features = create_features(raw_data, is_for_training=True)
            
            if data_features.empty:
                st.error("피처 생성 후 데이터가 충분하지 않습니다. 훈련 기간을 늘리거나 다른 종목을 선택하세요.")
                return

            train_data = data_features
            
            st.subheader(f"🔍 종목 분석: {selected_label} (총 {len(train_data)}일 데이터)")
            
            # 3. 모델 훈련 및 검증
            model, scaler, feature_columns = train_and_validate_model(train_data)
            
            # 4. 미래 예측
            with st.spinner(f"🔮 미래 {TARGET_PERIOD}일 예측 중 (Walk-Forward)..."):
                
                last_actual_close = raw_data['Close'].iloc[-1]
                
                # 예측을 위해 충분한 과거 데이터 확보
                last_data_for_prediction = raw_data.iloc[-100:].copy() 
                
                future_predictions_series = predict_future(
                    model, 
                    scaler, 
                    last_data_for_prediction, 
                    feature_columns 
                )
                
                st.subheader(f"📈 {selected_label} 가격 예측 결과")
                
                # 5. 결과 시각화
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
                
                # 통화 단위 설정 (KRW 또는 USD)
                currency = "원" if current_market in ['KRX', 'COIN'] else "$"
                st.caption(f"마지막 실제 종가: {currency}{last_actual_close:,.2f}")

                # 예측 수치 테이블
                st.markdown(f"##### 🗓️ 향후 {TARGET_PERIOD}일 예측 종가")
                st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format(f'{currency}{{:.2f}}'))


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
# TARGET_PERIOD = 10 # 예측할 미래 일수
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
#     훈련된 모델을 사용하여 향후 TARGET_PERIOD 일간의 주가를 예측합니다.
#     """
    
#     future_dates = [last_data.index[-1] + datetime.timedelta(days=i) for i in range(1, TARGET_PERIOD + 1)]
    
#     predictions = []
    
#     # Walk-Forward 방식으로 예측
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
#     # 나스닥 및 S&P 500을 포괄하는 주요 종목과 섹터별 대형주 목록으로 확장
#     TICKERS = [
#         '^GSPC', '^IXIC', 'SPY', 'QQQ', 'DIA', 'VIXY', # 주요 지수 및 ETF
#         'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', # 기술 대형주
#         'BRK-B', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', # 금융주
#         'JNJ', 'PFE', 'MRK', 'UNH', 'ABBV', 'LLY', # 헬스케어/제약
#         'XOM', 'CVX', 'SHEL', 'DHR', # 에너지
#         'WMT', 'COST', 'HD', 'LOW', # 소매/소비재
#         'KO', 'PEP', 'MCD', 'NKE', 'DIS', 'NFLX', # 소비재/서비스
#         'T', 'VZ', 'CMCSA', # 통신
#         'ADBE', 'CRM', 'AMD', 'QCOM', 'INTC', 'CSCO', # 반도체/소프트웨어
#         'CAT', 'GE', 'DE', # 산업재
#         'RYAAY', 'BABA', 'TCEHY', 'TSM', # 글로벌/기타 대형주
#         'PLTR', 'SNOW', 'RBLX', 'DOCU' # 성장주/소프트웨어 예시
#     ]
    
#     col1, col2, _ = st.columns([1, 1, 3])
    
#     with col1:
#         # 확장된 드롭다운 메뉴 사용
#         selected_ticker = st.selectbox(
#             "예측할 미국 상장 종목 선택 (S&P 500 및 나스닥 대표 종목 포함)", 
#             TICKERS, 
#             key='ticker_select'
#         )
#         # S&P500의 경우 티커가 '^GSPC'로 치환되므로, 이제 리스트에 직접 포함되어 수정 불필요
#         selected_ticker = selected_ticker.upper().strip()
    
#     with col2:
#         st.markdown("<br>", unsafe_allow_html=True) 
#         run_button = st.button("모델 훈련 및 예측 실행", type="primary")

#     st.markdown("---") 
    
#     # 입력 검증
#     if run_button and not selected_ticker:
#         st.warning("예측할 종목의 티커를 선택해주세요.")
#         return

#     if run_button:
#         with st.spinner(f"⏳ '{selected_ticker}' 데이터 로드 및 피처 생성 중 (훈련 기간: {TRAIN_DAYS}일)..."):
            
#             # 2. 데이터 로드 및 피처 생성
#             raw_data = load_data(selected_ticker)
#             if raw_data is None:
#                 # load_data에서 이미 오류 메시지를 출력했습니다.
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
#             with st.spinner(f"🔮 미래 {TARGET_PERIOD}일 예측 중 (Walk-Forward)..."): # 예측 기간 변수 사용
                
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
#                 st.markdown(f"##### 🗓️ 향후 {TARGET_PERIOD}일 예측 종가") # 예측 기간 변수 사용
#                 st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format('${:.2f}'))


# if __name__ == "__main__":
#     app()





# # import streamlit as st
# # import yfinance as yf
# # import pandas as pd
# # import numpy as np
# # import datetime
# # import lightgbm as lgb
# # from sklearn.model_selection import TimeSeriesSplit
# # from sklearn.metrics import mean_squared_error
# # from sklearn.preprocessing import StandardScaler

# # # --- 상수 정의 ---
# # TARGET_PERIOD = 7 # 예측할 미래 일수
# # TRAIN_DAYS = 365 # 훈련에 사용할 기간 (1년으로 단축)

# # # --- LightGBM 모델 하이퍼파라미터 (과적합 방지 최적화) ---
# # LGBM_PARAMS = {
# #     'objective': 'regression',
# #     'metric': 'rmse',
# #     'n_estimators': 1200,       # 훈련 기간 단축에 맞춰 Estimator 증가
# #     'learning_rate': 0.02,      # 과적합 방지를 위해 학습률 미세 조정
# #     'feature_fraction': 0.75,   # 피처 일부만 사용 (과적합 방지)
# #     'bagging_fraction': 0.75,   # 데이터 일부만 사용 (과적합 방지)
# #     'bagging_freq': 1,
# #     'num_leaves': 31,           # 트리의 복잡도 제한 (과적합 방지)
# #     'max_depth': 8,             # 트리의 깊이 제한
# #     'lambda_l1': 0.2,           # L1 규제 강화
# #     'lambda_l2': 0.2,           # L2 규제 강화
# #     'verbose': -1,              # 로그 출력 끔
# #     'n_jobs': -1,
# #     'seed': 42
# # }

# # # --- 1. 피처 엔지니어링 함수 ---
# # def create_features(df, is_for_training=True):
# #     """
# #     LightGBM 모델 훈련을 위한 시계열 피처를 생성합니다.
# #     (is_for_training=False일 경우 Target 생성 및 Target 관련 dropna 방지)
# #     """
# #     df = df.copy()

# #     # 1. 시간 기반 피처 (주기성 반영)
# #     df['Year'] = df.index.year
# #     df['Month'] = df.index.month
# #     df['Day'] = df.index.day
# #     df['DayOfWeek'] = df.index.dayofweek
# #     df['DayOfYear'] = df.index.dayofyear
    
# #     # 2. 지연 피처 (Lag Features)
# #     lags = [1, 3, 7, 14, 30]
# #     for lag in lags:
# #         df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
# #         df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
# #     # 3. 이동 평균 및 볼륨 지표
# #     windows = [5, 20, 60]
# #     for window in windows:
# #         df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
# #         df[f'Vol_{window}'] = df['Close'].rolling(window=window).std()

# #     # 4. 상대적인 변화율 (차분 피처)
# #     df['Daily_Change'] = df['Close'].pct_change()
    
# #     if is_for_training:
# #         # 5. 타겟 변수 (미래 1일 후의 종가) - 훈련 시에만 필요
# #         df['Target'] = df['Close'].shift(-1) 

# #     # 결측치 제거 (Lag Features, MA 때문에 발생하는 초기 결측치)
# #     # is_for_training=True일 경우 Target이 NaN인 마지막 행도 제거됩니다.
# #     df = df.dropna()
    
# #     return df

# # # --- 2. 데이터 로드 함수 ---
# # @st.cache_data(ttl=60*60*4) # 캐시 유지 시간을 4시간으로 설정하여 로딩 속도 개선
# # def load_data(ticker):
# #     """
# #     YFinance를 사용하여 주가 데이터를 로드하고 6개 핵심 컬럼 이름으로 정리합니다.
# #     """
# #     end_date = datetime.date.today()
# #     start_date = end_date - datetime.timedelta(days=TRAIN_DAYS + 100)
    
# #     try:
# #         data = yf.download(ticker, start=start_date, end=end_date)
# #         if data.empty:
# #             return None
        
# #         # yfinance 멀티 인덱스 문제 방지 및 컬럼 이름 정리 함수
# #         def sanitize_column_name(col):
# #             if isinstance(col, tuple):
# #                 name = '_'.join(map(str, col))
# #             else:
# #                 name = str(col)
            
# #             # 특수 문자 정리
# #             return name.replace(' ', '_').replace('.', '').replace(',', '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_')

# #         data.columns = [sanitize_column_name(col) for col in data.columns.to_list()]
        
# #         # --- 핵심: 6개 코어 컬럼의 이름을 표준화합니다. ---
# #         # NOTE: CORE_COL_NAMES는 공백을 포함합니다. (예: 'Adj Close')
# #         new_columns = {}
# #         CORE_COL_NAMES = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
# #         for col in data.columns:
# #             sanitized_col = col.upper() # 대소문자 구분 없이 처리
            
# #             # 컬럼 이름에 'Open', 'High' 등을 포함하는지 확인하고 표준화합니다.
# #             if 'OPEN' in sanitized_col:
# #                 new_columns[col] = 'Open'
# #             elif 'HIGH' in sanitized_col:
# #                 new_columns[col] = 'High'
# #             elif 'LOW' in sanitized_col:
# #                 new_columns[col] = 'Low'
# #             elif 'VOLUME' in sanitized_col:
# #                 new_columns[col] = 'Volume'
# #             # 'Adj Close'를 먼저 처리하여 일반 'Close'와 분리합니다.
# #             elif 'ADJ_CLOSE' in sanitized_col:
# #                 new_columns[col] = 'Adj Close'
# #             elif 'CLOSE' in sanitized_col:
# #                 new_columns[col] = 'Close'
        
# #         # 컬럼 이름 변경 적용
# #         data = data.rename(columns=new_columns)
        
# #         # 필수 컬럼이 모두 포함되었는지 확인
# #         if not all(col in data.columns for col in ['Close', 'Volume']):
# #             st.error(f"데이터에 'Close' 또는 'Volume' 컬럼이 없어 처리를 계속할 수 없습니다. (처리 후 컬럼: {data.columns.tolist()})")
# #             return None

# #         # Adj Close가 없으면 Close로 채워주는 폴백 로직
# #         if 'Adj Close' not in data.columns:
# #             data['Adj Close'] = data['Close']
        
# #         # 최종적으로 필요한 6개 컬럼만 남깁니다. (순서도 고정)
# #         final_cols = [col for col in CORE_COL_NAMES if col in data.columns]
# #         data = data[final_cols].copy()
        
# #         return data.dropna()
# #     except Exception as e:
# #         st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
# #         return None

# # # --- 3. 모델 훈련 및 검증 함수 ---
# # def train_and_validate_model(data_features):
# #     """
# #     시계열 분할을 이용해 LightGBM 모델을 훈련 및 검증하고 결과를 반환합니다.
# #     """
    
# #     # Feature와 Target 분리
# #     X = data_features.drop('Target', axis=1)
# #     y = data_features['Target']
    
# #     # --- LightGBM 오류 방지를 위한 피처 이름 정리 (Sanitization) ---
# #     # NOTE: 훈련 시 피처 이름을 표준화하여(공백 -> 언더바) 저장합니다. 
# #     # 이 이름(feature_columns)이 예측 시 사용됩니다.
# #     sanitized_columns = [
# #         str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '')
# #         for col in X.columns
# #     ]
# #     X.columns = sanitized_columns # 컬럼 이름 업데이트
# #     # ------------------------------------------------------------------
    
# #     # 스케일링
# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(X)
# #     X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
# #     # 시계열 교차 검증 (Time Series Split) 설정
# #     tscv = TimeSeriesSplit(n_splits=3)
    
# #     rmse_scores = []
    
# #     st.markdown("##### 🚀 모델 훈련 및 시계열 검증 진행 중...")
# #     progress_bar = st.progress(0)
    
# #     final_model = None
    
# #     for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled_df)):
# #         X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
# #         y_train, y_val = y.iloc[train_index], y.iloc[val_index]

# #         # LightGBM 모델 훈련 (조기 종료 적용)
# #         model = lgb.LGBMRegressor(**LGBM_PARAMS)
# #         model.fit(
# #             X_train, y_train,
# #             eval_set=[(X_val, y_val)],
# #             eval_metric='rmse',
# #             callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=-1)] # 조기 종료 라운드 조정
# #         )
        
# #         # 검증 세트 예측 및 RMSE 계산
# #         val_predictions = model.predict(X_val)
# #         rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
# #         rmse_scores.append(rmse)
        
# #         progress_bar.progress((fold + 1) / 3)
# #         st.caption(f"Fold {fold+1} 검증 완료. RMSE: {rmse:.4f}")
# #         final_model = model # 마지막 모델 저장

# #     avg_rmse = np.mean(rmse_scores)
# #     st.success(f"✅ 모델 훈련 완료. 평균 검증 RMSE: {avg_rmse:.4f}")
    
# #     return final_model, scaler, X.columns

# # # --- 4. 미래 예측 함수 ---
# # def predict_future(model, scaler, last_data, feature_columns):
# #     """
# #     훈련된 모델을 사용하여 향후 7일간의 주가를 예측합니다.
# #     """
    
# #     future_dates = [last_data.index[-1] + datetime.timedelta(days=i) for i in range(1, TARGET_PERIOD + 1)]
    
# #     predictions = []
    
# #     # Walk-Forward 방식으로 7일 예측
# #     for date in future_dates:
        
# #         # 1. 새로운 예측 시점 데이터 생성
# #         new_row = pd.DataFrame(index=[date])
# #         # 이전 예측값을 현재 종가로 설정
# #         new_row['Close'] = predictions[-1] if predictions else last_data['Close'].iloc[-1]
# #         new_row['Volume'] = last_data['Volume'].iloc[-1] 
        
# #         # 나머지 가격 컬럼(Open, High, Low, Adj Close)도 채워줍니다.
# #         price_cols = ['Open', 'High', 'Low', 'Adj Close']
# #         for col in price_cols:
# #              new_row[col] = new_row['Close'].iloc[0]
             
# #         # 2. 피처를 다시 생성하기 위해 과거 데이터에 새로운 행 추가
# #         temp_df = last_data.iloc[-60:].copy()
# #         temp_df = pd.concat([temp_df, new_row])
        
# #         # 피처 생성 (is_for_training=False: Target 생성 건너뛰기)
# #         temp_df = create_features(temp_df, is_for_training=False)
        
# #         # --- 핵심 수정: 예측에 사용되는 피처 데이터의 컬럼 이름 표준화 ---
# #         # 훈련 시 feature_columns가 'Adj Close'를 'Adj_Close'로 표준화했으므로, 
# #         # 예측 데이터의 컬럼 이름도 일치시켜야 KeyError가 발생하지 않습니다.
# #         sanitized_temp_columns = [
# #             str(col).replace(' ', '_') for col in temp_df.columns
# #         ]
# #         temp_df.columns = sanitized_temp_columns
# #         # -----------------------------------------------------------
        
# #         # 마지막 행(예측 일자)의 피처만 추출
# #         X_future_data = temp_df.iloc[-1].to_frame().T
        
# #         # 3. 모델이 기대하는 피처만 추출 및 정리 
# #         # feature_columns는 이제 'Adj_Close'와 같은 표준화된 이름을 포함합니다.
# #         X_future = X_future_data[feature_columns].fillna(0)
# #         X_future.columns = feature_columns # 컬럼 순서 및 이름 일치 강제

# #         # 4. 스케일링 적용
# #         X_future_scaled = scaler.transform(X_future)
        
# #         # 5. 예측
# #         next_price = model.predict(X_future_scaled)[0]
# #         predictions.append(next_price)
        
# #         # 6. 다음 예측을 위해 'last_data' 업데이트 (재귀적 예측을 위해)
# #         last_data = pd.concat([last_data, new_row])
        
# #     return pd.Series(predictions, index=future_dates)


# # # --- Streamlit 메인 앱 ---
# # st.set_page_config(layout="wide", page_title="LGBM 주가 예측 시스템")

# # def app():
# #     st.title("💡 LightGBM 고효율 주가 예측 시스템 (Feat. yfinance)")
# #     st.markdown("이 시스템은 **LightGBM**과 **시계열 특화 피처 엔지니어링**을 사용하여 과적합을 방지하고 예측 성능을 극대화합니다.")
# #     st.markdown("---")

# #     # 1. 종목 선택 및 실행 버튼
# #     TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY']
    
# #     col1, col2, _ = st.columns([1, 1, 3])
    
# #     with col1:
# #         selected_ticker = st.selectbox("예측할 종목 선택", TICKERS, key='ticker_select')
    
# #     with col2:
# #         st.markdown("<br>", unsafe_allow_html=True) 
# #         run_button = st.button("모델 훈련 및 예측 실행", type="primary")

# #     st.markdown("---") 

# #     if run_button:
# #         with st.spinner(f"⏳ '{selected_ticker}' 데이터 로드 및 피처 생성 중 (훈련 기간: {TRAIN_DAYS}일)..."):
            
# #             # 2. 데이터 로드 및 피처 생성
# #             raw_data = load_data(selected_ticker)
# #             if raw_data is None:
# #                 return

# #             # 훈련 데이터를 위한 피처 생성 (Target 포함)
# #             data_features = create_features(raw_data, is_for_training=True)
            
# #             if data_features.empty:
# #                  st.error("피처 생성 후 데이터가 충분하지 않습니다. 훈련 기간을 늘리거나 다른 종목을 선택하세요.")
# #                  return

# #             # 데이터 분할 (create_features의 dropna()로 인해 마지막 Target=NaN 행은 이미 제거됨)
# #             train_data = data_features
            
# #             st.subheader(f"🔍 종목 분석: {selected_ticker} (총 {len(train_data)}일 데이터)")
            
# #             # 3. 모델 훈련 및 검증
# #             model, scaler, feature_columns = train_and_validate_model(train_data)
            
# #             # 4. 미래 예측
# #             with st.spinner("🔮 미래 7일 예측 중 (Walk-Forward)..."):
                
# #                 last_actual_close = raw_data['Close'].iloc[-1]
                
# #                 # 예측을 위해 충분한 과거 데이터 확보
# #                 # raw_data는 이미 6개 컬럼으로 정제되었으므로 그대로 사용합니다.
# #                 last_data_for_prediction = raw_data.iloc[-100:].copy() 
                
# #                 # feature_columns는 이미 Target이 제거된 상태입니다.
                
# #                 future_predictions_series = predict_future(
# #                     model, 
# #                     scaler, 
# #                     last_data_for_prediction, 
# #                     feature_columns 
# #                 )
                
# #                 st.subheader(f"📈 {selected_ticker} 주가 예측 결과")
                
# #                 # 5. 결과 시각화
                
# #                 # 과거 및 예측 데이터 병합
# #                 past_prices = raw_data['Close'].iloc[-90:] # 최근 90일의 실제 가격 표시
                
# #                 predicted_df = pd.DataFrame({
# #                     'Actual': past_prices,
# #                     'Predicted': np.nan 
# #                 })
                
# #                 future_df = pd.DataFrame({
# #                     'Actual': np.nan,
# #                     'Predicted': future_predictions_series
# #                 })
                
# #                 final_df = pd.concat([predicted_df, future_df])
                
# #                 # 차트 생성
# #                 st.line_chart(final_df)
# #                 st.caption(f"마지막 실제 종가: ${last_actual_close:.2f}")

# #                 # 예측 수치 테이블
# #                 st.markdown("##### 🗓️ 향후 7일 예측 종가")
# #                 st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format('${:.2f}'))


# # if __name__ == "__main__":
# #     app()
