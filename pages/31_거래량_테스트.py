import streamlit as st
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler 
import plotly.express as px
import plotly.graph_objects as go
from ta.trend import MACD
from ta.momentum import RSI
import FinanceDataReader as fdr
import pyupbit
import requests
from json.decoder import JSONDecodeError

# --- 상수 정의 ---
TARGET_PERIOD = 10 # 예측할 미래 영업일 수

# 시장 매핑
MARKET_MAPPING = {
    "KRX": "한국 주식 (KRX)",
    "NASDAQ": "미국 증시 (NASDAQ)",
    "COIN": "코인 (Upbit)"
}

# --- LightGBM 모델 하이퍼파라미터 (안정화를 위해 학습률 조정) ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2000,          # Estimator 증가
    'learning_rate': 0.008,        # Learning Rate 하향 조정 (안정화)
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 1,
    'num_leaves': 31,              # 기본값으로 회귀 문제에 더 적합
    'max_depth': 8,                # 깊이 조정
    'lambda_l1': 0.3,              # 규제 완화
    'lambda_l2': 0.3,              # 규제 완화
    'min_child_samples': 10,       # 더 민감하게 패턴 학습
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# --------------------------
# 1. 멀티 마켓 종목 목록 로딩 함수 (기존과 동일)
# --------------------------
@st.cache_data(ttl=60*60*24)
def get_stock_listing(market_name):
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
        name_col = 'Name' if 'Name' in df.columns else 'Symbol'
        df['label'] = df[name_col].astype(str) + ' (' + df['Code'] + ')'
        return df
    except Exception as e:
        st.error(f"{market_name} 종목 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=60*60*24)
def get_coin_listing():
    try:
        url = "https://api.upbit.com/v1/market/all"
        response = requests.get(url, params={'isDetails': 'false'})
        response.raise_for_status()
        all_markets = response.json()
        
        krw_markets = [market for market in all_markets if market['market'].startswith('KRW-')]
        df_coin = pd.DataFrame(krw_markets)
        df_coin.rename(columns={'market': 'Code', 'korean_name': 'Name'}, inplace=True)
        
        df_coin['label'] = df_coin['Name'].astype(str) + ' (' + df_coin['Code'].str.replace('KRW-', '') + ')'
        
        return df_coin
    except Exception as e:
        st.error(f"코인 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# --------------------------
# 2. 피처 엔지니어링 함수 (로그 수익률 타겟 및 MACD/RSI 재도입)
# --------------------------
def create_features(df, is_for_training=True):
    """
    LightGBM 모델 훈련을 위한 시계열 피처를 생성합니다. (로그 수익률 타겟, MACD/RSI 포함)
    """
    df = df.copy()

    # 1. 시간 기반 피처
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear
    
    # 2. 지연 피처 (Lag Features)
    lags = [1, 3, 7, 14, 30]
    # 로그 변환된 수익률 지연 피처 추가 (가격 대신 변화율의 자기 상관성 학습)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)) 
    
    for lag in lags:
        # 종가 대신 로그 수익률의 지연 피처를 사용 (안정성 증가)
        df[f'LogR_Lag_{lag}'] = df['Log_Return'].shift(lag) 
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
    # 3. 이동 평균 및 볼륨 지표
    windows = [5, 20, 60]
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Vol_{window}'] = df['Close'].rolling(window=window).std()

    # 4. 상대적인 변화율 (차분 피처)
    df['Daily_Change'] = df['Close'].pct_change()
    
    # --- 5. 모멘텀, 변동성, 거래량 기반 신규 지표 ---
    df['Momentum_20'] = df['Close'].pct_change(periods=20)
    df['Volatility_20'] = df['Daily_Change'].rolling(window=20).std()
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # --- 6. MACD, RSI 재도입 (핵심 기술적 지표) ---
    macd = MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9, fillna=True)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff() # MACD와 Signal Line의 차이
    
    df['RSI'] = RSI(close=df['Close'], window=14, fillna=True).rsi()

    if is_for_training:
        # 7. 타겟 변수 (미래 1일 후의 로그 수익률)
        # Log Return = log(P_t+1 / P_t)
        df['Target'] = np.log(df['Close'].shift(-1) / df['Close'])

    # dropna()는 가장 긴 지연 피처/이동 평균을 기준으로 유효한 데이터만 남깁니다.
    df = df.dropna()
    
    return df

# --------------------------
# 3. 데이터 로드 함수 (기존과 동일)
# --------------------------
@st.cache_data(ttl=60*60*4) 
def load_data(ticker, market, train_days):
    end_date = datetime.date.today()
    # 기술적 지표 계산을 위해 넉넉한 기간 확보 (MA_60과 MACD/RSI 초기값 계산을 위해)
    start_date = end_date - datetime.timedelta(days=train_days + 150) 
    
    data = None
    
    try:
        if market in ['KRX', 'NASDAQ']:
            data = fdr.DataReader(ticker, start_date, end_date)
            data.index.name = 'Date'
            
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                st.error(f"데이터에 'Close' 또는 'Volume' 컬럼이 없어 처리를 계속할 수 없습니다.")
                return None
            
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
            
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        elif market == 'COIN':
            days_diff = (end_date - start_date).days
            count = days_diff + 1
            
            df_coin = pyupbit.get_ohlcv(ticker=ticker, interval='day', count=count)
            
            if df_coin is None or df_coin.empty:
                st.warning(f"오류: [{ticker}] 코인에 대한 데이터를 찾을 수 없습니다.")
                return None
                
            df_coin.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'value']
            df_coin.index.name = 'Date'
            
            df_coin['Adj Close'] = df_coin['Close']
            data = df_coin[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        if data is None or data.empty:
            return None

        # 코인 데이터는 주말 구분이 없으므로, 주식 데이터만 휴장일 처리
        if market in ['KRX', 'NASDAQ']:
             # Close가 0인 날 (휴장일 간주) 제외
            data = data[data['Close'] > 0].copy() 
            
        return data.dropna()
        
    except Exception as e:
        st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# --------------------------
# 4. 모델 훈련 및 예측 함수 (로그 수익률 타겟 및 안정화된 Walk-Forward)
# --------------------------
def train_and_validate_model(data_features, scaler_type):
    
    X = data_features.drop('Target', axis=1)
    y = data_features['Target'] # Target은 로그 수익률

    # LightGBM이 인식할 수 있도록 컬럼명 정리 (특수 문자 제거)
    sanitized_columns = [
        str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '').replace('-', '_')
        for col in X.columns
    ]
    X.columns = sanitized_columns
    
    # 스케일러 선택 (로그 수익률을 예측하기 때문에 X에만 스케일링 적용)
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else: # StandardScaler
        scaler = StandardScaler() 
        
    st.info(f"선택된 스케일러: **{scaler_type}**를 사용하여 **특징(X)** 데이터를 전처리합니다.")
    
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    # TimeSeriesSplit은 시계열 예측에서 가장 적합한 검증 방법입니다.
    tscv = TimeSeriesSplit(n_splits=3)
    
    rmse_scores = []
    
    st.markdown("##### 🚀 모델 훈련 및 시계열 검증 진행 중...")
    progress_bar = st.progress(0)
    
    final_model = None
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled_df)):
        X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=-1)]
        )
        
        val_predictions = model.predict(X_val)
        # RMSE는 로그 수익률에 대한 오차입니다.
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions)) 
        rmse_scores.append(rmse)
        
        progress_bar.progress((fold + 1) / 3)
        st.caption(f"Fold {fold+1} 검증 완료. **로그 수익률 RMSE**: {rmse:.6f}")
        final_model = model

    avg_rmse = np.mean(rmse_scores)
    st.success(f"✅ 모델 훈련 완료. 평균 검증 **로그 수익률 RMSE**: {avg_rmse:.6f}")
    
    return final_model, scaler, X.columns, avg_rmse

def predict_future(model, scaler, last_data, feature_columns, market_key):
    """
    훈련된 모델을 사용하여 향후 TARGET_PERIOD 일간의 주가를 Walk-Forward 방식으로 예측합니다.
    - 예측 타겟: 로그 수익률
    - 안정화: 예측된 로그 수익률을 현재 가격에 복리 적용하여 다음 가격 예측의 기반으로 사용.
    - 휴장일 제외: 다음 영업일만 예측 루프에 포함.
    """
    
    current_date = last_data.index[-1]
    last_actual_close = last_data['Close'].iloc[-1]
    
    future_predictions_log_returns = []
    future_prices = [last_actual_close] # 시작 가격
    future_dates = []

    # 예측 루프: TARGET_PERIOD 만큼의 영업일(Trade Date)을 찾을 때까지 반복
    i = 1
    while len(future_predictions_log_returns) < TARGET_PERIOD:
        
        # 1. 다음 날짜 (일단 달력상 다음 날)
        next_date = current_date + datetime.timedelta(days=i)
        
        # 2. 영업일 필터링 (KRX, NASDAQ만 적용)
        is_trading_day = True
        # 코인은 365일 거래되므로 주식 시장만 휴장일 확인
        if market_key in ['KRX', 'NASDAQ']:
            # 주말 체크: 토요일(5), 일요일(6)
            if next_date.weekday() in [5, 6]:
                i += 1
                continue
            # 공휴일 체크 (fdr로 확인 불가, 임시로 주말만 체크)
            # 실제 서비스에서는 공휴일 API를 사용해야 합니다.
            
        # 3. Walk-Forward를 위한 데이터 생성
        # 예측에 사용할 기반 가격은 이전에 누적된 예측 가격 (복리 효과)
        current_prediction_base_price = future_prices[-1] 
        
        # 가상의 다음 날 데이터 생성 (가격은 현재 예측 기반 가격으로 임시 설정)
        new_row = pd.DataFrame(index=[next_date])
        new_row['Close'] = current_prediction_base_price
        # Volume, Open, High, Low 등은 마지막 실제 데이터로 복사 (가정)
        # 이 가정이 터무니없는 예측의 원인이 될 수 있으나, 미래 데이터는 알 수 없으므로 최선의 추정입니다.
        for col in ['Open', 'High', 'Low', 'Adj Close']:
              new_row[col] = new_row['Close'].iloc[0]
        new_row['Volume'] = last_data['Volume'].iloc[-1] 
              
        # 다음 날 피처 생성을 위해 기존 데이터에 가상 데이터 추가
        # 기술적 지표 계산에 필요한 데이터만 포함 (약 60일치)
        temp_df = last_data.iloc[-60:].copy() 
        # 마지막 데이터의 Close를 현재 예측 기반 가격으로 업데이트
        temp_df.loc[temp_df.index[-1], 'Close'] = current_prediction_base_price 
        temp_df = pd.concat([temp_df, new_row])
        
        # 4. 피처 생성 (로그 수익률, MACD/RSI 포함)
        temp_df_features = create_features(temp_df, is_for_training=False)
        
        # 5. 예측 입력 데이터 준비
        sanitized_temp_columns = [
            str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '').replace('-', '_')
            for col in temp_df_features.columns
        ]
        temp_df_features.columns = sanitized_temp_columns

        # 예측에 사용할 마지막 행 (다음 날)의 특징 벡터
        X_future_data = temp_df_features.iloc[-1].to_frame().T
        
        # 훈련된 피처 컬럼만 사용
        X_future = X_future_data[feature_columns].fillna(0)
        X_future.columns = feature_columns

        # 6. 스케일러로 변환 및 로그 수익률 예측
        X_future_scaled = scaler.transform(X_future)
        
        # next_log_return은 로그 수익률 예측값입니다.
        next_log_return = model.predict(X_future_scaled)[0] 
        
        # 7. 가격으로 역변환 및 결과 저장 (복리 적용 안정화)
        # P_t+1 = P_t * exp(log_return)
        next_price = current_prediction_base_price * np.exp(next_log_return)
        
        future_predictions_log_returns.append(next_log_return)
        future_prices.append(next_price) # 다음 루프의 기반 가격으로 사용
        future_dates.append(next_date)
        
        # 다음 예측을 위해 'last_data' 업데이트
        last_data = pd.concat([last_data, new_row])
        current_date = next_date
        i = 1 # 다음 영업일을 찾기 위해 1일씩 증가

    # 첫 번째 가격(last_actual_close) 제거 후 반환
    return pd.Series(future_prices[1:], index=future_dates)

def display_feature_importance(model, feature_columns):
    # (특징 중요도 시각화 함수는 기존과 동일)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(20)

    fig = px.bar(
        feature_importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='모델 특징 중요도 (Feature Importance)',
        labels={'Importance': '중요도', 'Feature': '특징 이름'}
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 5. Streamlit 메인 앱 (기존과 동일)
# --------------------------
st.set_page_config(layout="wide", page_title="LGBM 멀티 자산 예측 시스템 (안정화)")

def app():
    st.title("📈 LightGBM 예측 시스템: 안정화 및 로그 수익률 기반 예측")
    st.markdown("**로그 수익률 타겟**, **핵심 기술 지표(MACD/RSI)**, **휴장일 제외 Walk-Forward**로 예측 안정성을 극대화했습니다.")
    st.markdown("---")

    # 1. 시장 및 종목 선택
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1]) 
    
    with col1:
        selected_market_name = st.selectbox(
            "📊 예측할 자산 선택",
            list(MARKET_MAPPING.values()),
            key='market_select'
        )
    
    market_key = [k for k, v in MARKET_MAPPING.items() if v == selected_market_name][0]

    with col3:
        # 훈련 기간 선택 UI
        selected_train_days = st.number_input(
            "📅 훈련기간(단위:일)",
            min_value=120,
            max_value=3650,
            value=730, # 안정적인 학습을 위해 기본 2년(730일)로 상향
            step=30,
            key='train_days_input',
            help="모델 훈련에 사용할 과거 데이터 기간 설정."
        )

    with col4:
        # 스케일러 선택 UI 추가
        selected_scaler = st.selectbox(
            "⚖️ 스케일러 선택",
            ["RobustScaler", "StandardScaler"],
            key='scaler_select',
            help="RobustScaler는 이상치에 강하고, StandardScaler는 일반적인 정규 분포에 적합합니다. 특징(X)에만 적용됩니다."
        )
        
    with col2:
        # 2. 종목 목록 동적 로딩
        stock_list_df = pd.DataFrame()
        default_ticker = ""

        if market_key == 'KRX':
            stock_list_df = get_stock_listing('KRX')
            default_ticker = '005930' # 삼성전자
            
        elif market_key == 'NASDAQ':
            stock_list_df = get_stock_listing('NASDAQ')
            default_ticker = 'AAPL'
            
        elif market_key == 'COIN':
            stock_list_df = get_coin_listing()
            default_ticker = 'KRW-BTC'
        
        if not stock_list_df.empty:
            options = stock_list_df['label'].tolist()
            try:
                # 안전한 기본값 설정
                default_index = options.index(stock_list_df[stock_list_df['Code'] == default_ticker]['label'].iloc[0])
            except:
                default_index = 0
                
            selected_label = st.selectbox(
                f"🏷️ 예측할 {selected_market_name} 종목/코인", 
                options, 
                index=default_index, 
                key='ticker_label_select'
            )
            
            selected_ticker = stock_list_df[stock_list_df['label'] == selected_label]['Code'].iloc[0].upper().strip()
            
        else:
            st.warning("선택한 시장의 종목 목록을 불러올 수 없습니다.")
            selected_ticker = ""
    
    st.markdown("---")
    
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 3, 1])
    with col_btn_center:
        run_button = st.button("모델 훈련 및 예측 실행", type="primary", use_container_width=True)

    if run_button and (not selected_ticker or selected_train_days < 120):
        st.warning("예측할 종목을 선택하고, 훈련 기간은 최소 120일 이상 설정해주세요.")
        return

    if run_button:
        current_market = market_key 
        
        with st.spinner(f"⏳ '{selected_ticker}' ({current_market}) 데이터 로드 및 피처 생성 중..."):
            
            # 2. 데이터 로드 및 피처 생성
            raw_data = load_data(selected_ticker, current_market, selected_train_days) 
            if raw_data is None:
                return

            data_features = create_features(raw_data, is_for_training=True)
            
            min_data_needed = 60 # 피처 생성에 필요한 최소 일수
            if len(data_features) < min_data_needed:
                st.error(f"피처 생성 후 데이터가 너무 적습니다 ({len(data_features)}일). 훈련 기간을 늘리거나 다른 종목을 선택하세요.")
                return

            train_data = data_features
            
            st.subheader(f"📊 분석 결과: {selected_label}")
            
            # 3. 모델 훈련 및 검증
            model, scaler, feature_columns, avg_rmse = train_and_validate_model(train_data, selected_scaler)
            
            # 4. 특징 중요도 시각화
            st.markdown("---")
            st.subheader("💡 훈련 모델 진단: 특징 중요도 분석")
            st.markdown(f"모델의 평균 검증 오차(로그 수익률 RMSE): **{avg_rmse:.6f}**")
            st.markdown("새로 추가된 **MACD, RSI** 지표와 **로그 수익률 지연 피처(LogR_Lag)**의 중요도를 확인해 보세요.")
            display_feature_importance(model, feature_columns)
            
            # 5. 미래 예측
            with st.spinner(f"🔮 미래 {TARGET_PERIOD}일 예측 중 (Walk-Forward, 영업일 기준)..."):
                
                last_actual_close = raw_data['Close'].iloc[-1]
                # 예측에 필요한 충분한 과거 데이터
                last_data_for_prediction = raw_data.iloc[-100:].copy() 
                
                # 예측 함수에 시장 키 전달 (휴장일 로직 적용 위함)
                future_predictions_series = predict_future(
                    model, 
                    scaler, 
                    last_data_for_prediction, 
                    feature_columns,
                    current_market
                )
                
                st.markdown("---")
                st.subheader(f"📈 {selected_label} 가격 예측 시각화 (예측 영업일 기준)")
                
                # 결과 시각화
                past_prices = raw_data['Close'].iloc[-90:]
                
                predicted_df = pd.DataFrame({
                    'Actual': past_prices,
                    'Predicted': np.nan 
                })
                
                future_df = future_predictions_series.to_frame(name='Predicted')
                future_df['Actual'] = np.nan
                
                # 실제 데이터와 예측 데이터를 합칠 때 인덱스를 기준으로 정렬
                final_df = pd.concat([predicted_df, future_df]).sort_index()
                
                # 마지막 예측 가격을 빨간 점으로 강조
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Actual'], mode='lines', name='실제 종가', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Predicted'], mode='lines+markers', name='예측 종가', line=dict(color='red', dash='dot'), marker=dict(size=4)))

                fig.update_layout(
                    title=f'{selected_label} 실제 가격 vs. 예측 가격',
                    yaxis_title='가격',
                    xaxis_title='날짜',
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                currency = "원" if current_market in ['KRX', 'COIN'] else "$"
                st.caption(f"마지막 실제 종가: {currency}{last_actual_close:,.2f}")

                # 예측 수치 테이블
                st.markdown(f"##### 🗓️ 향후 {TARGET_PERIOD} 영업일 예측 종가")
                st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format(f'{currency}{{:.2f}}'))


if __name__ == "__main__":
    app()
