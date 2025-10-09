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
import FinanceDataReader as fdr
import pyupbit
import requests
from json.decoder import JSONDecodeError

# --- 상수 정의 ---
TARGET_PERIOD = 10 # 예측할 미래 영업일 수
QUANTILE_ALPHA = 0.05 # 95% 신뢰구간을 위한 퀀타일 (0.025 및 0.975)

# 시장 매핑
MARKET_MAPPING = {
    "KRX": "한국 주식 (KRX)",
    "NASDAQ": "미국 증시 (NASDAQ)",
    "COIN": "코인 (Upbit)"
}

# --- LightGBM 모델 하이퍼파라미터 (안정화 설정 유지) ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.008,
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 1,
    'num_leaves': 31,
    'max_depth': 8,
    'lambda_l1': 0.3,
    'lambda_l2': 0.3,
    'min_child_samples': 10,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# --------------------------
# 0. 도우미 함수 (컬럼 일반화, MACD, RSI 수동 계산)
# --------------------------
def sanitize_columns(columns):
    """LightGBM이 인식할 수 있도록 컬럼명을 정리하고 통일합니다."""
    return [
        str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '').replace('-', '_')
        for col in columns
    ]

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """MACD와 MACD Signal을 수동으로 계산합니다."""
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, macd_signal

def calculate_rsi(series, window=14):
    """RSI를 수동으로 계산합니다."""
    diff = series.diff()
    gain = diff.mask(diff < 0, 0)
    loss = diff.mask(diff > 0, 0).abs()
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(1e-10))) 
    return rsi

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
# 2. 피처 엔지니어링 함수 (기존과 동일)
# --------------------------
def create_features(df, is_for_training=True):
    df = df.copy()

    # 1. 시간 기반 피처
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear
    
    # 2. 지연 피처 (Lag Features)
    lags = [1, 3, 7, 14, 30]
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)) 
    
    for lag in lags:
        df[f'LogR_Lag_{lag}'] = df['Log_Return'].shift(lag) 
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
    # 3. 이동 평균 및 볼륨 지표
    windows = [5, 20, 60]
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Vol_{window}'] = df['Close'].rolling(window=window).std()

    # 4. 상대적인 변화율 (차분 피처)
    df['Daily_Change'] = df['Close'].pct_change()
    
    # 5. 모멘텀, 변동성, 거래량 기반 신규 지표
    df['Momentum_20'] = df['Close'].pct_change(periods=20)
    df['Volatility_20'] = df['Daily_Change'].rolling(window=20).std()
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # 6. MACD, RSI 수동 계산
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
    df['RSI'] = calculate_rsi(df['Close'])

    if is_for_training:
        # 7. 타겟 변수 (미래 1일 후의 로그 수익률)
        df['Target'] = np.log(df['Close'].shift(-1) / df['Close'])

    df = df.dropna()
    
    return df

# --------------------------
# 3. 데이터 로드 함수 (기존과 동일)
# --------------------------
@st.cache_data(ttl=60*60*4) 
def load_data(ticker, market, train_days):
    end_date = datetime.date.today()
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

        if market in ['KRX', 'NASDAQ']:
            data = data[data['Close'] > 0].copy() 
            
        return data.dropna()
        
    except Exception as e:
        st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# --------------------------
# 4. 모델 훈련 및 예측 함수
# --------------------------
def train_and_validate_model(data_features, scaler_type, n_splits):
    
    X = data_features.drop('Target', axis=1)
    y = data_features['Target'] 

    X.columns = sanitize_columns(X.columns)
    
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else: 
        scaler = StandardScaler() 
        
    st.info(f"선택된 스케일러: **{scaler_type}**를 사용하여 **특징(X)** 데이터를 전처리합니다.")
    
    # 스케일링
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    residual_data = pd.DataFrame()
    
    st.markdown("##### 🚀 모델 훈련 및 시계열 검증 진행 중...")
    progress_bar = st.progress(0)
    final_model = None
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled_df)):
        X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        # 훈련 시 Numpy 배열로 명시적 변환
        model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=-1)]
        )
        
        # 예측 시에도 Numpy 배열로 명시적 변환
        val_predictions = model.predict(X_val.values)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions)) 
        rmse_scores.append(rmse)
        
        # 잔차(Residual) 계산
        residuals = y_val - val_predictions
        
        # 실제 수익률 RMSE 계산 (지수 함수를 사용해 역변환)
        actual_return_rmse = np.sqrt(np.mean((np.expm1(y_val) - np.expm1(val_predictions))**2)) * 100
        
        fold_residual_df = pd.DataFrame({
            'Residual': residuals,
            'Fold': f'Fold {fold+1}',
            'Target': y_val
        })
        residual_data = pd.concat([residual_data, fold_residual_df])
        
        progress_bar.progress((fold + 1) / n_splits)
        st.caption(f"Fold {fold+1} 검증 완료. **로그 수익률 RMSE**: {rmse:.6f} (**실제 수익률 RMSE**: {actual_return_rmse:.4f}%)")
        final_model = model

    avg_rmse = np.mean(rmse_scores)
    st.success(f"✅ 모델 훈련 완료. 평균 검증 **로그 수익률 RMSE**: {avg_rmse:.6f}")
    
    return final_model, scaler, X.columns, avg_rmse, residual_data, X, y 
    # X, y를 함께 반환하여 퀀타일 모델 훈련에 사용합니다.

def predict_future(models, scaler, last_data, feature_columns, market_key):
    
    current_date = last_data.index[-1] 
    last_actual_close = last_data['Close'].iloc[-1]
    
    future_predictions = [] 
    future_low = [] 
    future_high = [] 
    future_dates = []

    day_counter = 1 
    
    while len(future_predictions) < TARGET_PERIOD:
        
        next_date = current_date + datetime.timedelta(days=day_counter) 
        
        if market_key in ['KRX', 'NASDAQ']:
            if next_date.weekday() in [5, 6]:
                day_counter += 1 
                continue
            
        current_prediction_base_price = future_predictions[-1] if future_predictions else last_actual_close
        
        # 가상의 다음 날 데이터 생성 (인덱스: Timestamp)
        new_row = pd.DataFrame(index=[next_date])
        new_row['Close'] = current_prediction_base_price
        for col in ['Open', 'High', 'Low', 'Adj Close']:
              new_row[col] = new_row['Close'].iloc[0]
        new_row['Volume'] = last_data['Volume'].iloc[-1] 
              
        temp_df = last_data.iloc[-60:].copy() 
        temp_df.at[temp_df.index[-1], 'Close'] = current_prediction_base_price 
        temp_df = pd.concat([temp_df, new_row])
        
        # 피처 생성 및 일반화
        temp_df_features = create_features(temp_df, is_for_training=False)
        temp_df_features.columns = sanitize_columns(temp_df_features.columns)

        X_future_data = temp_df_features.iloc[-1].to_frame().T
        X_future = X_future_data[feature_columns].fillna(0)
        
        # 예측 입력 데이터는 Numpy 배열로 변환
        X_future_scaled = scaler.transform(X_future)
        
        # 퀀타일 예측 (95% CI)
        log_return_median = models['median'].predict(X_future_scaled)[0] 
        log_return_low = models['low'].predict(X_future_scaled)[0] 
        log_return_high = models['high'].predict(X_future_scaled)[0] 
        
        # 가격으로 역변환 (복리 적용)
        next_price_median = current_prediction_base_price * np.exp(log_return_median)
        next_price_low = current_prediction_base_price * np.exp(log_return_low)
        next_price_high = current_prediction_base_price * np.exp(log_return_high)
        
        future_predictions.append(next_price_median)
        future_low.append(next_price_low)
        future_high.append(next_price_high)
        future_dates.append(next_date)
        
        # 다음 예측을 위해 'current_date' 업데이트 및 카운터 리셋
        last_data = pd.concat([last_data, new_row])
        current_date = next_date
        day_counter = 1 

    return pd.DataFrame({
        'Predicted': future_predictions,
        'Low_CI': future_low,
        'High_CI': future_high
    }, index=future_dates)

# --------------------------
# 5. 시각화 및 분석 함수
# --------------------------
def display_feature_importance(model, feature_columns):
    
    importances = model.feature_importances_
    
    # [보완] 중요도 정규화 (0~100 스케일)
    total_importance = importances.sum()
    if total_importance > 0:
        normalized_importances = (importances / total_importance) * 100
    else:
        # 합이 0이면 모두 0으로 표시
        normalized_importances = importances 

    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': normalized_importances
    }).sort_values(by='Importance', ascending=False).head(20)

    fig = px.bar(
        feature_importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='모델 특징 중요도 (0-100% 스케일 보정)',
        labels={'Importance': '상대적 중요도 (%)', 'Feature': '특징 이름'},
        height=500
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def display_residual_analysis(residual_data):
    st.markdown("##### 🔬 잔차(Residual) 분석")
    st.caption("잔차는 **실제 로그 수익률 - 예측 로그 수익률**이며, 잔차의 분포는 모델의 학습 신뢰도를 나타냅니다.")

    # 잔차 히스토그램
    fig_hist = px.histogram(
        residual_data, 
        x='Residual', 
        color='Fold', 
        marginal='box',
        nbins=50,
        title='검증 잔차 분포 (로그 수익률)',
        labels={'Residual': '잔차 (로그 수익률)'},
        height=400
    )
    fig_hist.update_layout(xaxis_title="잔차 (Log Return)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 잔차 시계열 (Residual Time Series)
    fig_ts = go.Figure()
    for fold in residual_data['Fold'].unique():
        fold_data = residual_data[residual_data['Fold'] == fold]
        fig_ts.add_trace(go.Scatter(
            x=fold_data.index, 
            y=fold_data['Residual'], 
            mode='markers', 
            name=fold,
            marker=dict(size=4)
        ))
    
    fig_ts.update_layout(
        title='검증 잔차 시계열 분포 (로그 수익률)',
        yaxis_title='잔차 (Log Return)',
        xaxis_title='날짜',
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_ts, use_container_width=True)


# --------------------------
# 6. Streamlit 메인 앱
# --------------------------
st.set_page_config(layout="wide", page_title="LGBM 멀티 자산 예측 시스템 (최종 안정화 + 보완)")

def app():
    st.title("🏆 LightGBM 예측 시스템: 최종 보완 버전")
    st.markdown("**잔차 분석, 신뢰구간, 유연한 TimeSeriesSplit**이 적용되어 모델의 신뢰도를 높였습니다.")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1]) 
    
    with col1:
        selected_market_name = st.selectbox(
            "📊 예측할 자산 선택",
            list(MARKET_MAPPING.values()),
            key='market_select'
        )
    
    market_key = [k for k, v in MARKET_MAPPING.items() if v == selected_market_name][0]

    with col3:
        selected_train_days = st.number_input(
            "📅 훈련기간(단위:일)",
            min_value=120,
            max_value=3650,
            value=730, 
            step=30,
            key='train_days_input',
            help="모델 훈련에 사용할 과거 데이터 기간 설정."
        )

    with col4:
        selected_scaler = st.selectbox(
            "⚖️ 스케일러 선택",
            ["RobustScaler", "StandardScaler"],
            key='scaler_select',
            help="특징(X)에만 적용됩니다."
        )
        
    with col5:
        default_n_splits = 5
        selected_n_splits = st.number_input(
            "✂️ TimeSeriesSplit 분할 수 (k)",
            min_value=3,
            max_value=10,
            value=default_n_splits, 
            step=1,
            key='n_splits_input',
            help="검증 데이터셋 개수. 데이터가 충분하지 않으면 작게 설정하세요."
        )

    with col2:
        stock_list_df = pd.DataFrame()
        default_ticker = ""

        if market_key == 'KRX':
            stock_list_df = get_stock_listing('KRX')
            default_ticker = '005930'
            
        elif market_key == 'NASDAQ':
            stock_list_df = get_stock_listing('NASDAQ')
            default_ticker = 'AAPL'
            
        elif market_key == 'COIN':
            stock_list_df = get_coin_listing()
            default_ticker = 'KRW-BTC'
        
        if not stock_list_df.empty:
            options = stock_list_df['label'].tolist()
            try:
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
            
            raw_data = load_data(selected_ticker, current_market, selected_train_days) 
            if raw_data is None:
                return

            data_features = create_features(raw_data, is_for_training=True)
            
            min_data_needed = 60 
            if len(data_features) < min_data_needed:
                st.error(f"피처 생성 후 데이터가 너무 적습니다 ({len(data_features)}일). 훈련 기간을 늘리거나 다른 종목을 선택하세요.")
                return

            train_data = data_features
            
            st.subheader(f"📊 분석 결과: {selected_label}")
            
            # 1. 중앙값 (Median) 예측 모델 훈련 및 검증
            st.markdown("#### 🥇 중앙값 (Median) 모델 훈련")
            model_median, scaler, feature_columns, avg_rmse, residual_data, X_raw, y_raw = train_and_validate_model(
                train_data, selected_scaler, selected_n_splits
            )
            
            # 2. 신뢰구간 (CI) 모델 훈련
            models = {'median': model_median}
            
            # [수정] LGBM_PARAMS의 복사본을 만들고 'objective' 키를 제거합니다. (오류 해결 핵심)
            LGBM_QUANTILE_PARAMS = LGBM_PARAMS.copy()
            if 'objective' in LGBM_QUANTILE_PARAMS:
                del LGBM_QUANTILE_PARAMS['objective']

            # X와 y를 Numpy 배열로 명시적 변환
            X_train_scaled = scaler.transform(X_raw).astype('float32')
            y_train_values = y_raw.values
            
            st.markdown("#### 🥈 신뢰구간 모델 훈련 (Quantile Regression)")
            with st.spinner("⏳ 95% 신뢰구간 하한선(Low CI) 모델 훈련 중..."):
                # 수정: **LGBM_QUANTILE_PARAMS 사용
                lgbm_low = lgb.LGBMRegressor(objective='quantile', alpha=QUANTILE_ALPHA/2, **LGBM_QUANTILE_PARAMS).fit(
                    X_train_scaled, y_train_values
                )
                models['low'] = lgbm_low
            
            with st.spinner("⏳ 95% 신뢰구간 상한선(High CI) 모델 훈련 중..."):
                # 수정: **LGBM_QUANTILE_PARAMS 사용
                lgbm_high = lgb.LGBMRegressor(objective='quantile', alpha=1-(QUANTILE_ALPHA/2), **LGBM_QUANTILE_PARAMS).fit(
                    X_train_scaled, y_train_values
                )
                models['high'] = lgbm_high
            st.success("✅ 퀀타일 회귀 모델 훈련 완료.")

            st.markdown("---")
            st.subheader("💡 훈련 모델 진단")
            
            # 잔차 분석 시각화
            display_residual_analysis(residual_data)
            
            # 특징 중요도 시각화
            st.markdown("---")
            display_feature_importance(model_median, feature_columns) 

            # 예측 실행
            with st.spinner(f"🔮 미래 {TARGET_PERIOD}일 예측 중 (Walk-Forward, 95% CI)..."):
                
                last_actual_close = raw_data['Close'].iloc[-1]
                last_data_for_prediction = raw_data.iloc[-100:].copy() 
                
                future_predictions_df = predict_future(
                    models, 
                    scaler, 
                    last_data_for_prediction, 
                    feature_columns,
                    current_market
                )
                
                st.markdown("---")
                st.subheader(f"📈 {selected_label} 가격 예측 시각화 (95% 신뢰구간)")
                
                past_prices = raw_data['Close'].iloc[-90:]
                
                predicted_df = pd.DataFrame({
                    'Actual': past_prices,
                    'Predicted': np.nan,
                    'Low_CI': np.nan,
                    'High_CI': np.nan
                })
                
                final_df = pd.concat([predicted_df, future_predictions_df]).sort_index()
                
                # Plotly 시각화 (신뢰구간 포함)
                fig = go.Figure()
                
                # 신뢰구간 음영 추가
                fig.add_trace(go.Scatter(
                    x=final_df.index, 
                    y=final_df['High_CI'], 
                    fill=None, 
                    mode='lines', 
                    line=dict(width=0), 
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=final_df.index, 
                    y=final_df['Low_CI'], 
                    fill='tonexty', 
                    mode='lines', 
                    line=dict(width=0), 
                    fillcolor='rgba(255, 0, 0, 0.1)', 
                    name='95% 신뢰구간'
                ))
                
                # 예측선 (중앙값)
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Predicted'], mode='lines', name='예측 종가 (Median)', line=dict(color='red', dash='dot')))
                
                # 실제 가격
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Actual'], mode='lines', name='실제 종가', line=dict(color='blue')))

                fig.update_layout(
                    title=f'{selected_label} 실제 가격 vs. 예측 가격 및 95% 신뢰구간',
                    yaxis_title='가격',
                    xaxis_title='날짜',
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                currency = "원" if current_market in ['KRX', 'COIN'] else "$"
                st.caption(f"마지막 실제 종가: {currency}{last_actual_close:,.2f}")

                st.markdown(f"##### 🗓️ 향후 {TARGET_PERIOD} 영업일 예측 결과")
                
                # 로그 수익률을 실제 수익률(%)로 변환하여 표시
                predictions_display = future_predictions_df.copy()
                
                # 수익률 계산: P_t+1 / P_t - 1
                return_pct = (predictions_display['Predicted'] / predictions_display['Predicted'].shift(1)) - 1
                
                # 첫날의 수익률은 마지막 실제 종가를 기준으로 계산
                return_pct.iloc[0] = (predictions_display['Predicted'].iloc[0] / last_actual_close) - 1
                
                predictions_display['일일 예측 수익률 (%)'] = return_pct * 100
                predictions_display.rename(columns={'Predicted': '예측 종가 (Median)', 'Low_CI': '95% CI 하한', 'High_CI': '95% CI 상한'}, inplace=True)
                
                st.dataframe(predictions_display[['예측 종가 (Median)', '95% CI 하한', '95% CI 상한', '일일 예측 수익률 (%)']].style.format({
                    '예측 종가 (Median)': f'{currency}{{:.2f}}',
                    '95% CI 하한': f'{currency}{{:.2f}}',
                    '95% CI 상한': f'{currency}{{:.2f}}',
                    '일일 예측 수익률 (%)': '{:.2f}%'
                }))


if __name__ == "__main__":
    app()
