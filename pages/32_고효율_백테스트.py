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
# pykrx 추가
from pykrx import stock 
# FinanceDataReader는 NASDAQ용으로 남겨둠
import FinanceDataReader as fdr 
import pyupbit
import requests
from json.decoder import JSONDecodeError

# --- 상수 정의 ---
TARGET_PERIOD = 10 # 예측할 미래 영업일 수
CI_Z_SCORE = 1.96 # 95% 신뢰구간을 위한 Z-score (정규분포 가정)
TOP_N_FEATURES = 12 # 사용할 상위 특징 개수 (모델 경량화)
DEFAULT_TRAIN_DAYS = 730 # 기본 훈련 기간 2년으로 설정

# 시장 매핑
MARKET_MAPPING = {
    "KRX": "한국 주식 (KRX) [pykrx 적용]",
    "NASDAQ": "미국 증시 (NASDAQ) [fdr 유지]",
    "COIN": "코인 (Upbit) [pyupbit 유지]"
}

# --- LightGBM 모델 하이퍼파라미터 (최종 고속화 설정) ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 500, 
    'learning_rate': 0.015, 
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 1,
    'num_leaves': 21, 
    'max_depth': 6,
    'lambda_l1': 0.3,
    'lambda_l2': 0.3,
    'min_child_samples': 10,
    'verbose': -1, # 모델 생성 시 verbose 설정
    'n_jobs': -1,
    'seed': 42
}

# --------------------------
# 0. 도우미 함수 
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
# 1. 멀티 마켓 종목 목록 로딩 함수 (pykrx/fdr 적용)
# --------------------------
@st.cache_data(ttl=60*60*24)
def get_stock_listing(market_name, clear_cache=False): 
    
    # 📌 KRX: pykrx 사용
    if market_name == 'KRX':
        st.info("pykrx를 사용하여 KRX 종목 리스트를 가져오는 중입니다. 시간이 걸릴 수 있습니다.")
        try:
            ticker_list = stock.get_market_ticker_list()
            
            valid_tickers = []
            
            # [핵심 수정]: 종목명 조회 시 오류 처리 추가
            for ticker in ticker_list:
                try:
                    name = stock.get_market_ticker_name(ticker)
                    # 유효한 종목명만 추가 (종목명이 None이나 빈 문자열이 아닌 경우)
                    if name and name.strip():
                        valid_tickers.append({'Code': ticker, 'Name': name})
                except Exception as name_error:
                    # 종목명 조회 중 오류가 발생하면 해당 종목은 건너뜀
                    # st.warning(f"종목 코드 {ticker}의 종목명 조회 중 오류 발생: {name_error}")
                    continue
            
            df = pd.DataFrame(valid_tickers)
            
            if df.empty:
                st.error("pykrx에서 유효한 KRX 종목 리스트를 가져오지 못했습니다.")
                return pd.DataFrame()

            df['label'] = df['Name'].astype(str) + ' (' + df['Code'] + ')'
            return df
            
        except Exception as e:
            st.error(f"KRX 종목 리스트를 가져오는 중 오류가 발생했습니다 (pykrx): {e}")
            return pd.DataFrame()        
    # 📌 NASDAQ: fdr 유지
    elif market_name == 'NASDAQ':
        try:
            df = fdr.StockListing('NASDAQ')
            df.rename(columns={'Symbol': 'Code'}, inplace=True)
            name_col = 'Name' if 'Name' in df.columns else 'Symbol'
            df['label'] = df[name_col].astype(str) + ' (' + df['Code'] + ')'
            return df
        except Exception as e:
            st.error(f"NASDAQ 종목 리스트를 가져오는 중 오류가 발생했습니다 (fdr): {e}")
            return pd.DataFrame()
            
    else:
        return pd.DataFrame()
        
@st.cache_data(ttl=60*60*24)
def get_coin_listing(clear_cache=False): 
    # 코인: pyupbit/requests 유지
    try:
        url = "https://api.upbit.com/v1/market/all"
        # 📌 헤더에 User-Agent 추가하여 외부 접속 문제 완화 시도
        headers = {'User-Agent': 'Mozilla/5.0'} 
        response = requests.get(url, params={'isDetails': 'false'}, headers=headers)
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
# 2. 피처 엔지니어링 함수 (유지)
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
# 3. 데이터 로드 함수 (pykrx/fdr/pyupbit 적용)
# --------------------------
@st.cache_data(ttl=60*60*4) 
def load_data(ticker, market, train_days, clear_cache=False): 
    
    end_date = datetime.date.today()
    # 넉넉하게 피처 생성을 위해 150일 추가
    start_date = end_date - datetime.timedelta(days=train_days + 150) 
    
    data = None
    
    try:
        # 📌 KRX: pykrx 사용
        if market == 'KRX':
            # pykrx는 날짜 포맷 'YYYYMMDD' 사용
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')
            
            # 주가, 거래량 데이터 로드
            data = stock.get_market_ohlcv_by_date(
                fromdate=start_date_str, 
                todate=end_date_str, 
                ticker=ticker,
                freq='d'
            )
            
            if data.empty:
                 st.warning(f"오류: [{ticker}] 한국 주식 데이터를 찾을 수 없습니다.")
                 return None
                 
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
            data = data.drop(columns=['Change'])
            data.index.name = 'Date'
            data['Adj Close'] = data['Close']
            
        # 📌 NASDAQ: fdr 사용
        elif market == 'NASDAQ':
            data = fdr.DataReader(ticker, start_date, end_date)
            data.index.name = 'Date'
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        # 📌 COIN: pyupbit 사용
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

        # 데이터 검증
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            st.error(f"데이터에 'Close' 또는 'Volume' 컬럼이 없어 처리를 계속할 수 없습니다.")
            return None

        if market in ['KRX', 'NASDAQ']:
            data = data[data['Close'] > 0].copy() 
            
        return data.dropna()
        
    except Exception as e:
        st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# --------------------------
# 4. 모델 훈련 및 예측 함수 (유지)
# --------------------------
def train_and_validate_model(data_features, scaler_type, n_splits):
    
    X_all = data_features.drop('Target', axis=1)
    y = data_features['Target'] 

    X_all.columns = sanitize_columns(X_all.columns)
    
    # 1. 임시 모델 훈련을 위한 TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_importances = pd.Series(0, index=X_all.columns)
    
    st.markdown("##### 🔍 특징 중요도 계산 및 상위 특징 선정 중...")

    # 특징 중요도 계산을 위해 전체 특징을 사용하여 모델 훈련
    for fold, (train_index, val_index) in enumerate(tscv.split(X_all)):
        X_train, X_val = X_all.iloc[train_index], X_all.iloc[val_index]
        y_train = y.iloc[train_index]
        
        # 임시 모델은 스케일링 없이 훈련
        temp_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        # verbose=-1은 LGBM_PARAMS에 포함되어 있으므로, fit() 인자에서 제거함
        temp_model.fit(X_train.values, y_train.values) 
        
        model_importances += pd.Series(temp_model.feature_importances_, index=X_all.columns)
    
    model_importances /= n_splits
    
    # 상위 N개 특징 선정
    top_features_series = model_importances.nlargest(TOP_N_FEATURES)
    top_feature_names = top_features_series.index.tolist()
    top_feature_importances = top_features_series.values
    st.info(f"선택된 상위 특징 ({TOP_N_FEATURES}개): {', '.join(top_feature_names)}")

    # 2. 상위 특징만으로 데이터셋 재구성 및 스케일러/모델 훈련
    X_top = X_all[top_feature_names]
    
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else: 
        scaler = StandardScaler() 
        
    st.info(f"선택된 스케일러: **{scaler_type}**를 사용하여 **상위 {TOP_N_FEATURES}개 특징(X_top)** 데이터에 `fit`하고 전처리합니다.")
    
    # 상위 특징만으로 스케일러 훈련 및 변환 (Scaler가 Top Feature만 기억하게 함)
    X_scaled = scaler.fit_transform(X_top)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_top.index, columns=X_top.columns)
    
    rmse_scores = []
    residual_data = pd.DataFrame()
    final_model = None

    st.markdown("##### 🚀 중앙값 모델 훈련 및 시계열 검증 진행 중 (Top Feature 사용)...")
    progress_bar = st.progress(0)
    
    # 최종 모델 훈련 및 검증 루프 (X_scaled_df는 이미 Top Feature만 포함)
    for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled_df)):
        X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=-1)]
        )
        
        val_predictions = model.predict(X_val.values)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions)) 
        rmse_scores.append(rmse)
        
        # 잔차(Residual) 계산
        residuals = y_val - val_predictions
        
        # 잔차 데이터 축적
        fold_residual_df = pd.DataFrame({
            'Residual': residuals,
            'Fold': f'Fold {fold+1}'
        }, index=y_val.index)
        residual_data = pd.concat([residual_data, fold_residual_df])
        
        progress_bar.progress((fold + 1) / n_splits)
        st.caption(f"Fold {fold+1} 검증 완료. **로그 수익률 RMSE**: {rmse:.6f}")
        final_model = model

    avg_rmse = np.mean(rmse_scores)
    st.success(f"✅ 모델 훈련 완료. 평균 검증 **로그 수익률 RMSE**: {avg_rmse:.6f}")
    
    residual_std = residual_data['Residual'].std()

    return final_model, scaler, top_feature_names, top_feature_importances, avg_rmse, residual_data, X_top, y, residual_std


def predict_future(model, scaler, last_data, feature_columns, residual_std, market_key):
    # Walk-Forward 로직은 유지
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
            # 주식 시장은 주말 건너뛰기
            if next_date.weekday() in [5, 6]:
                day_counter += 1 
                continue
            
        current_prediction_base_price = future_predictions[-1] if future_predictions else last_actual_close
        
        # 가상의 다음 날 데이터 생성
        new_row = pd.DataFrame(index=[next_date])
        new_row['Close'] = current_prediction_base_price
        for col in ['Open', 'High', 'Low', 'Adj Close']:
              new_row[col] = new_row['Close'].iloc[0]
        new_row['Volume'] = last_data['Volume'].iloc[-1] 
              
        temp_df = last_data.iloc[-60:].copy() 
        temp_df.at[temp_df.index[-1], 'Close'] = current_prediction_base_price 
        temp_df = pd.concat([temp_df, new_row])
        
        # 피처 생성
        temp_df_features = create_features(temp_df, is_for_training=False)
        temp_df_features.columns = sanitize_columns(temp_df_features.columns)

        # 상위 특징 12개만 사용
        X_future_data = temp_df_features.iloc[-1].to_frame().T
        X_future = X_future_data[feature_columns].fillna(0) 
        
        # 스케일링
        X_future_scaled = scaler.transform(X_future)
        
        # 중앙값 예측 (로그 수익률)
        log_return_median = model.predict(X_future_scaled)[0] 
        
        # 신뢰구간 계산
        ci_margin = CI_Z_SCORE * residual_std
        log_return_low = log_return_median - ci_margin 
        log_return_high = log_return_median + ci_margin 
        
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
# 5. 시각화 및 분석 함수 (유지)
# --------------------------
def display_feature_importance(feature_columns, importances):
    total_importance = importances.sum()
    if total_importance > 0:
        normalized_importances = (importances / total_importance) * 100
    else:
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
        title=f'모델 특징 중요도 (상위 {TOP_N_FEATURES}개 사용)',
        labels={'Importance': '상대적 중요도 (%)', 'Feature': '특징 이름'},
        height=500
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def display_residual_analysis(residual_data, residual_std):
    st.markdown("##### 🔬 잔차(Residual) 분석 및 신뢰도")
    st.caption(f"잔차의 표준편차: **{residual_std:.6f}** (95% CI는 이 값의 1.96배를 사용하여 계산됩니다.)")

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
# 6. Streamlit 메인 앱 (유지)
# --------------------------
st.set_page_config(layout="wide", page_title="LGBM 멀티 자산 예측 시스템 (최고 속도 최적화)")

def app():
    st.title("🏆 LightGBM 예측 시스템: 최고 속도 최적화 버전")
    st.markdown("**KRX 데이터 로드 부분을 `pykrx`로 전면 교체**하여 `fdr`의 JSON 파싱 오류를 해결했습니다.")
    st.markdown("---")

    # --- 사이드바: 캐시 관리 기능 추가 ---
    with st.sidebar:
        st.markdown("## ⚙️ 설정 및 유지보수")
        if st.button("🔴 Streamlit 캐시 지우고 새로고침", help="데이터 로딩 오류 발생 시 클릭하세요.", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.caption("캐시를 지우면 모든 데이터를 새로 불러옵니다.")
        st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1]) 
    
    clear_cache = False 
    
    with col1:
        selected_market_name = st.selectbox(
            "📊 예측할 자산 선택",
            list(MARKET_MAPPING.values()),
            key='market_select'
        )
    
    # 딕셔너리 값(Value)에서 키(Key) 추출
    market_key = [k for k, v in MARKET_MAPPING.items() if v == selected_market_name][0]

    with col3:
        selected_train_days = st.number_input(
            "📅 훈련기간(단위:일)",
            min_value=120,
            max_value=3650,
            value=DEFAULT_TRAIN_DAYS, 
            step=30,
            key='train_days_input',
            help="모델 훈련에 사용할 과거 데이터 기간 설정 (2~3년 권장)."
        )

    with col4:
        selected_scaler = st.selectbox(
            "⚖️ 스케일러 선택",
            ["RobustScaler", "StandardScaler"],
            key='scaler_select',
            help="특징(X)에만 적용됩니다."
        )
        
    with col5:
        default_n_splits = 3 
        selected_n_splits = st.number_input(
            "✂️ TimeSeriesSplit 분할 수 (k)",
            min_value=2,
            max_value=3, 
            value=default_n_splits, 
            step=1,
            key='n_splits_input',
            help="검증 데이터셋 개수 (속도 향상을 위해 2~3으로 제한)."
        )
        if selected_n_splits > 3:
            st.warning("속도 향상을 위해 Fold 수는 3 이하를 권장합니다.")
            selected_n_splits = 3

    with col2:
        stock_list_df = pd.DataFrame()
        default_ticker = ""

        if market_key == 'KRX':
            stock_list_df = get_stock_listing('KRX', clear_cache=clear_cache) 
            default_ticker = '005930'
            
        elif market_key == 'NASDAQ':
            stock_list_df = get_stock_listing('NASDAQ', clear_cache=clear_cache) 
            default_ticker = 'AAPL'
            
        elif market_key == 'COIN':
            stock_list_df = get_coin_listing(clear_cache=clear_cache)
            default_ticker = 'KRW-BTC'
        
        if not stock_list_df.empty:
            options = stock_list_df['label'].tolist()
            try:
                # KRX는 pykrx 로직에 따라 Name이 '삼성전자'가 아닌 '삼성전자(보통주)' 등일 수 있음
                # 따라서 Code로 Default Index를 찾는 것이 안정적임
                default_label = stock_list_df[stock_list_df['Code'] == default_ticker]['label'].iloc[0]
                default_index = options.index(default_label)
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
            st.warning("선택한 시장의 종목 목록을 불러올 수 없습니다. 캐시를 지우거나 나중에 다시 시도하세요.")
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
            
            raw_data = load_data(selected_ticker, current_market, selected_train_days, clear_cache=clear_cache) 
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
            model_median, scaler, top_feature_names, top_feature_importances, avg_rmse, residual_data, X_top, y, residual_std = train_and_validate_model(
                train_data, selected_scaler, selected_n_splits
            )
            
            st.success(f"✅ 모델 훈련 완료. (잔차 기반 신뢰구간 사용)")

            st.markdown("---")
            st.subheader("💡 훈련 모델 진단")
            
            # 잔차 분석 시각화
            display_residual_analysis(residual_data, residual_std)
            
            # 특징 중요도 시각화
            st.markdown("---")
            display_feature_importance(top_feature_names, top_feature_importances) 

            # 예측 실행
            with st.spinner(f"🔮 미래 {TARGET_PERIOD}일 예측 중 (Walk-Forward, 잔차 기반 95% CI)..."):
                
                last_actual_close = raw_data['Close'].iloc[-1]
                last_data_for_prediction = raw_data.iloc[-100:].copy() 
                
                future_predictions_df = predict_future(
                    model_median, 
                    scaler, 
                    last_data_for_prediction, 
                    top_feature_names, 
                    residual_std,
                    current_market
                )
                
                st.markdown("---")
                st.subheader(f"📈 {selected_label} 가격 예측 시각화 (잔차 기반 95% 신뢰구간)")
                
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
                    title=f'{selected_label} 실제 가격 vs. 예측 가격 및 잔차 기반 95% 신뢰구간',
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
                
                return_pct = (predictions_display['Predicted'] / predictions_display['Predicted'].shift(1)) - 1
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
