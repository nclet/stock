# optimized_app.py
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
import os

# --------------------------
# 설정(사용자 조정 가능)
# --------------------------
TARGET_PERIOD = 10               # 예측 영업일 수
QUANTILE_ALPHA = 0.05            # 95% CI -> alpha=0.05
TOP_FEATURES = 12                # 최종 모델에 사용할 상위 피처 개수
DEFAULT_N_SPLITS = 3             # TimeSeriesSplit 기본 분할 수 (속도 목적)
RECENT_YEARS = 3                 # 최근 N년 데이터만 사용

# --------------------------
# LightGBM 파라미터 (속도/정확도 균형)
# 권장: num_leaves 31~63, n_estimators 200~400, learning_rate 0.05~0.08
# --------------------------
LGBM_PARAMS_BASE = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'n_estimators': 300,
    'learning_rate': 0.06,
    'num_leaves': 45,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 50,
    'lambda_l1': 0.2,
    'lambda_l2': 0.3,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# --------------------------
# 헬퍼: 컬럼명 안전화, MACD, RSI (원래 코드와 동일하되 shift 적용 권장)
# --------------------------
def sanitize_columns(columns):
    return [
        str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '').replace('-', '_')
        for col in columns
    ]

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, macd_signal

def calculate_rsi(series, window=14):
    diff = series.diff()
    gain = diff.mask(diff < 0, 0)
    loss = diff.mask(diff > 0, 0).abs()
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(1e-10)))
    return rsi

# --------------------------
# 종목/코인 리스트 로드 (기존과 유사)
# --------------------------
@st.cache_data(ttl=60*60*24)
def get_stock_listing(market_name):
    try:
        code = market_name if market_name in ['KRX', 'NASDAQ'] else None
        if code is None:
            return pd.DataFrame()
        df = fdr.StockListing(code)
        if df is None or df.empty:
            return pd.DataFrame()
        if 'Code' not in df.columns and 'Symbol' in df.columns:
            df.rename(columns={'Symbol': 'Code'}, inplace=True)
        if 'Code' not in df.columns:
            return pd.DataFrame()
        df['Code'] = df['Code'].astype(str)
        name_col = 'Name' if 'Name' in df.columns else ('Symbol' if 'Symbol' in df.columns else df.columns[0])
        df['label'] = df[name_col].astype(str) + ' (' + df['Code'] + ')'
        return df
    except Exception as e:
        # 안전한 예비 처리: 빈 DataFrame 반환
        st.error(f"{market_name} 종목 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60*60*24)
def get_coin_listing():
    try:
        url = "https://api.upbit.com/v1/market/all"
        response = requests.get(url, params={'isDetails': 'false'}, timeout=10)
        response.raise_for_status()
        all_markets = response.json()
        krw_markets = [m for m in all_markets if m['market'].startswith('KRW-')]
        df = pd.DataFrame(krw_markets)
        if df.empty:
            return df
        df.rename(columns={'market': 'Code', 'korean_name': 'Name'}, inplace=True)
        df['label'] = df['Name'].astype(str) + ' (' + df['Code'].str.replace('KRW-', '') + ')'
        return df
    except Exception as e:
        st.error(f"코인 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# --------------------------
# 데이터 로드 (최근 N년으로 제한)
# --------------------------
@st.cache_data(ttl=60*60*4)
def load_data(ticker, market, train_days, recent_years=RECENT_YEARS):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=train_days + 150)
    try:
        if market in ['KRX', 'NASDAQ']:
            df = fdr.DataReader(ticker, start_date, end_date)
            df.index.name = 'Date'
            if 'Close' not in df.columns or 'Volume' not in df.columns:
                st.error("데이터에 'Close' 또는 'Volume' 컬럼이 없습니다.")
                return None
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
        elif market == 'COIN':
            days_diff = (end_date - start_date).days
            count = min(days_diff + 1, 2000)
            df_coin = pyupbit.get_ohlcv(ticker=ticker, interval='day', count=count)
            if df_coin is None or df_coin.empty:
                st.warning(f"오류: [{ticker}] 코인에 대한 데이터를 찾을 수 없습니다.")
                return None
            df_coin.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'value']
            df_coin.index.name = 'Date'
            df_coin['Adj Close'] = df_coin['Close']
            df = df_coin[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
        else:
            return None

        # 최근 recent_years 년 데이터만 사용 (속도 및 최신성)
        cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=365 * recent_years))
        df = df[df.index >= cutoff].copy()

        if df.empty:
            return None
        # 필터링
        if market in ['KRX', 'NASDAQ']:
            df = df[df['Close'] > 0].copy()
        return df.dropna()
    except Exception as e:
        st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# --------------------------
# 피처 생성 (미래 누수 방지: 대부분 shift(1) 적용)
# --------------------------
def create_features(df, is_for_training=True):
    df = df.copy()
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 시간 기반
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear

    # 기본 로그수익률 (shift를 이용해 현재 시점 이전 정보만 사용)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)).shift(0)  # 이전 종가 대비로 현재값 사용

    # Lag features
    lags = [1, 3, 7, 14, 30]
    for lag in lags:
        df[f'LogR_Lag_{lag}'] = df['Log_Return'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

    # 이동평균 및 변동성 (shift(1)로 누수 방지)
    windows = [5, 20, 60]
    for w in windows:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean().shift(1)
        df[f'Vol_{w}'] = df['Close'].rolling(window=w).std().shift(1)

    df['Daily_Change'] = df['Close'].pct_change().shift(1)
    df['Momentum_20'] = df['Close'].pct_change(periods=20).shift(1)
    df['Volatility_20'] = df['Daily_Change'].rolling(window=20).std().shift(1)
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio_20'] = df['Volume_Ratio_20'].shift(1)

    # MACD, RSI (shift(1))
    macd, macd_signal = calculate_macd(df['Close'])
    df['MACD'] = macd.shift(1)
    df['MACD_Signal'] = macd_signal.shift(1)
    df['MACD_Diff'] = (macd - macd_signal).shift(1)
    df['RSI'] = calculate_rsi(df['Close']).shift(1)

    if is_for_training:
        # target: 다음일 로그수익률
        df['Target'] = np.log(df['Close'].shift(-1) / df['Close'])
    df = df.dropna()
    return df

# --------------------------
# 모델 훈련 (1) 초기 CV로 잔차와 중요도 수집, (2) 상위 피처 선택 후 전체 재학습
# --------------------------
def train_with_feature_selection(df_features, scaler_type='RobustScaler', n_splits=DEFAULT_N_SPLITS, top_k=TOP_FEATURES):
    X = df_features.drop(columns=['Target'])
    y = df_features['Target']

    X.columns = sanitize_columns(X.columns)

    scaler = RobustScaler() if scaler_type == "RobustScaler" else StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # 1) TimeSeries CV로 모델 학습(빠르게): residual 수집 및 임포턴스 집계
    tscv = TimeSeriesSplit(n_splits=n_splits)
    residuals_all = []
    importances = np.zeros(X_scaled.shape[1])
    rmses = []

    st.info(f"TimeSeriesSplit n_splits={n_splits}로 빠른 CV 수행 (잔차 수집 및 피처 중요도 집계)")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        params = LGBM_PARAMS_BASE.copy()
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train.values, y_train.values,
                  eval_set=[(X_val.values, y_val.values)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=-1)]
                  )

        # 예측 및 residual
        y_val_pred = model.predict(X_val.values)
        residuals_all.append(y_val - y_val_pred)
        importances += model.feature_importances_
        rmses.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))

        st.caption(f"Fold {fold+1}/{n_splits} 완료: RMSE={rmses[-1]:.6f}")

    # 합치기
    residuals_concat = pd.concat(residuals_all)
    avg_rmse = np.mean(rmses)
    st.success(f"CV 완료: 평균 RMSE = {avg_rmse:.6f}")

    # 피처 중요도 상위 top_k 선택
    feature_names = X_scaled.columns
    importances_series = pd.Series(importances, index=feature_names)
    top_features = importances_series.nlargest(top_k).index.tolist()
    st.info(f"선택된 상위 {top_k}개 피처: {top_features}")

    # 2) 최종 모델: top_features로 전체 데이터 재학습 (더 안정적)
    X_final = X_scaled[top_features]
    final_model = lgb.LGBMRegressor(**LGBM_PARAMS_BASE)
    final_model.fit(X_final.values, y.values)

    st.success("최종 모델(전체 데이터, top features) 학습 완료")

    # 잔차 기반 신뢰구간 기준값(quantiles)
    q_low = np.quantile(residuals_concat.values, QUANTILE_ALPHA / 2)
    q_high = np.quantile(residuals_concat.values, 1 - (QUANTILE_ALPHA / 2))

    # 요약 정보 반환
    return {
        'model': final_model,
        'scaler': scaler,
        'feature_columns': top_features,
        'avg_rmse': avg_rmse,
        'residuals': residuals_concat,
        'q_low': q_low,
        'q_high': q_high
    }

# --------------------------
# 미래 예측: single model + residual quantile로 CI 산출
# --------------------------
def predict_future_with_ci(trained_dict, last_data, market_key):
    model = trained_dict['model']
    scaler = trained_dict['scaler']
    feature_columns = trained_dict['feature_columns']
    q_low = trained_dict['q_low']
    q_high = trained_dict['q_high']

    # append rolling window of last_data and iteratively generate future rows (recursive using predicted price)
    current_date = last_data.index[-1]
    last_actual_close = last_data['Close'].iloc[-1]

    future_dates = []
    preds = []
    low_ci = []
    high_ci = []

    # We'll keep a working copy for feature generation (most recent 120 days)
    work_df = last_data.copy()

    day_counter = 1
    while len(preds) < TARGET_PERIOD:
        next_date = current_date + datetime.timedelta(days=day_counter)
        # Skip weekends for 주식시장
        if market_key in ['KRX', 'NASDAQ'] and next_date.weekday() in [5, 6]:
            day_counter += 1
            continue

        # create new row with last close as base (recursive)
        base_price = preds[-1] if preds else last_actual_close
        new_row = pd.DataFrame(index=[next_date])
        new_row['Close'] = base_price
        for col in ['Open', 'High', 'Low', 'Adj Close']:
            new_row[col] = base_price
        new_row['Volume'] = work_df['Volume'].iloc[-1]

        # append and regenerate features for the extended frame
        temp = pd.concat([work_df, new_row])
        temp_features = create_features(temp, is_for_training=False)
        # select last row and chosen features
        X_future = temp_features.iloc[[-1]].copy()
        X_future.columns = sanitize_columns(X_future.columns)
        # Keep missing feature columns filled with 0
        for col in feature_columns:
            if col not in X_future.columns:
                X_future[col] = 0.0
        X_future = X_future[feature_columns].fillna(0.0)

        # scale using scaler (note: scaler was fit on full original X)
        X_future_scaled = scaler.transform(X_future)

        # predict median log-return
        log_return_pred = model.predict(X_future_scaled)[0]

        # CI by adding quantile offsets
        log_return_low = log_return_pred + q_low
        log_return_high = log_return_pred + q_high

        # convert to price
        next_price = base_price * np.exp(log_return_pred)
        next_price_low = base_price * np.exp(log_return_low)
        next_price_high = base_price * np.exp(log_return_high)

        # store
        future_dates.append(next_date)
        preds.append(next_price)
        low_ci.append(next_price_low)
        high_ci.append(next_price_high)

        # update work_df
        work_df = pd.concat([work_df, new_row])
        current_date = next_date
        day_counter = 1

    df_out = pd.DataFrame({
        'Predicted': preds,
        'Low_CI': low_ci,
        'High_CI': high_ci
    }, index=future_dates)
    return df_out

# --------------------------
# 시각화 보조 함수 (기존 로직을 재사용)
# --------------------------
def display_feature_importance(model, feature_columns):
    importances = model.feature_importances_
    total = importances.sum()
    if total > 0:
        normalized = (importances / total) * 100
    else:
        normalized = importances
    df = pd.DataFrame({'Feature': feature_columns, 'Importance': normalized}).sort_values('Importance', ascending=False)
    fig = px.bar(df, x='Importance', y='Feature', orientation='h', height=450, title='모델 특징 중요도 (Top features)')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def display_residual_analysis_from_series(residual_series):
    st.markdown("##### 🔬 잔차(Residual) 분석 (CV 기반)")
    st.caption("잔차는 (실제 로그수익률 - 예측 로그수익률) 입니다.")

    resid = residual_series
    fig = px.histogram(resid, nbins=60, marginal='box', title='잔차 분포 (로그 수익률)')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(layout="wide", page_title="Optimized LGBM 예측 시스템")

def app():
    st.title("⚡ 속도 최적화된 LGBM 예측 시스템 (잔차 기반 CI, Top-K 피처)")
    st.markdown("변경사항: Quantile 3회 학습 → 1회 중앙값 학습 + 잔차로 CI 산출, 상위 피처만으로 최종 재학습, CV 분할수 축소, 최근 N년 데이터만 사용.")

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        recent_years = st.number_input("최근 데이터 사용 연수 (years)", min_value=1, max_value=10, value=RECENT_YEARS)
        top_k = st.number_input("Top 피처 개수", min_value=5, max_value=30, value=TOP_FEATURES)
        n_splits = st.number_input("TimeSeriesSplit 분할 수", min_value=2, max_value=6, value=DEFAULT_N_SPLITS)
        scaler_choice = st.selectbox("스케일러", ["RobustScaler", "StandardScaler"])
        st.markdown("---")
        st.button("🔴 Streamlit 캐시 지우기", on_click=st.cache_data.clear)

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        market = st.selectbox("시장", ["KRX", "NASDAQ", "COIN"])
    with col2:
        train_days = st.number_input("훈련기간(일)", min_value=120, max_value=3650, value=365*recent_years, step=30)
    with col3:
        ticker_input = st.text_input("티커 (예: KRX:005930 or AAPL or KRW-BTC)", value="005930" if market == "KRX" else ("AAPL" if market == "NASDAQ" else "KRW-BTC"))

    run_btn = st.button("모델 훈련 및 예측 실행")

    if not run_btn:
        st.info("설정을 조정한 뒤 '모델 훈련 및 예측 실행' 버튼을 눌러주세요.")
        return

    # 표기 정리
    if market == 'KRX':
        ticker = ticker_input.strip()
        market_key = 'KRX'
    elif market == 'NASDAQ':
        ticker = ticker_input.strip().upper()
        market_key = 'NASDAQ'
    else:
        ticker = ticker_input.strip()
        market_key = 'COIN'

    with st.spinner("데이터 로드 및 피처 생성 중..."):
        raw_data = load_data(ticker, market_key, train_days, recent_years=recent_years)
        if raw_data is None:
            st.error("데이터를 불러오지 못했습니다.")
            return
        data_features = create_features(raw_data, is_for_training=True)
        if len(data_features) < 120:
            st.error(f"피처 생성 후 데이터가 너무 적습니다 ({len(data_features)}일). 훈련기간을 늘려주세요.")
            return

    st.subheader(f"분석: {ticker} (최근 {recent_years}년 데이터 사용)")

    # 모델 학습 (feature selection 포함)
    with st.spinner("모델 학습(빠른 CV -> top-k 선택 -> 전체 재학습) 진행 중..."):
        trained = train_with_feature_selection(data_features, scaler_type=scaler_choice, n_splits=n_splits, top_k=top_k)

    # 진단
    st.markdown("---")
    st.subheader("모델 진단")
    st.markdown(f"- 평균 CV RMSE: **{trained['avg_rmse']:.6f}**")
    display_feature_importance(trained['model'], trained['feature_columns'])
    display_residual_analysis_from_series(trained['residuals'])

    # 미래 예측
    with st.spinner(f"미래 {TARGET_PERIOD}일 예측 중..."):
        last_for_pred = raw_data.iloc[-200:].copy()
        future_df = predict_future_with_ci(trained, last_for_pred, market_key)

    # 시각화
    st.markdown("---")
    st.subheader("예측 결과 (중앙값 + 95% CI)")
    past_prices = raw_data['Close'].iloc[-90:]
    predicted_df = pd.DataFrame({'Actual': past_prices, 'Predicted': np.nan, 'Low_CI': np.nan, 'High_CI': np.nan})
    final_df = pd.concat([predicted_df, future_df]).sort_index()

    fig = go.Figure()
    # CI
    fig.add_trace(go.Scatter(x=final_df.index, y=final_df['High_CI'], fill=None, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Low_CI'], fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(255,0,0,0.12)', name='95% CI'))
    # Pred and actual
    fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Predicted'], mode='lines', name='예측 (Median)', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Actual'], mode='lines', name='실제', line=dict(color='blue')))

    fig.update_layout(title=f"{ticker} 실제 vs 예측 (95% CI)", yaxis_title="가격", xaxis_title="날짜", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 예측 표
    st.markdown("##### 향후 예측표")
    display_df = future_df.copy()
    currency = "원" if market_key in ['KRX', 'COIN'] else "$"
    # 일별 수익률
    returns = (display_df['Predicted'] / display_df['Predicted'].shift(1)) - 1
    returns.iloc[0] = (display_df['Predicted'].iloc[0] / raw_data['Close'].iloc[-1]) - 1
    display_df['Daily Return (%)'] = returns * 100
    display_df.rename(columns={'Predicted': 'Predicted (Median)', 'Low_CI': '95% CI Low', 'High_CI': '95% CI High'}, inplace=True)

    st.dataframe(display_df[['Predicted (Median)', '95% CI Low', '95% CI High', 'Daily Return (%)']].style.format({
        'Predicted (Median)': f'{currency}{{:,.2f}}',
        '95% CI Low': f'{currency}{{:,.2f}}',
        '95% CI High': f'{currency}{{:,.2f}}',
        'Daily Return (%)': '{:.2f}%'
    }))

    # 결과 다운로드
    csv = display_df.to_csv(index=True)
    st.download_button("📥 예측 결과 다운로드 (CSV)", csv.encode(), f"{ticker}_predictions.csv")

if __name__ == "__main__":
    app()
