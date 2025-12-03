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
import re
import shap
from sklearn.inspection import permutation_importance
import logging

# 로깅 레벨 설정
logging.basicConfig(level=logging.INFO)

# ------------------------
# ✨ 상수 및 페이지 설정
# ------------------------
st.set_page_config(page_title="🇺🇸 미국 증시 중단기 추세 예측 (피처 안정화)", layout="wide")
st.title("🦅 미국 증시 추세 예측 모델 (10일 누적 수익률 예측)")

st.markdown("""
**S&P 500**의 **향후 $\mathbf{10}$거래일 누적 수익률**을 예측합니다. **Rolling Feature Importance**를 적용하여 피처 선택의 안정성을 높였습니다.
""")

# ------------------------
# 뉴스 키워드 상수 정의 (Feature 2, 3)
# ------------------------
POSITIVE_KEYWORDS = ['긍정', '상승', '호재', '기대', '강세', '돌파', '매수', '낙관', '수혜', '성장', '회복', '최고', '상향']
NEGATIVE_KEYWORDS = ['부정', '하락', '악재', '우려', '약세', '침체', '매도', '비관', '리스크', '경고', '인하', '폭락', '충격', '경색']
FED_ECONOMIC_KEYWORDS = ['연준', '금리', 'FOMC', '인상', '인하', '테이퍼링', '수급', '유동성', '물가', '인플레이션', '경기둔화']

# ------------------------
# LightGBM 파라미터
# ------------------------
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
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# ------------------------
# 0. 도우미 함수
# ------------------------
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

# ------------------------
# 1. 데이터 수집 함수
# ------------------------
@st.cache_data(show_spinner="⏳ FRED 데이터 로드 중...")
def get_fred_data():
    try:
        fred_api_key = st.secrets.get("fred", {}).get("FRED_API_KEY")
        if not fred_api_key:
             st.warning("⚠️ FRED API 키가 설정되지 않아 데이터를 로드할 수 없습니다.")
             return {}
    except Exception:
        return {}

    TICKERS = {
        "DGS10": "10Y", "DGS2": "2Y", 
        "BAMLC0A4CBBB": "BBB_OAS", "M2SL": "M2", "GDPC1": "GDP",
        "SP500PE": "SP500_EPS"
    }
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def fetch_single_fred(ticker, observation_start):
        params = {
            "series_id": ticker, "api_key": fred_api_key, "file_type": "json", 
            "observation_start": observation_start.strftime("%Y-%m-%d")
        }
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json().get('observations', [])
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date']).dt.normalize().dt.tz_localize(None)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            return ticker, df[['date', 'value']].rename(columns={'value': TICKERS[ticker]}).set_index('date')
        except Exception as e:
            return ticker, pd.DataFrame()

    start_date = datetime.now().date() - timedelta(days=365 * 3)
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_single_fred, ticker, start_date): ticker for ticker in TICKERS.keys()}
        for future in futures:
            ticker, df = future.result()
            if not df.empty: 
                 results[TICKERS[ticker]] = df.shift(1, freq='D').ffill()
    
    if '10Y' in results and '2Y' in results:
        df_yield = pd.merge(results['10Y'], results['2Y'], left_index=True, right_index=True, how='inner')
        results['YIELD_CURVE'] = (df_yield['10Y'] - df_yield['2Y']).rename('YIELD_CURVE').to_frame()
    return results

@st.cache_data(show_spinner="⏳ Fear & Greed Index 로드 중...")
def get_fear_greed_index(limit=1095): 
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize().dt.tz_localize(None)
        df = df.rename(columns={"value": "FGI", "timestamp": "Date"})
        df = df[["Date", "FGI"]].sort_values("Date").set_index('Date')
        return df.shift(1, freq='D').ffill()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60*60*24)
def get_stock_listing(market_name, clear_cache=False):
    if market_name == 'KRX': market_code = 'KRX'
    elif market_name == 'NASDAQ': market_code = 'NASDAQ'
    else: return pd.DataFrame()
        
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
def get_coin_listing(clear_cache=False):
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

@st.cache_data(ttl=60*60*4)
def load_data(ticker, market, train_days, clear_cache=False):
    end_date = datetime.now().date() # datetime.date.today() 대신 수정
    start_date = end_date - timedelta(days=train_days + 150)
    data = None
    try:
        if market in ['KRX', 'NASDAQ']:
            data = fdr.DataReader(ticker, start_date, end_date)
            data.index.name = 'Date'
            if 'Close' not in data.columns: return None
            if 'Adj Close' not in data.columns: data['Adj Close'] = data['Close']
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        elif market == 'COIN':
            days_diff = (end_date - start_date).days
            count = days_diff + 1
            df_coin = pyupbit.get_ohlcv(ticker=ticker, interval='day', count=count)
            if df_coin is None or df_coin.empty: return None
            df_coin.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'value']
            df_coin.index.name = 'Date'
            df_coin['Adj Close'] = df_coin['Close']
            data = df_coin[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        if data is None or data.empty: return None
        if market in ['KRX', 'NASDAQ']: data = data[data['Close'] > 0].copy()
        
        # 인덱스 시간대 제거 및 정규화
        data.index = pd.to_datetime(data.index).normalize().tz_localize(None)
        return data.dropna()
    except Exception as e:
        st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

@st.cache_data(show_spinner="⏳ 주가, 원자재, DXY, NASDAQ 데이터 로드 중...")
def load_market_data(start_date, end_date):
    load_start_date = start_date - timedelta(days=50) 
    tickers = {
        '^GSPC': 'SP500_Close', '^IXIC': 'NASDAQ_Close', '^VIX': 'VIX', 
        'CL=F': 'WTI', 'GC=F': 'GOLD', 'HG=F': 'COPPER', 'DX-Y.NYB': 'DXY'
    }
    all_data = []
    total_tickers = len(tickers)
    progress_bar = st.progress(0, text="시장 데이터 로드 중...")
    
    for i, (ticker, name) in enumerate(tickers.items()):
        try:
            progress_bar.progress((i + 1) / total_tickers, text=f"{name} ({ticker}) 로드 중...")
            df = fdr.DataReader(ticker, start=load_start_date, end=end_date)
            df = df[['Close']].rename(columns={'Close': name})
            df.index = pd.to_datetime(df.index).normalize().dt.tz_localize(None)
            all_data.append(df)
            time.sleep(0.05)
        except Exception as e:
            continue
            
    progress_bar.empty()
    st.success("✅ 시장 데이터 로드 완료!")
    if not all_data: return pd.DataFrame()
    df_merged = pd.concat(all_data, axis=1, join='outer').sort_index()
    df_merged.index.name = 'Date'
    if 'DXY' in df_merged.columns:
        df_merged['DXY'] = df_merged['DXY'].shift(1, freq='D').ffill()
    return df_merged

# ------------------------
# 2. 감성 분석 모델 로드 및 함수
# ------------------------
@st.cache_resource
def load_sentiment_model():
    hf_token = st.secrets.get("HF_TOKEN")
    model_name = "snunlp/KR-FinBert-SC"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        return None, None, None 
    
tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    if not text or not sentiment_model: return 0.0 
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = sentiment_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        neg_idx, pos_idx = None, None
        for idx, label in sentiment_model.config.id2label.items():
            if 'negative' in label.lower() or '부정' in label: neg_idx = idx
            elif 'positive' in label.lower() or '긍정' in label: pos_idx = idx
        negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
        positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0
        return positive_score - negative_score
    except Exception as e:
        return 0.0

def analyze_text_keywords(title, description):
    text = title + " " + description
    pos_count = sum(text.count(word) for word in POSITIVE_KEYWORDS)
    neg_count = sum(text.count(word) for word in NEGATIVE_KEYWORDS)
    pos_neg_ratio = (pos_count + 1) / (neg_count + 1)
    fed_count = sum(text.count(word) for word in FED_ECONOMIC_KEYWORDS)
    total_words = len(text.split())
    fed_ratio = fed_count / total_words if total_words > 0 else 0
    return pos_count, neg_count, pos_neg_ratio, fed_ratio

def get_naver_news_api(query, display=100, start=1, sort="date"): 
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError:
        return pd.DataFrame(columns=['Date', 'Title', 'Description']) 

    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        data = response.json()
        items = data.get('items', [])
        news_data = []
        for item in items:
            title = re.sub('<[^<]+?>', '', item.get('title', ''))
            description = re.sub('<[^<]+?>', '', item.get('description', ''))
            pub_date = item.get('pubDate', '')
            try: pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception: pub_date_dt = None
            news_data.append({'Date': pub_date_dt, 'Title': title, 'Description': description})
        return pd.DataFrame(news_data)
    except Exception as e:
        return pd.DataFrame(columns=['Date', 'Title', 'Description'])

# ------------------------
# 3. 피처 엔지니어링 함수
# ------------------------
def create_features(df_merge):
    df = df_merge.copy()
    
    if 'NASDAQ_Close' in df.columns and 'SP500_Close' in df.columns:
        df['NASDAQ_SP500_Ratio'] = df['NASDAQ_Close'] / df['SP500_Close']
    
    df['Return_10D'] = df['SP500_Close'].pct_change(periods=10).shift(-10) * 100
    df['Daily_Return'] = df['SP500_Close'].pct_change() * 100

    MACRO_FEATURES_TO_ENHANCE = ['YIELD_CURVE', 'BBB_OAS', 'DXY', 'VIX', 'WTI', 'GOLD', 'COPPER']
    for col in MACRO_FEATURES_TO_ENHANCE:
        if col in df.columns:
            df[f'{col}_PCT_CHANGE'] = df[col].pct_change()
            window = 60
            if df[col].rolling(window=window).std().iloc[-1] != 0:
                df[f'{col}_ZSCORE_60D'] = (df[col] - df[col].rolling(window=window).mean()) / df[col].rolling(window=window).std()
            else:
                df[f'{col}_ZSCORE_60D'] = 0
            df.rename(columns={col: f'{col}_RAW'}, inplace=True)
    
    news_agg_features = ['Sentiment_Score', 'News_Count', 'Avg_Pos_Neg_Ratio', 'Avg_Fed_Ratio']
    ma_windows = [3, 5, 10]
    for feature in news_agg_features:
        if feature in df.columns:
            for window in ma_windows:
                df[f'{feature}_MA_{window}D'] = df[feature].rolling(window=window, min_periods=1).mean()
    
    if 'News_Count' in df.columns:
        df['News_Count_Vol_1D'] = df['News_Count'].diff(1).abs()
        df['News_Count_Change_1D'] = df['News_Count'].diff(1)

    lags = [1, 3, 5, 10] 
    lag_factors = ['Daily_Return', 'FGI', 'NASDAQ_SP500_Ratio', 'SP500_EPS', 'GDP', 'M2']
    for col in MACRO_FEATURES_TO_ENHANCE:
        if f'{col}_RAW' in df.columns:
            lag_factors.append(f'{col}_RAW')
            lag_factors.append(f'{col}_PCT_CHANGE')
            lag_factors.append(f'{col}_ZSCORE_60D')

    new_news_factors = [col for col in df.columns if col.startswith(tuple(news_agg_features)) or col.startswith('News_Count_Vol') or col.startswith('News_Count_Change')]
    lag_factors.extend(new_news_factors)
    lag_factors = list(set(f for f in lag_factors if f in df.columns))
    
    for factor in lag_factors:
        for lag in lags:
            df[f'{factor}_Lag_{lag}'] = df[factor].shift(lag)
            
    if 'VIX_RAW' in df.columns:
        df['VIX_RAW_Change_5D'] = df['VIX_RAW'].diff(5)
    df['SP500_SMA_20'] = df['SP500_Close'].rolling(window=20).mean()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=['Return_10D']).dropna()
    
    base_features = [col for col in df.columns if not col.endswith(('Return', 'Close', '10D', '_2Y', '_10Y')) and 'SP500_' not in col and 'NASDAQ_' not in col]
    features = [f for f in base_features if ('Lag' in f or 'Change' in f or 'SMA' in f or f in ['GDP', 'M2', 'SP500_EPS', 'NASDAQ_SP500_Ratio'])]
    features.extend([f for f in base_features if any(f.startswith(macro) for macro in MACRO_FEATURES_TO_ENHANCE)])
    features.extend([f for f in base_features if any(f.startswith(news_f) for news_f in news_agg_features) or f.startswith('News_Count_Vol') or f.startswith('News_Count_Change')])
    features = list(set(features))
    
    return df, features

# --------------------------
# 4. 모델 훈련 및 예측 함수 (수정됨)
# --------------------------
def train_and_validate_model(data_features, scaler_type, n_splits, top_n_features):
    
    X = data_features.drop('Target', axis=1)
    y = data_features['Target']

    X.columns = sanitize_columns(X.columns)
    
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    st.info(f"선택된 스케일러: **{scaler_type}**를 사용하여 **특징(X)** 데이터를 전처리합니다.")
    
    # **[핵심 수정 1]** 먼저 피처 선택(Feature Selection)을 수행합니다.
    # 전체 데이터를 사용하여 임시 모델을 학습하고 중요도를 계산합니다.
    temp_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    temp_model.fit(X, y)
    
    feature_importances = pd.Series(temp_model.feature_importances_, index=X.columns)
    selected_features = feature_importances.nlargest(top_n_features).index.tolist()
    
    st.info(f"선택된 상위 {len(selected_features)}개 특징({top_n_features}개 설정)만 사용하여 최종 모델 및 예측을 진행합니다.")
    
    # **[핵심 수정 2]** 선택된 피처로 데이터셋을 필터링합니다.
    X_filtered = X[selected_features]
    
    # **[핵심 수정 3]** 필터링된 데이터셋(X_filtered)에 대해 스케일러를 fit_transform 합니다.
    # 이제 scaler는 selected_features에 대한 정보만 기억합니다.
    X_scaled = scaler.fit_transform(X_filtered)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X_filtered.columns)
    
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
        model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=-1)]
        )
        
        val_predictions = model.predict(X_val.values)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        rmse_scores.append(rmse)
        
        residuals = y_val - val_predictions
        actual_return_rmse = np.sqrt(np.mean((np.expm1(y_val) - np.expm1(val_predictions))**2)) * 100
        
        fold_residual_df = pd.DataFrame({
            'Residual': residuals,
            'Fold': f'Fold {fold+1}',
            'Target': y_val
        })
        fold_residual_df.index = y_val.index
        residual_data = pd.concat([residual_data, fold_residual_df])
        
        progress_bar.progress((fold + 1) / n_splits)
        final_model = model

    avg_rmse = np.mean(rmse_scores)
    st.success(f"✅ 모델 훈련 완료. 평균 검증 **로그 수익률 RMSE**: {avg_rmse:.6f}")
    
    # 반환 값: 최종 모델, 학습된 스케일러, 선택된 피처 목록, RMSE, 잔차, 필터링된 원본 데이터(X), 타겟(y)
    return final_model, scaler, selected_features, avg_rmse, residual_data, X_filtered, y

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
        
        new_row = pd.DataFrame(index=[next_date])
        new_row['Close'] = current_prediction_base_price
        for col in ['Open', 'High', 'Low', 'Adj Close']:
             new_row[col] = new_row['Close'].iloc[0]
        new_row['Volume'] = last_data['Volume'].iloc[-1]
        
        temp_df = last_data.iloc[-60:].copy()
        temp_df.at[temp_df.index[-1], 'Close'] = current_prediction_base_price
        temp_df = pd.concat([temp_df, new_row])
        
        temp_df_features = create_features(temp_df, is_for_training=False)
        temp_df_features.columns = sanitize_columns(temp_df_features.columns)

        X_future_data = temp_df_features.iloc[-1].to_frame().T
        # **[핵심]** 여기서 feature_columns(선택된 피처들)만 딱 골라서 사용합니다.
        X_future = X_future_data[feature_columns].fillna(0)
        
        # **[핵심]** 이제 scaler는 feature_columns에 대해서만 학습되었으므로 오류가 나지 않습니다.
        X_future_scaled = scaler.transform(X_future)
        
        log_return_median = models['median'].predict(X_future_scaled)[0]
        log_return_low = models['low'].predict(X_future_scaled)[0]
        log_return_high = models['high'].predict(X_future_scaled)[0]
        
        next_price_median = current_prediction_base_price * np.exp(log_return_median)
        next_price_low = current_prediction_base_price * np.exp(log_return_low)
        next_price_high = current_prediction_base_price * np.exp(log_return_high)
        
        future_predictions.append(next_price_median)
        future_low.append(next_price_low)
        future_high.append(next_price_high)
        future_dates.append(next_date)
        
        new_row_for_history = new_row.copy()
        new_row_for_history['Close'] = next_price_median 
        last_data = pd.concat([last_data, new_row_for_history])
        
        current_date = next_date 
        day_counter = 1
    
    return pd.DataFrame({
        'Predicted': future_predictions,
        'Low_CI': future_low,
        'High_CI': future_high
    }, index=future_dates)

# --------------------------
# 5. 시각화 및 분석 함수 (기존과 동일)
# --------------------------
def display_feature_importance(model, feature_columns):
    importances = model.feature_importances_
    feature_names = model.feature_name_
    importance_mapping = dict(zip(feature_names, importances))
    filtered_importances = []
    for col in feature_columns:
        filtered_importances.append(importance_mapping.get(col, 0))
    total_importance = sum(filtered_importances)
    if total_importance > 0:
        normalized_importances = (np.array(filtered_importances) / total_importance) * 100
    else:
        normalized_importances = np.array(filtered_importances)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': normalized_importances
    }).sort_values(by='Importance', ascending=False)
    fig = px.bar(
        feature_importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title=f'모델 특징 중요도 (상위 {len(feature_columns)}개)',
        labels={'Importance': '상대적 중요도 (%)', 'Feature': '특징 이름'},
        height=500
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def display_residual_analysis(residual_data):
    st.markdown("##### 🔬 잔차(Residual) 분석")
    st.caption("잔차는 **실제 로그 수익률 - 예측 로그 수익률**이며, 잔차의 분포는 모델의 학습 신뢰도를 나타냅니다.")
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
st.set_page_config(layout="wide", page_title="LGBM 멀티 자산 예측 시스템 (훈련 시간 최적화)")

def app():
    st.title("🏆 LightGBM 예측 시스템: 훈련 시간 최적화 버전")
    st.markdown("**훈련 시간 단축**을 위해 특징 개수 축소 및 교차 검증 분할 수를 줄였습니다.")
    st.markdown("---")

    # --- 사이드바: 캐시 관리 기능 추가 ---
    with st.sidebar:
        st.markdown("## ⚙️ 설정 및 유지보수")
        if st.button("🔴 Streamlit 캐시 지우고 새로고침", help="데이터 로딩 오류 발생 시 클릭하세요.", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.caption("캐시를 지우면 모든 데이터를 새로 불러옵니다.")
        st.markdown("---")
    # ------------------------------------

    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 1, 1, 1])
    
    clear_cache = False
    
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
        default_n_splits = 3
        selected_n_splits = st.number_input(
            "✂️ TimeSeriesSplit 분할 수 (k)",
            min_value=3,
            max_value=10,
            value=default_n_splits,
            step=1,
            key='n_splits_input',
            help="검증 데이터셋 개수. (작을수록 빠름)"
        )
        
    with col6:
        selected_top_n_features = st.number_input(
            "🔝 Top N 특징 개수",
            min_value=10,
            max_value=100,
            value=TOP_N_FEATURES_DEFAULT,
            step=5,
            key='top_n_features_input',
            help="모델 정확도를 위해 사용될 상위 특징 개수 설정. (작을수록 빠름)"
        )


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

    if run_button and (not selected_ticker or selected_train_days < 120 or selected_n_splits < 3 or selected_top_n_features < 10):
        st.warning("예측할 종목을 선택하고, 필수 설정값을 확인해주세요. (훈련기간 최소 120일, 분할 수 최소 3, 특징 개수 최소 10)")
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
            # [수정 완료] 모델 훈련, 특징 선택, 스케일러 학습이 이 함수 내에서 순차적으로 이루어짐
            model_median, scaler, feature_columns, avg_rmse, residual_data, X_raw, y_raw = train_and_validate_model(
                train_data, selected_scaler, selected_n_splits, selected_top_n_features
            )
            
            # 2. 신뢰구간 (CI) 모델 훈련
            models = {'median': model_median}
            
            LGBM_QUANTILE_PARAMS = LGBM_PARAMS.copy()
            if 'objective' in LGBM_QUANTILE_PARAMS:
                del LGBM_QUANTILE_PARAMS['objective']
            
            # [오류 수정] X_raw는 이미 상위 특징만 포함하고 있으므로 바로 변환
            X_train_scaled = scaler.transform(X_raw).astype('float32')
            y_train_values = y_raw.values
            
            st.markdown("#### 🥈 신뢰구간 모델 훈련 (Quantile Regression)")
            with st.spinner("⏳ 95% 신뢰구간 하한선(Low CI) 모델 훈련 중..."):
                lgbm_low = lgb.LGBMRegressor(objective='quantile', alpha=QUANTILE_ALPHA/2, **LGBM_QUANTILE_PARAMS).fit(
                    X_train_scaled, y_train_values
                )
                models['low'] = lgbm_low
            
            with st.spinner("⏳ 95% 신뢰구간 상한선(High CI) 모델 훈련 중..."):
                lgbm_high = lgb.LGBMRegressor(objective='quantile', alpha=1-(QUANTILE_ALPHA/2), **LGBM_QUANTILE_PARAMS).fit(
                    X_train_scaled, y_train_values
                )
                models['high'] = lgbm_high
            st.success("✅ 퀀타일 회귀 모델 훈련 완료.")

            st.markdown("---")
            st.subheader("💡 훈련 모델 진단")
            
            display_residual_analysis(residual_data)
            
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
                    feature_columns, # 필터링된 특징 컬럼 사용 (scaler와 호환됨)
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
                
                fig = go.Figure()
                
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
                
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Predicted'], mode='lines', name='예측 종가 (Median)', line=dict(color='red', dash='dot')))
                
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
