import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance 
import urllib.parse
from json.decoder import JSONDecodeError
import FinanceDataReader as fdr
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import MinMaxScaler
import time
from concurrent.futures import ThreadPoolExecutor
import re
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor 

# ------------------------
# ✨ 상수 및 페이지 설정
# ------------------------
st.set_page_config(page_title="🇺🇸 미국 증시 중단기 추세 예측 (개선 V3)", layout="wide")
st.title("🦅 미국 증시 추세 예측 모델 (7가지 개선 반영)")

st.markdown("""
**S&P 500**의 향후 $\mathbf{10}$거래일 누적 수익률을 예측합니다. **뉴스 키워드, 매크로 지연, 브레드/VIX 구조** 등 7가지 성능 개선 요소를 반영했습니다.
""")

# ------------------------
# 0. 매크로 데이터 수집 함수 (발표 지연 shift 적용)
# ------------------------
# ... (get_fred_data, get_fear_greed_index 함수 정의) ... 
@st.cache_data(show_spinner="⏳ FRED 데이터 로드 중...")
def get_fred_data():
    """FRED에서 여러 경제 지표를 병렬로 가져옵니다. (1일 shift 적용)"""
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
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            return ticker, df[['date', 'value']].rename(columns={'value': TICKERS[ticker]}).set_index('date')
        except Exception as e:
            st.warning(f"⚠️ FRED 데이터 로드 실패 ({ticker}): {e}")
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
    """Alternative.me에서 Fear & Greed Index를 가져옵니다. (1일 shift 적용)"""
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        df = df.rename(columns={"value": "FGI", "timestamp": "Date"})
        df = df[["Date", "FGI"]].sort_values("Date").set_index('Date')
        return df.shift(1, freq='D').ffill()
    except Exception as e:
        st.warning(f"⚠️ Fear & Greed Index 로드 오류: {e}")
        return pd.DataFrame()


# ------------------------
# 1. 팩터 및 증시 데이터 로드
# ------------------------
@st.cache_data(show_spinner="⏳ 주가, Breadth, Put/Call, VIX Term 로드 중...")
def load_market_data(start_date, end_date):
    """S&P 500, NASDAQ, VIX, WTI, Copper, Gold, DXY, Breadth, Put/Call 데이터를 로드합니다."""
    load_start_date = start_date - timedelta(days=50) 
    tickers = {
        '^GSPC': 'SP500_Close', '^IXIC': 'NASDAQ_Close', '^VIX': 'VIX', 
        'CL=F': 'WTI', 'GC=F': 'GOLD', 'HG=F': 'COPPER', 'DX-Y.NYB': 'DXY',
        '^VIX9D': 'VIX_9D'
    }
    
    BREADTH_TICKER = '^ADLINE' 
    
    all_data = []
    
    progress_bar = st.progress(0, text="시장 데이터 로드 중...")
    
    for i, (ticker, name) in enumerate(tickers.items()):
        try:
            progress_bar.progress(i / len(tickers), text=f"{name} ({ticker}) 로드 중...")
            df = fdr.DataReader(ticker, start=load_start_date, end=end_date)
            df = df[['Close']].rename(columns={'Close': name})
            df.index = df.index.date
            all_data.append(df)
            time.sleep(0.05)
        except Exception as e:
            st.warning(f"⚠️ {name} ({ticker}) 데이터 로드 실패: {e}")
            continue

    try:
         df_ad = fdr.DataReader(BREADTH_TICKER, start=load_start_date, end=end_date)[['Close']].rename(columns={'Close': 'AD_Line'})
         df_ad.index = df_ad.index.date
         all_data.append(df_ad)
    except Exception:
         st.warning("⚠️ A/D Line 데이터 로드 실패. 피처에서 제외됩니다.")

    progress_bar.empty()
    st.success("✅ 시장 데이터 로드 완료!")
    if not all_data: return pd.DataFrame()
    
    df_merged = pd.concat(all_data, axis=1, join='outer').sort_index()
    df_merged.index.name = 'Date'
    
    if 'DXY' in df_merged.columns:
        df_merged['DXY'] = df_merged['DXY'].shift(1, freq='D').ffill()
        
    return df_merged

# ------------------------
# 2. 감성/키워드 분석 모델 로드 및 함수 정의 (순서 조정)
# ------------------------

# 🚨 NameError 해결: 함수 정의를 먼저 배치
@st.cache_resource
def load_sentiment_model():
    """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
    hf_token = st.secrets.get("HF_TOKEN")
    model_name = "snunlp/KR-FinBert-SC"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰 설정 또는 라이브러리 버전을 확인해주세요.")
        st.stop()
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model() # 👈 호출 (이제 정의가 위에 있음)

def analyze_sentiment(text):
    """Calculates sentiment score for the given text."""
    if not text: return 0.0
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


def get_naver_news_api(query, display=100, start=1, sort="date"): 
    """Naver News Search API에서 데이터를 가져옵니다. (기존 로직 유지)"""
    try:
        client_id = st.secrets.get("naver", {}).get("client_id")
        client_secret = st.secrets.get("naver", {}).get("client_secret")
        if not client_id or not client_secret:
             st.error("❌ 네이버 API 키(client_id/client_secret)가 Streamlit Secrets의 [naver] 섹션에 설정되어 있지 않습니다.")
             return pd.DataFrame(columns=['Date', 'Title']) 
    except Exception as e:
        st.error(f"❌ 네이버 API 키 로드 중 예외 발생: {e}")
        return pd.DataFrame(columns=['Date', 'Title'])

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
            pub_date = item.get('pubDate', '')
            try: pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception: pub_date_dt = None
            news_data.append({'Date': pub_date_dt, 'Title': title})
        return pd.DataFrame(news_data)
    except Exception:
        return pd.DataFrame(columns=['Date', 'Title'])

NEGATIVE_KEYWORDS = [
    "긴축", "금리인상", "매파", "고용둔화", "경기침체", 
    "정책 규제", "파산", "물가 폭등", "인플레이션", "유가 급등"
]
POSITIVE_KEYWORDS = [
    "금리인하", "정책완화", "비둘기파", "AI 투자", "기술주 실적 개선", 
    "경기회복", "유동성 공급", "수요 증가", "소비 호조"
]

NEGATIVE_KEYWORDS_REGEX = r'\b(' + '|'.join(map(re.escape, NEGATIVE_KEYWORDS)) + r')\b'
POSITIVE_KEYWORDS_REGEX = r'\b(' + '|'.join(map(re.escape, POSITIVE_KEYWORDS)) + r')\b'

def extract_news_features(df_news):
    """뉴스 데이터프레임에서 감성 점수와 호재/악재 키워드 카운트를 추출합니다."""
    if df_news.empty:
        return pd.DataFrame(columns=['Date', 'Sentiment_Score', 'Negative_Keyword_Count', 'Positive_Keyword_Count', 'News_Count'])

    # 1. 감성 분석 (보조적으로 유지)
    df_news['Sentiment_Score'] = df_news['Title'].apply(analyze_sentiment)
    
    # 2. 키워드 카운트 (주력)
    df_news['Negative_Keyword_Count'] = df_news['Title'].apply(
        lambda x: len(re.findall(NEGATIVE_KEYWORDS_REGEX, x, re.IGNORECASE))
    )
    df_news['Positive_Keyword_Count'] = df_news['Title'].apply(
        lambda x: len(re.findall(POSITIVE_KEYWORDS_REGEX, x, re.IGNORECASE))
    )
    
    # 3. 일별 집계
    news_grouped = df_news.groupby('Date').agg(
        Sentiment_Score=('Sentiment_Score', 'mean'),
        Negative_Keyword_Count=('Negative_Keyword_Count', 'sum'),
        Positive_Keyword_Count=('Positive_Keyword_Count', 'sum'),
        News_Count=('Title', 'count')
    ).reset_index().set_index('Date')
    
    return news_grouped

# ------------------------
# 3. 피처 엔지니어링 함수 (개선된 피처 포함)
# ------------------------
# ... (create_features 함수 정의) ...
def create_features(df_merge):
    """모든 팩터에 대해 시계열 피처를 생성하고 데이터를 정리합니다."""
    df = df_merge.copy()
    
    if 'NASDAQ_Close' in df.columns and 'SP500_Close' in df.columns:
        df['NASDAQ_SP500_Ratio'] = df['NASDAQ_Close'] / df['SP500_Close']
    
    df['Return_10D'] = df['SP500_Close'].pct_change(periods=10).shift(-10) * 100
    df['Daily_Return'] = df['SP500_Close'].pct_change() * 100

    if 'VIX_9D' in df.columns and 'VIX' in df.columns:
        df['VIX_9D_VIX_Ratio'] = df['VIX_9D'] / df['VIX']
        df['VIX_Term_Structure'] = df['VIX_9D_VIX_Ratio'].apply(lambda x: 1 if x > 1.0 else 0) 
        
    if 'Sentiment_Score' in df.columns:
        df['Sentiment_MA_5D'] = df['Sentiment_Score'].rolling(window=5).mean()
    if 'News_Count' in df.columns:
          df['News_Count_5D'] = df['News_Count'].rolling(window=5).mean()
        
    if 'Negative_Keyword_Count' in df.columns:
        df['Negative_Keyword_MA_5D'] = df['Negative_Keyword_Count'].rolling(window=5).mean()
    if 'Positive_Keyword_Count' in df.columns:
        df['Positive_Keyword_MA_5D'] = df['Positive_Keyword_Count'].rolling(window=5).mean()
        df['Keyword_Net_MA_5D'] = df['Positive_Keyword_Count'] - df['Negative_Keyword_Count']

    lags = [1, 3, 5, 10] 
    
    lag_factors = [
        'Daily_Return', 'VIX', 'FGI', 
        'Sentiment_MA_5D', 'News_Count_5D', 
        'Negative_Keyword_MA_5D', 'Positive_Keyword_MA_5D', 
        'Keyword_Net_MA_5D',
        'YIELD_CURVE', 'BBB_OAS', 'WTI', 'GOLD', 'COPPER',
        'DXY', 'NASDAQ_SP500_Ratio', 'SP500_EPS',
        'VIX_9D_VIX_Ratio', 'AD_Line' 
    ]
    
    for factor in lag_factors:
        if factor in df.columns:
            for lag in lags:
                df[f'{factor}_Lag_{lag}'] = df[factor].shift(lag)
            
    df['VIX_Change_5D'] = df['VIX'].diff(5)
    df['SP500_SMA_20'] = df['SP500_Close'].rolling(window=20).mean()
    
    df = df.dropna()
    
    features_to_include = [f for f in df.columns if 'Lag' in f or 'Change' in f or 'Ratio' in f or 'SMA' in f or f in ['GDP', 'M2', 'SP500_EPS', 'VIX_Term_Structure']]
    features = list(set(features_to_include))
    
    return df, features

# ------------------------
# 4. Streamlit 실행 로직
# ------------------------

@st.cache_resource(show_spinner="🚀 Soft Voting 앙상블 모델 훈련 중/로드 중...")
def train_voting_model(_X_train_df, _y_train, _lgbm_params, _xgb_params, _cat_params, _features, version_key=1):
    lgbm_model = lgb.LGBMRegressor(**_lgbm_params)
    xgb_model = xgb.XGBRegressor(**_xgb_params)
    cat_model = CatBoostRegressor(**_cat_params) 
    
    estimators = [('lgbm', lgbm_model), ('xgb', xgb_model), ('cat', cat_model)]
    weights = [1, 1, 1]

    voting_model = VotingRegressor(
        estimators=estimators,
        weights=weights
    )
    voting_model.fit(_X_train_df, _y_train) 
    
    lgbm_shap_model = lgb.LGBMRegressor(**_lgbm_params)
    lgbm_shap_model.fit(_X_train_df, _y_train)
    
    return voting_model, lgbm_shap_model

LGBM_PARAMS = {'objective': 'regression', 'metric': 'rmse', 'n_estimators': 300, 'learning_rate': 0.01, 'num_leaves': 21, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
XGB_PARAMS = {'objective': 'reg:squarederror', 'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1}
CAT_PARAMS = {'loss_function': 'RMSE', 'iterations': 300, 'learning_rate': 0.05, 'depth': 6, 'random_seed': 42, 'verbose': 0}

st.markdown("---")
# UI 입력 요소
col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    news_query = st.text_input(
        "📰 뉴스 키워드 (OR 연산)", 
        value="미국증시전망 OR 금리인상 OR 연준 OR 경기침체", 
        help="OR 연산으로 연결된 핵심 키워드를 입력하여 관련 기사 200개를 수집합니다."
    )
with col2:
    start_date = st.date_input("분석 시작일", datetime.now() - timedelta(days=365 * 2)) 
with col3:
    end_date = st.date_input("분석 종료일", datetime.now())
    
if st.button("🚀 데이터 로드, 분석 및 예측 시작 (7가지 개선 적용)", type="primary", use_container_width=True):
    
    # 1. 데이터 로드
    market_df = load_market_data(start_date, end_date)
    fred_data = get_fred_data()
    fg_df = get_fear_greed_index(limit=365 * 3)
    
    # 1-2. 뉴스 감성/키워드 분석
    with st.spinner(f"뉴스 크롤링 및 감성/키워드 분석 중... (키워드: {news_query})"):
        # 2회 호출로 최대 200개 기사 수집
        news_batch_1 = get_naver_news_api(news_query, display=100, start=1) 
        news_batch_2 = get_naver_news_api(news_query, display=100, start=101)
        
        all_news = pd.concat([news_batch_1, news_batch_2]).drop_duplicates(subset=['Title']).reset_index(drop=True)
        
        if all_news.empty or 'Date' not in all_news.columns or all_news['Date'].isnull().all():
            st.warning("⚠️ 유효한 기사 데이터 수집 실패. 뉴스 피처를 0으로 채웁니다.")
            news_features_df = pd.DataFrame(columns=['Date', 'Sentiment_Score', 'Negative_Keyword_Count', 'Positive_Keyword_Count', 'News_Count']).set_index('Date')
        else:
            news_features_df = extract_news_features(all_news) 
            st.success(f"✅ 뉴스 감성/키워드 분석 완료! (총 {len(all_news)}개 기사 분석)")

    # 2. 데이터 병합 (개선된 피처 반영)
    df_merge = market_df
    if not fg_df.empty: df_merge = pd.merge(df_merge, fg_df, left_index=True, right_index=True, how='left')
    for name, df_fred in fred_data.items(): df_merge = pd.merge(df_merge, df_fred, left_index=True, right_index=True, how='left')
    if not news_features_df.empty:
        df_merge = pd.merge(df_merge, news_features_df, left_index=True, right_index=True, how='left')
    
    df_merge = df_merge.fillna(method='ffill').fillna(0)
    
    # 3. 피처 엔지니어링 및 데이터 준비
    df_ml, features_full = create_features(df_merge)
    
    df_ml = df_ml.tail(500)
    df_ml = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)]

    if len(df_ml) <= 100:
        st.error("데이터가 부족합니다. 분석 기간을 늘리세요. (최소 100일 필요)")
        st.stop()
        
    X_full = df_ml[features_full]
    y = df_ml['Return_10D'] 
    
    # 4. 피처 선택: LightGBM 중요도 기반
    st.subheader("⚙️ 피처 선택 (LightGBM 중요도 기반 Top 15)")
    
    temp_model = lgb.LGBMRegressor(**LGBM_PARAMS) 
    temp_model.fit(X_full, y)

    feature_importances = pd.Series(temp_model.feature_importances_, index=X_full.columns)
    features = feature_importances.nlargest(15).index.tolist()
    
    st.info(f"선택된 피처 수: **{len(features)}개**.")
    X = df_ml[features] 
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True) 
    X = X[features] 

    scaler = MinMaxScaler()
    X_scaled_all = scaler.fit_transform(X) 
    X_scaled_all_df = pd.DataFrame(X_scaled_all, columns=X.columns, index=X.index)
    
    test_size = max(30, int(0.2 * len(X_scaled_all_df)))
    X_train_df, X_test_df = X_scaled_all_df.iloc[:-test_size], X_scaled_all_df.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    X_train_df = X_train_df[features]
    X_test_df = X_test_df[features]

    # 5. 앙상블 모델 훈련 및 시계열 교차검증 (TS Split)
    st.header("📊 시계열 교차검증 (TimeSeriesSplit)")
    
    n_splits = 3 
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    r2_scores_lgbm = []
    
    with st.spinner(f"⏳ TimeSeriesSplit 교차검증 중..."):
        
        for i, (train_index, val_index) in enumerate(tscv.split(X_train_df)):
            X_train_fold, X_val_fold = X_train_df.iloc[train_index], X_train_df.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            lgbm_fold = lgb.LGBMRegressor(**LGBM_PARAMS)
            
            lgbm_fold.fit(X_train_fold, y_train_fold,
                          eval_set=[(X_val_fold, y_val_fold)],
                          eval_metric='rmse',
                          callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
            
            y_val_pred = lgbm_fold.predict(X_val_fold)
            r2_scores_lgbm.append(r2_score(y_val_fold, y_val_pred))
            
        avg_r2 = np.mean(r2_scores_lgbm)
        st.info(f"✅ TimeSeriesSplit 평균 R² (LGBM 기준): **{avg_r2:.4f}**")
        st.dataframe(pd.DataFrame({'Fold': range(1, n_splits + 1), 'R2 Score': r2_scores_lgbm}), use_container_width=True)
    st.markdown("---")

    voting_model, lgbm_model = train_voting_model(
        X_train_df, 
        y_train, 
        LGBM_PARAMS, 
        XGB_PARAMS, 
        CAT_PARAMS, 
        tuple(features),
        version_key=1 
    )
        
    y_train_pred_lgbm = lgbm_model.predict(X_train_df)
    residuals = y_train - y_train_pred_lgbm
    residual_std = residuals.std()
    CI_FACTOR = 1.645 * residual_std 
    
    y_test_pred = voting_model.predict(X_test_df)
    last_data_df = X_scaled_all_df.iloc[-1][features].to_frame().T 
    next_day_return_pred = voting_model.predict(last_data_df)[0]
    
    low_ci = next_day_return_pred - CI_FACTOR
    high_ci = next_day_return_pred + CI_FACTOR
    
    # 6. 결과 출력
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    st.markdown("---")
    st.header("📈 최종 예측 결과 및 모델 성능")
    
    col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

    def format_pred_value(value): return f"{value:+.2f}%"

    with col_pred1:
        st.metric(label="📊 향후 10거래일 S&P 500 예측 수익률", value=format_pred_value(next_day_return_pred), delta=f"90% CI: {low_ci:+.2f}% ~ {high_ci:+.2f}%")
    with col_pred2:
        st.metric(label="✅ 테스트 R² (앙상블)", value=f"{r2:.2f}", help=f"MSE: {mse:.4f}. 1에 가까울수록 적합도가 높음.")
    with col_pred3:
        current_vix = df_ml['VIX'].iloc[-1]
        vix_trend = "하락 (강세) 🟢" if df_ml['VIX_Change_5D'].iloc[-1] < 0 else "상승 (약세) 🔴"
        st.metric(label="🔥 현재 VIX 지수", value=f"{current_vix:.2f}", delta=vix_trend)
    with col_pred4:
        action = "강력 매수/추세 추종" if next_day_return_pred > 1.0 and low_ci > 0.0 else ("매도/리스크 관리" if next_day_return_pred < -1.0 else "관망/중립")
        action_color = "#D4EDDA" if "매수" in action else ("#F8D7DA" if "매도" in action else "#FFF3CD")
        st.markdown(f"""
        <div style='padding: 10px; border-radius: 5px; text-align: center; 
            background-color: {action_color}; color: {"#155724" if "매수" in action else ("#721C24" if "매도" in action else "#856404")}; 
            font-weight: bold; margin-top: 15px;'>
            최종 투자 시그널: {action}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")

    # 7. SHAP + Permutation Importance 비교
    st.header("💡 예측 해석: SHAP + Permutation Importance")
    
    col_shap, col_perm = st.columns(2)

    # SHAP 분석
    with col_shap:
        st.subheader("1. SHAP (LightGBM)")
        st.markdown(f"모델의 **국소적 기여도** 분석 (최종 예측치 `{next_day_return_pred:+.2f}%` 기여)")
        try:
            explainer = shap.TreeExplainer(lgbm_model) 
            shap_values = explainer.shap_values(last_data_df)
            
            shap_df = pd.DataFrame({
                'Feature': last_data_df.columns,
                'SHAP Value': shap_values[0]
            })
            shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values('Abs SHAP', ascending=False).head(5)

            fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                              color='SHAP Value', color_continuous_scale=px.colors.diverging.RdBu,
                              title="Top 5 SHAP Value", height=400)
            fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_shap, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ SHAP 해석 로드 중 오류 발생: {e}.")

    # Permutation Importance 분석
    with col_perm:
        st.subheader("2. Permutation Importance (LightGBM)")
        st.markdown("테스트 데이터셋에서 **전역적 중요도** 분석 (랜덤하게 섞었을 때 성능 저하)")
        try:
            r = permutation_importance(lgbm_model, X_test_df, y_test,
                                       n_repeats=10,
                                       random_state=42,
                                       n_jobs=-1)
            
            perm_df = pd.DataFrame({
                'Feature': X_test_df.columns[r.importances_mean.argsort()[::-1]],
                'Importance': r.importances_mean[r.importances_mean.argsort()[::-1]]
            }).head(5)
            
            fig_perm = px.bar(perm_df, x='Importance', y='Feature', orientation='h', 
                             title='Top 5 Permutation Importance', height=400,
                             color='Importance', color_continuous_scale=px.colors.sequential.Sunset)
            fig_perm.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_perm, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ Permutation Importance 계산 중 오류: {e}")
            
    st.markdown("---")


    # 8. 피처 상관관계 히트맵 시각화 추가
    st.header("🔗 피처 상관관계 히트맵")
    st.markdown("훈련에 사용된 **LightGBM 중요도 기반 Top 15 피처**와 타겟(`Return_10D`) 간의 상관관계를 시각적으로 확인합니다.")

    correlation_df = df_ml[features + ['Return_10D']].copy().rename(columns={'Return_10D': 'Target_10D_Return'})
    N_TOP_FEATURES = len(features) 
    
    try:
        corr_matrix = correlation_df.corr()
        fig_heatmap = px.imshow(corr_matrix, 
                                 x=corr_matrix.columns, 
                                 y=corr_matrix.columns,
                                 color_continuous_scale='RdBu_r', 
                                 title=f'LGBM 선택 {N_TOP_FEATURES}개 피처 간의 상관관계 히트맵')
        fig_heatmap.update_xaxes(side="top")
        
        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, val in enumerate(row):
                annotations.append(
                    dict(x=corr_matrix.columns[j], y=corr_matrix.columns[i], 
                         text=f"{val:.2f}", showarrow=False, font=dict(color="black" if abs(val) < 0.6 else "white"))
                )
        fig_heatmap.update_layout(annotations=annotations, height=800)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ 히트맵 생성 중 오류: {e}")
        
    st.markdown("---")


    # 9. 주요 매크로 팩터 추이 시각화
    st.header("📊 주요 매크로 팩터 추이 (S&P 500과 비교)")
    
    df_macro_plot = df_ml[df_ml.index >= start_date].copy()

    fig_macro = go.Figure()
    
    fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['SP500_Close'], name='S&P 500 (좌측 축)', line=dict(color='#1f77b4', width=2), yaxis='y1'))
    
    if 'YIELD_CURVE' in df_macro_plot.columns:
        fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['YIELD_CURVE'], name='장단기 금리차 (10Y-2Y)', line=dict(color='red', width=1.5), yaxis='y2', opacity=0.8))
        fig_macro.add_hline(y=0, line_dash="dash", line_color="red", yref="y2")     
    
    if 'BBB_OAS' in df_macro_plot.columns:
        fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['BBB_OAS'], name='BBB 회사채 스프레드', line=dict(color='green', width=1.5), yaxis='y3', opacity=0.8))
        
    if 'DXY' in df_macro_plot.columns:
        fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['DXY'], name='USD Index (DXY)', line=dict(color='purple', width=1.5), yaxis='y4', opacity=0.8))
        
    if 'AD_Line' in df_macro_plot.columns:
        fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['AD_Line'], name='A/D Line (브레드)', line=dict(color='orange', width=1.5), yaxis='y5', opacity=0.8))

    fig_macro.update_layout(title="S&P 500 vs. 경기/신용 리스크 지표", xaxis_title="날짜",
        yaxis=dict(title=dict(text='S&P 500 종가', font=dict(color="#1f77b4")), domain=[0, 1]),
        yaxis2=dict(title=dict(text='금리차 (%)', font=dict(color="red")), overlaying='y', side='right', position=0.85, showgrid=False),
        yaxis3=dict(title=dict(text='BBB OAS', font=dict(color="green")), overlaying='y', side='right', position=0.90, showgrid=False),
        yaxis4=dict(title=dict(text='DXY', font=dict(color="purple")), overlaying='y', side='right', position=0.95, showgrid=False),
        yaxis5=dict(title=dict(text='AD Line', font=dict(color="orange")), overlaying='y', side='right', position=1.0, showgrid=False),
        hovermode="x unified", height=600, legend=dict(x=0, y=1.05, orientation="h"))
    
    st.plotly_chart(fig_macro, use_container_width=True)


    # 10. 예측 vs. 실제 수익률 시각화 (앙상블 모델)
    st.subheader("📈 Soft Voting 앙상블 예측 vs. 실제 수익률 (90% 신뢰구간)")
    
    y_test_df = pd.DataFrame({
        'Actual': y_test, 'Predicted': y_test_pred,
        'Low_CI': y_test_pred - CI_FACTOR, 'High_CI': y_test_pred + CI_FACTOR
    }, index=df_ml.index[-test_size:])

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['High_CI'], mode='lines', line=dict(width=0), showlegend=False))
    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Low_CI'], fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', mode='lines', line=dict(width=0), name='90% 신뢰구간'))
    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Actual'], mode='markers', name='실제 10일 누적 수익률', marker=dict(color='blue', size=5, opacity=0.8)))
    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Predicted'], mode='lines', name='앙상블 예측 수익률 (Median)', line=dict(color='red', width=2)))

    fig_pred.update_layout(title=f"테스트 기간 S&P 500 10일 누적 수익률 예측 결과", xaxis_title="날짜", yaxis_title="수익률(%)", hovermode="x unified", height=500)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # 11. 팩터 중요도 시각화
    st.subheader("🔍 팩터 중요도 (LightGBM 기준)")
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': lgbm_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                     title='LightGBM 모델 상위 15개 팩터 중요도',
                     color='Importance', color_continuous_scale=px.colors.sequential.Viridis)
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)


st.markdown("---")
st.warning("⚠️ **면책 조항:** 이 모델은 교육 및 분석 목적으로만 제공됩니다. 실제 투자에 사용하기 전에 충분한 검증과 리스크 분석을 수행해야 합니다.")
