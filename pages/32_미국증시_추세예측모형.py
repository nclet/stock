# streamlit_app.py
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
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

# Optional packages
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

st.set_page_config(page_title="🇺🇸 미국 증시 중단기 추세 예측 (향상판)", layout="wide")
st.title("🦅 미국 증시 추세 예측 모델 (10일 누적 수익률 예측 — 뉴스량/키워드 기반, Optuna, CatBoost 포함)")

st.markdown("""
- 뉴스: **뉴스량 + 키워드 카운트**를 사용합니다. (감성 점수 대신)
- Macro: 발표 지연(shift)을 적용합니다.
- 추가 팩터: Breadth, VIX term, Put/Call 비율(가능한 경우), Put/Call이 없으면 NaN 처리
- Feature selection: **rolling 방식**(최근 window에서 상관도/중요도로 top-k 선택)
- Hyperparam tuning: **Optuna (선택)**으로 LightGBM 튜닝 가능
- 모델: LGBM, XGB, RF, (가능하면 CatBoost) → VotingRegressor
- 해석: **SHAP + Permutation Importance 비교** (둘 다 plotly 시각화)
""")

# -----------------------
# 0. 유틸 및 설정
# -----------------------
@st.cache_data
def get_today_date():
    return datetime.now().date()

TODAY = get_today_date()

def safe_request_json(url, params=None, headers=None, timeout=15):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"HTTP 오류/JSON 파싱 오류: {e}")
        return None

# -----------------------
# 1. 데이터 로드 함수들
# -----------------------
@st.cache_data(show_spinner="⏳ FRED 데이터 로드 중...")
def get_fred_data(start_date):
    """
    FRED에서 주요 시계열(10Y, 2Y, BBB OAS, M2, SP500 EPS) 가져오기.
    start_date: date 객체
    """
    try:
        fred_api_key = st.secrets["fred"]["FRED_API_KEY"]
    except Exception:
        st.warning("FRED API 키가 설정되지 않았습니다. FRED 관련 변수는 비어있을 수 있습니다.")
        fred_api_key = None

    TICKERS = {
        "DGS10": "10Y", "DGS2": "2Y",
        "BAMLC0A4CBBB": "BBB_OAS", "M2SL": "M2", "GDPC1": "GDP",
        "SP500PE": "SP500_EPS"
    }
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    results = {}
    if not fred_api_key:
        return results

    def fetch_single(ticker):
        params = {"series_id": ticker, "api_key": fred_api_key, "file_type": "json", "observation_start": start_date.strftime("%Y-%m-%d")}
        try:
            data = safe_request_json(BASE_URL, params=params)
            if not data:
                return ticker, pd.DataFrame()
            obs = data.get('observations', [])
            df = pd.DataFrame(obs)
            if df.empty:
                return ticker, pd.DataFrame()
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            df = df[['date','value']].rename(columns={'date':'Date','value':TICKERS[ticker]}).set_index('Date')
            return ticker, df
        except Exception as e:
            return ticker, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch_single, t): t for t in TICKERS}
        for fut in futures:
            ticker, df = fut.result()
            if not df.empty:
                results[TICKERS[ticker]] = df
    # yield curve if possible
    if '10Y' in results and '2Y' in results:
        df_y = pd.merge(results['10Y'], results['2Y'], left_index=True, right_index=True, how='inner')
        results['YIELD_CURVE'] = (df_y['10Y'] - df_y['2Y']).rename('YIELD_CURVE').to_frame()
    return results

@st.cache_data(show_spinner="⏳ 시장 데이터 로드 중...")
def load_market_data(start_date, end_date):
    """
    FinanceDataReader로 S&P500, NASDAQ, VIX, WTI, GOLD, COPPER, DXY 불러오기
    """
    load_start = start_date - timedelta(days=60)
    tickers = {
        '^GSPC': 'SP500_Close', '^IXIC': 'NASDAQ_Close', '^VIX': 'VIX',
        'CL=F': 'WTI', 'GC=F': 'GOLD', 'HG=F': 'COPPER', 'DX-Y.NYB': 'DXY'
    }
    all_dfs = []
    for t, name in tickers.items():
        try:
            df = fdr.DataReader(t, start=load_start, end=end_date)
            df = df[['Close']].rename(columns={'Close': name})
            df.index = df.index.date
            all_dfs.append(df)
        except Exception as e:
            st.warning(f"{name} 데이터 로드 실패: {e}")
            continue
    if not all_dfs:
        return pd.DataFrame()
    merged = pd.concat(all_dfs, axis=1).sort_index()
    merged.index.name = 'Date'
    return merged

# Try to fetch Put/Call ratio from FRED if available (example series name might differ)
@st.cache_data(show_spinner="⏳ Put/Call 데이터 로드 시도중...")
def get_put_call_series(start_date):
    # try a known provider or FRED series; fallback to NaN series
    try:
        fred_api_key = st.secrets["fred"]["FRED_API_KEY"]
        if not fred_api_key:
            return pd.DataFrame()
        BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
        # There is no universal 'PUTCALL' series; many timelines differ. Try common series:
        cand = ["PUTCALL", "CBOEPC"]  # placeholders
        for series in cand:
            params = {"series_id": series, "api_key": fred_api_key, "file_type": "json", "observation_start": start_date.strftime("%Y-%m-%d")}
            data = safe_request_json(BASE_URL, params=params)
            if data and data.get('observations'):
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date']).dt.date
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value']).rename(columns={'date':'Date','value':series}).set_index('Date')
                return df
    except Exception:
        pass
    return pd.DataFrame()

# -----------------------
# 2. 뉴스 크롤링 및 피처
# -----------------------
def get_naver_news_api(query, display=100, start=1, sort="date"):
    """
    네이버 뉴스 검색 (한 번에 최대 100건).
    query: already a short string (e.g., '미국증시 전망 OR 금리인상')
    """
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except Exception:
        st.warning("네이버 API 키가 설정되어 있지 않습니다. 뉴스 피처는 비어있게됩니다.")
        return pd.DataFrame(columns=['Date','Title'])

    enc = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc}&display={display}&start={start}&sort={sort}"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        items = data.get('items', [])
        if not items:
            return pd.DataFrame(columns=['Date','Title'])
        news = []
        for it in items:
            title = re.sub('<[^<]+?>','', it.get('title',''))
            pub = it.get('pubDate','')
            try:
                d = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception:
                d = None
            news.append({'Date': d, 'Title': title})
        return pd.DataFrame(news)
    except Exception as e:
        st.warning(f"네이버 뉴스 API 오류: {e}")
        return pd.DataFrame(columns=['Date','Title'])

# 뉴스->피처 추출: 뉴스량 + 키워드 카운트 (악재/호재)
NEGATIVE_KEYWORDS = ["긴축","금리인상","매파","경기침체","은행부실","디폴트","인플레이션","유가급등"]
POSITIVE_KEYWORDS = ["금리인하","정책완화","유동성","경기회복","실적호조","수요증가","투자증가"]
NEG_RE = re.compile(r'\b(' + '|'.join(map(re.escape,NEGATIVE_KEYWORDS)) + r')\b', flags=re.I)
POS_RE = re.compile(r'\b(' + '|'.join(map(re.escape,POSITIVE_KEYWORDS)) + r')\b', flags=re.I)

def extract_news_features(df_news):
    """
    입력: raw news dataframe with columns ['Date','Title']
    출력: daily aggregated df with columns:
      Sentiment_Score (optional, here we don't use model sentiment), News_Count,
      Negative_Keyword_Count, Positive_Keyword_Count
    """
    if df_news is None or df_news.empty:
        return pd.DataFrame(columns=['News_Count','Negative_Keyword_Count','Positive_Keyword_Count'])
    df = df_news.copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['News_Count'] = 1
    # keyword counts on Title
    df['Negative_Keyword_Count'] = df['Title'].apply(lambda s: len(NEG_RE.findall(s)) if isinstance(s,str) else 0)
    df['Positive_Keyword_Count'] = df['Title'].apply(lambda s: len(POS_RE.findall(s)) if isinstance(s,str) else 0)
    grouped = df.groupby('Date').agg({
        'News_Count':'sum',
        'Negative_Keyword_Count':'sum',
        'Positive_Keyword_Count':'sum'
    })
    # ensure date index name
    grouped.index.name = 'Date'
    return grouped

# -----------------------
# 3. 피처 엔지니어링
# -----------------------
def compute_breadth_proxy(df_prices):
    """
    Breadth proxy: 비중(%) of days where SP500_Close > SP500_SMA20
    If we had constituents, we would compute advancing/declining.
    Returns Series indexed by Date with 'Breadth' measure [0,1]
    """
    s = df_prices['SP500_Close']
    sma20 = s.rolling(20).mean()
    breadth = (s > sma20).astype(int)
    # smooth
    breadth_ma = breadth.rolling(5).mean()
    return breadth_ma.rename('Breadth')

def create_features(df_merge, macro_shift_days=1):
    """
    df_merge: merged DataFrame indexed by Date with columns for macro and news raw features
    macro_shift_days: how many business days to shift macro variables to account release lag
    """
    df = df_merge.copy()
    # shift macro variables (to simulate release lag)
    macro_cols = [c for c in df.columns if c in ['YIELD_CURVE','10Y','2Y','M2','BBB_OAS','SP500_EPS','GDP']]
    for c in macro_cols:
        df[c] = df[c].shift(macro_shift_days)

    # Basic returns
    df['Daily_Return'] = df['SP500_Close'].pct_change() * 100
    df['Return_10D'] = df['SP500_Close'].pct_change(periods=10).shift(-10) * 100

    # News derived: ensure News_Count exists
    if 'News_Count' in df.columns:
        df['News_Count_5D'] = df['News_Count'].rolling(5).mean()
    if 'Negative_Keyword_Count' in df.columns:
        df['Neg_KW_MA_5D'] = df['Negative_Keyword_Count'].rolling(5).mean()
    if 'Positive_Keyword_Count' in df.columns:
        df['Pos_KW_MA_5D'] = df['Positive_Keyword_Count'].rolling(5).mean()

    # VIX term: difference between short-term VIX and 20-day mean (proxy)
    if 'VIX' in df.columns:
        df['VIX_20d_MA'] = df['VIX'].rolling(20).mean()
        df['VIX_Term'] = df['VIX'] - df['VIX_20d_MA']  # positive => term steepness
        df['VIX_Term_5d'] = df['VIX_Term'].rolling(5).mean()

    # Breadth proxy
    try:
        breadth = compute_breadth_proxy(df)
        df = df.join(breadth)
    except Exception:
        pass

    # Put/Call: keep as-is if exists
    # lag features
    lags = [1,3,5,10]
    lag_factors = [c for c in df.columns if c not in ['Return_10D']]
    for f in lag_factors:
        for lag in lags:
            df[f'{f}_lag_{lag}'] = df[f].shift(lag)

    # Additional technicals
    df['SP500_SMA_20'] = df['SP500_Close'].rolling(20).mean()
    df['VIX_Change_5D'] = df['VIX'].diff(5) if 'VIX' in df.columns else np.nan

    # drop rows where target NaN
    df = df.dropna()
    # select candidate features automatically
    candidate_features = [col for col in df.columns if col not in ['Return_10D']]
    # keep numeric
    candidate_features = [c for c in candidate_features if np.issubdtype(df[c].dtype, np.number)]
    return df, candidate_features

# -----------------------
# 4. Rolling Feature Selection
# -----------------------
def rolling_feature_selection(df, candidate_features, target_col='Return_10D', rolling_window=250, top_k=15):
    """
    Rolling feature selection:
      - compute correlation on the most recent rolling_window days,
      - select top_k by absolute correlation.
    Returns selected_features list.
    """
    if len(df) < max(rolling_window, 50):
        # fallback: global correlation
        corr = df[candidate_features].corrwith(df[target_col]).abs().sort_values(ascending=False)
        return corr.head(top_k).index.tolist()
    recent = df.tail(rolling_window)
    corr = recent[candidate_features].corrwith(recent[target_col]).abs().sort_values(ascending=False)
    return corr.head(top_k).index.tolist()

# -----------------------
# 5. Optuna tuning for LGBM (optional)
# -----------------------
def optuna_tune_lgb(X_train, y_train, n_trials=25, timeout=None):
    if not OPTUNA_AVAILABLE:
        st.warning("Optuna가 설치되어 있지 않습니다. 튜닝을 건너뜁니다.")
        return None
    def objective(trial):
        params = {
            'objective':'regression',
            'metric':'rmse',
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt','dart']),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 16),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        cv = TimeSeriesSplit(n_splits=3)
        rmse_scores = []
        for tr_idx, val_idx in cv.split(X_train):
            Xtr, Xv = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr, yv = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            m = lgb.LGBMRegressor(**params)
            m.fit(Xtr, ytr, eval_set=[(Xv,yv)], early_stopping_rounds=50, verbose=False)
            pred = m.predict(Xv)
            rmse_scores.append(np.sqrt(mean_squared_error(yv,pred)))
        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params

# -----------------------
# 6. Training helper
# -----------------------
@st.cache_resource
def train_voting_model(_X_train_df, _y_train, lgb_params, xgb_params, rf_params, features, use_catboost=False):
    models = []
    lgbm_model = lgb.LGBMRegressor(**lgb_params)
    models.append(('lgbm', lgbm_model))
    xgb_model = xgb.XGBRegressor(**xgb_params)
    models.append(('xgb', xgb_model))
    rf_model = RandomForestRegressor(**rf_params)
    models.append(('rf', rf_model))
    if use_catboost and CATBOOST_AVAILABLE:
        cat = CatBoostRegressor(verbose=0, random_state=42)
        models.append(('cat', cat))
    voting = VotingRegressor(estimators=models)
    voting.fit(_X_train_df[features], _y_train)
    # also train a standalone LGBM for SHAP
    lgbm_shap = lgb.LGBMRegressor(**lgb_params)
    lgbm_shap.fit(_X_train_df[features], _y_train)
    return voting, lgbm_shap

# -----------------------
# 7. Permutation importance
# -----------------------
def compute_permutation_importance(model, X, y, features, n_repeats=10):
    try:
        res = permutation_importance(model, X[features], y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
        imp = pd.Series(res.importances_mean, index=features).sort_values(ascending=False)
        return imp
    except Exception as e:
        st.warning(f"Permutation importance 실패: {e}")
        return pd.Series(dtype=float)

# -----------------------
# 8. Streamlit UI / Main flow
# -----------------------
st.markdown("---")
col1, col2, col3 = st.columns([1.5,1,1])
with col1:
    news_query = st.text_input("뉴스 키워드 (짧게, OR/| 로 연결)", value="미국증시 OR 금리인상 OR 연준", help="플레이스홀더: '미국증시 OR 금리인상'")
with col2:
    start_date = st.date_input("분석 시작일", datetime.now() - timedelta(days=365*2))
with col3:
    end_date = st.date_input("분석 종료일", datetime.now())

# tuning / options
st.sidebar.header("옵션")
use_optuna = st.sidebar.checkbox("Optuna로 LGBM 하이퍼파라미터 튜닝 (시간 소요됨)", value=False)
optuna_trials = st.sidebar.number_input("Optuna trials", min_value=10, max_value=200, value=25, step=5)
use_cat = st.sidebar.checkbox("CatBoost 포함 (설치되어 있으면)", value=False)
macro_shift_days = st.sidebar.number_input("Macro release shift days", min_value=0, max_value=5, value=1)
rolling_window = st.sidebar.number_input("Rolling window for feature selection", min_value=60, max_value=1000, value=250)
top_k_features = st.sidebar.number_input("Top-K features (rolling selection)", min_value=5, max_value=50, value=15)

if st.button("🚀 실행: 데이터 로드 → 특성 생성 → 학습 → 예측"):
    with st.spinner("데이터 로드 중..."):
        market_df = load_market_data(start_date, end_date)
        fred = get_fred_data(start_date)
        putcall = get_put_call_series(start_date)
        # assemble merge
        df_merge = market_df.copy()
        if fred:
            for name, df in fred.items():
                df_merge = pd.merge(df_merge, df, left_index=True, right_index=True, how='left')
        if not putcall.empty:
            df_merge = pd.merge(df_merge, putcall, left_index=True, right_index=True, how='left')

        # news
        # first and second batch
        # keep query short to avoid 0 results
        q = news_query
        try:
            nb1 = get_naver_news_api(q, display=100, start=1)
            nb2 = get_naver_news_api(q, display=100, start=101)
            all_news = pd.concat([nb1, nb2]).drop_duplicates(subset=['Title']).reset_index(drop=True)
        except Exception:
            all_news = pd.DataFrame(columns=['Date','Title'])
        st.info(f"네이버에서 수집된 기사 수: {len(all_news)}")

        news_features = extract_news_features(all_news) if not all_news.empty else pd.DataFrame()
        if not news_features.empty:
            df_merge = pd.merge(df_merge, news_features, left_index=True, right_index=True, how='left')

        # fillna
        df_merge = df_merge.fillna(method='ffill').fillna(0)

    with st.spinner("피처 생성 중..."):
        df_ml, candidate_features = create_features(df_merge, macro_shift_days=macro_shift_days)
        st.write(f"후보 피처 수: {len(candidate_features)}")
        if df_ml.empty or 'Return_10D' not in df_ml.columns:
            st.error("학습용 데이터가 부족합니다. 데이터 소스를 확인하세요.")
            st.stop()

    # limit to recent 500 rows for speed (same logic as before)
    df_ml = df_ml.tail(500)
    df_ml = df_ml.loc[(df_ml.index >= start_date) & (df_ml.index <= end_date)]
    if len(df_ml) <= 100:
        st.error("데이터 포인트가 부족합니다 (최소 100일 권장).")
        st.stop()

    # Rolling feature selection
    selected_features = rolling_feature_selection(df_ml, candidate_features, target_col='Return_10D', rolling_window=rolling_window, top_k=top_k_features)
    st.write(f"선택된 피처 (Top {top_k_features}): {selected_features}")

    X = df_ml[selected_features]
    y = df_ml['Return_10D']

    # cleaning
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # scale
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # train-test split
    test_size = max(30, int(0.2 * len(X_scaled)))
    X_train = X_scaled.iloc[:-test_size]
    X_test = X_scaled.iloc[-test_size:]
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    # optuna tuning (if chosen)
    if use_optuna:
        with st.spinner("Optuna 튜닝 중... (시간 소요)"):
            best = optuna_tune_lgb(X_train, y_train, n_trials=optuna_trials)
            if best:
                # map optuna params to lgb params keys (ensure required defaults)
                lgb_params = dict(best)
                lgb_params.update({'objective':'regression','metric':'rmse','random_state':42,'n_jobs':-1,'verbose':-1})
            else:
                lgb_params = {'objective':'regression','metric':'rmse','n_estimators':300,'learning_rate':0.01,'num_leaves':21,'max_depth':7,'random_state':42,'n_jobs':-1,'verbose':-1}
    else:
        lgb_params = {'objective':'regression','metric':'rmse','n_estimators':300,'learning_rate':0.01,'num_leaves':21,'max_depth':7,'random_state':42,'n_jobs':-1,'verbose':-1}

    xgb_params = {'objective':'reg:squarederror','n_estimators':500,'learning_rate':0.01,'max_depth':7,'random_state':42,'n_jobs':-1}
    rf_params = {'n_estimators':100,'max_depth':10,'random_state':42,'n_jobs':-1}

    # CV with TimeSeriesSplit (display R2 per fold)
    st.header("📊 TimeSeriesSplit 교차검증 (LGBM)")
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2s = []
    fold_idx = 1
    for tr_idx, val_idx in tscv.split(X_train):
        Xtr, Xv = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yv = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(Xtr, ytr, eval_set=[(Xv,yv)], early_stopping_rounds=50, verbose=False)
        pred = m.predict(Xv)
        r2s.append(r2_score(yv,pred))
        st.write(f"Fold {fold_idx} R2: {r2s[-1]:.4f}")
        fold_idx += 1
    st.info(f"평균 R2: {np.mean(r2s):.4f}")

    # Train final Voting model
    with st.spinner("최종 앙상블 모델 훈련 중..."):
        voting_model, lgbm_shap_model = train_voting_model(X_train, y_train, lgb_params, xgb_params, rf_params, selected_features, use_cat=use_cat and CATBOOST_AVAILABLE)

    # predictions
    y_pred_test = voting_model.predict(X_test[selected_features])
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # residual-based CI (using lgbm_shap_model residuals)
    y_train_pred = lgbm_shap_model.predict(X_train[selected_features])
    residuals = y_train - y_train_pred
    resid_std = residuals.std()
    CI_FACTOR = 1.645 * resid_std

    # next 10-day prediction (use last row)
    last_row = X_scaled.iloc[[-1]][selected_features]
    next_pred = voting_model.predict(last_row)[0]
    low_ci = next_pred - CI_FACTOR
    high_ci = next_pred + CI_FACTOR

    # Results display
    st.markdown("---")
    st.header("📈 최종 예측 결과")
    c1,c2,c3,c4 = st.columns(4)
    def fmt(x): return f"{x:+.2f}%"
    c1.metric("향후 10거래일 예측 수익률", fmt(next_pred), delta=f"90% CI: {low_ci:+.2f}% ~ {high_ci:+.2f}%")
    c2.metric("테스트 R² (앙상블)", f"{r2:.3f}", help=f"MSE: {mse:.4f}")
    current_vix = df_ml['VIX'].iloc[-1] if 'VIX' in df_ml.columns else np.nan
    c3.metric("현재 VIX (마지막값)", f"{current_vix:.2f}")
    action = "강력 매수" if (next_pred>1.0 and low_ci>0) else ("매도" if (next_pred<-1.0) else "관망")
    c4.markdown(f"### 최종 시그널: **{action}**")

    # SHAP
    st.markdown("---")
    st.header("🔍 모델 해석: SHAP + Permutation Importance 비교 (Top features)")
    try:
        explainer = shap.TreeExplainer(lgbm_shap_model)
        shap_vals = explainer.shap_values(last_row)
        shap_summary = pd.DataFrame({'feature': selected_features, 'shap_value': np.abs(shap_vals[0])})
        shap_summary = shap_summary.sort_values('shap_value', ascending=False).head(10)
        fig_shap = px.bar(shap_summary, x='shap_value', y='feature', orientation='h', title="SHAP (abs) Top features")
        st.plotly_chart(fig_shap, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP 계산 실패: {e}")

    # Permutation importance
    try:
        perm_imp = compute_permutation_importance(voting_model, X_test, y_test, selected_features, n_repeats=10)
        perm_top = perm_imp.head(10).reset_index()
        perm_top.columns = ['feature','importance']
        fig_perm = px.bar(perm_top, x='importance', y='feature', orientation='h', title='Permutation Importance (test)')
        st.plotly_chart(fig_perm, use_container_width=True)
    except Exception as e:
        st.warning(f"Permutation importance 계산 실패: {e}")

    # Compare SHAP vs Permutation (merge)
    try:
        if 'shap_summary' in locals() and not perm_imp.empty:
            comp = shap_summary.set_index('feature').join(perm_imp.rename('perm_imp'), how='inner')
            comp = comp.reset_index().rename(columns={'index':'feature','shap_value':'shap_abs'})
            comp = comp.sort_values(['perm_imp'], ascending=False).head(10)
            fig_comp = px.scatter(comp, x='perm_imp', y='shap_abs', text='feature', title='Permutation Importance vs SHAP(abs)')
            fig_comp.update_traces(textposition='middle right')
            st.plotly_chart(fig_comp, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP vs Permutation 비교 실패: {e}")

    # Prediction vs Actual chart
    st.markdown("---")
    st.header("예측 vs 실제 (테스트 구간)")
    df_plot = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test}, index=X_test.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Actual'], mode='markers+lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Predicted'], mode='lines', name='Predicted'))
    fig.update_layout(title='테스트 기간: Actual vs Predicted 10D Return', xaxis_title='Date', yaxis_title='Return(%)')
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance from LGBM model
    try:
        fi = pd.Series(lgbm_shap_model.feature_importances_, index=selected_features).sort_values(ascending=False).head(20).reset_index()
        fi.columns = ['feature','importance']
        fig_fi = px.bar(fi, x='importance', y='feature', orientation='h', title='LGBM Feature Importance (top20)')
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance 출력 실패: {e}")

    st.success("모델 실행 완료 ✅")

st.markdown("---")
st.warning("⚠️ 면책: 이 모델은 교육/연구용입니다. 실제 투자 결정 전에 충분한 검증 및 리스크 점검을 하세요.")
