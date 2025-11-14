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
import optuna # Optuna import
import crypto # crypto는 random.random을 대체하여 random.random()을 사용하여 임시 대체합니다.

# ------------------------
# ✨ 상수 및 페이지 설정
# ------------------------
st.set_page_config(page_title="🇺🇸 미국 증시 중단기 추세 예측", layout="wide")
st.title("🦅 미국 증시 추세 예측 모델 (10일 누적 수익률 예측)")

st.markdown("""
**S&P 500**의 **향후 $\mathbf{10}$거래일 누적 수익률**을 예측합니다. $\text{LGBM}$ 중요도 기반으로 피처를 선택하고, 네이버 뉴스 감성 분석을 활용합니다.
""")

# ------------------------
# 뉴스 키워드 상수 정의 (Feature 2, 3)
# ------------------------
POSITIVE_KEYWORDS = ['긍정', '상승', '호재', '기대', '강세', '돌파', '매수', '낙관', '수혜', '성장', '회복', '최고', '상향']
NEGATIVE_KEYWORDS = ['부정', '하락', '악재', '우려', '약세', '침체', '매도', '비관', '리스크', '경고', '인하', '폭락', '충격', '경색']
# 연준/금리/수급 관련 키워드 (Feature 3)
FED_ECONOMIC_KEYWORDS = ['연준', '금리', 'FOMC', '인상', '인하', '테이퍼링', '수급', '유동성', '물가', '인플레이션', '경기둔화']


# ------------------------
# 0. 매크로 데이터 수집 함수
# ------------------------
@st.cache_data(show_spinner="⏳ FRED 데이터 (금리차, M2, BBB OAS, SP500 EPS) 로드 중...")
def get_fred_data():
    try:
        # FRED API 키는 Streamlit Secrets에서 가져와야 합니다. (실제 키로 대체 필요)
        fred_api_key = st.secrets["fred"]["FRED_API_KEY"]
    except KeyError:
        st.warning("⚠️ FRED API 키가 Streamlit Secrets의 'fred' 섹션에 설정되어 있지 않습니다. 더미 키로 진행합니다.")
        fred_api_key = "DEMO_KEY" 
    
    if fred_api_key == "DEMO_KEY":
        st.warning("⚠️ FRED API 키가 없으므로 FRED 데이터 로드가 실패할 수 있습니다.")
        
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
            # .replace('.', np.nan) 대신 value가 문자열일 수 있으므로 errors='coerce' 사용
            df['value'] = pd.to_numeric(df['value'], errors='coerce') 
            df = df.dropna(subset=['value'])
            return ticker, df[['date', 'value']].rename(columns={'value': TICKERS[ticker]}).set_index('date')
        except Exception as e:
            # st.warning(f"⚠️ FRED 데이터 로드 실패 ({ticker}): {e}") 
            return ticker, pd.DataFrame()

    start_date = datetime.now().date() - timedelta(days=365 * 3)
    results = {}
    total_tickers = len(TICKERS)
    progress_bar = st.empty()
    
    # ThreadPoolExecutor를 사용하여 병렬 로딩
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_single_fred, ticker, start_date): ticker for ticker in TICKERS.keys()}
        loaded_count = 0
        for future in futures:
            ticker, df = future.result()
            if not df.empty: results[TICKERS[ticker]] = df
            loaded_count += 1
            progress_bar.progress(loaded_count / total_tickers, text=f"FRED 지표 로드 중... ({loaded_count}/{total_tickers})")
    progress_bar.empty()

    if '10Y' in results and '2Y' in results:
        df_yield = pd.merge(results['10Y'], results['2Y'], left_index=True, right_index=True, how='inner')
        # 기존 'YIELD_CURVE' 이름으로 금리차 저장
        results['YIELD_CURVE'] = (df_yield['10Y'] - df_yield['2Y']).rename('YIELD_CURVE').to_frame() 
    return results

@st.cache_data(show_spinner="⏳ Fear & Greed Index 로드 중...")
def get_fear_greed_index(limit=1095): 
    """Alternative.me에서 Fear & Greed Index를 가져옵니다."""
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        df = df.rename(columns={"value": "FGI", "timestamp": "Date"})
        return df[["Date", "FGI"]].sort_values("Date").set_index('Date')
    except Exception as e:
        st.warning(f"⚠️ Fear & Greed Index 로드 오류: {e}")
        return pd.DataFrame()

# ------------------------
# 1. 팩터 및 증시 데이터 로드
# ------------------------
@st.cache_data(show_spinner="⏳ 주가, 원자재, DXY, NASDAQ 데이터 로드 중...")
def load_market_data(start_date, end_date):
    """S&P 500, NASDAQ, VIX, WTI, Copper, Gold, DXY 데이터를 로드합니다."""
    # 시계열 피처 생성을 위해 실제 시작일보다 넉넉하게 로드
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
            # Volume 데이터는 사용하지 않으므로 Close만 선택
            df = df[['Close']].rename(columns={'Close': name}) 
            df.index = df.index.date
            all_data.append(df)
            time.sleep(0.05) # 서버 부하 방지를 위해 잠깐 대기
        except Exception as e:
            st.warning(f"⚠️ {name} ({ticker}) 데이터 로드 실패: {e}")
            continue
            
    progress_bar.empty()
    st.success("✅ 시장 데이터 로드 완료!")
    if not all_data: return pd.DataFrame()
    df_merged = pd.concat(all_data, axis=1, join='outer').sort_index()
    df_merged.index.name = 'Date'
    return df_merged

# ------------------------
# 2. 감성 분석 모델 로드 및 함수
# ------------------------
@st.cache_resource
def load_sentiment_model():
    """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
    # secrets는 재현 가능한 코드 블록에서 사용할 수 없습니다. 실제 환경에서는 st.secrets를 사용하세요.
    # hf_token = st.secrets.get("HF_TOKEN") 
    hf_token = "" # 실행 환경을 위한 플레이스홀더
    model_name = "snunlp/KR-FinBert-SC"
    try:
        # 모델 로딩 시도 (GPU 가용 시 사용)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        # st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}") # 오류 발생 시 경고는 콘솔에만 표시
        return None, None, None # 모델 로드 실패 시 None 반환
    
tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    """Calculates sentiment score for the given text."""
    if not text or not sentiment_model: return 0.0 # 모델 로드 실패 시 0.0 반환
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = sentiment_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        neg_idx, pos_idx = None, None
        
        # 라벨 인덱스 찾기
        for idx, label in sentiment_model.config.id2label.items():
            if 'negative' in label.lower() or '부정' in label: neg_idx = idx
            elif 'positive' in label.lower() or '긍정' in label: pos_idx = idx
        
        # 부정/긍정 인덱스를 찾지 못했을 경우 안전하게 처리
        negative_score = probabilities[neg_idx].item() if neg_idx is not None and neg_idx < len(probabilities) else 0
        positive_score = probabilities[pos_idx].item() if pos_idx is not None and pos_idx < len(probabilities) else 0
        
        return positive_score - negative_score
    except Exception as e:
        # st.warning(f"Sentiment analysis failed: {e}") # 분석 중 오류 발생 시 0.0 반환
        return 0.0

# 기사 내용 기반 키워드 및 비중 분석 함수
def analyze_text_keywords(title, description):
    """
    기사 제목과 내용(Description)을 기반으로
    1. 긍정/부정 단어 카운트 및 비율 (Feature 2)
    2. 연준/금리/수급 관련 키워드 비중 (Feature 3)
    을 계산합니다.
    """
    text = title + " " + description
    
    # 1. Pos/Neg Counts
    pos_count = sum(text.count(word) for word in POSITIVE_KEYWORDS)
    neg_count = sum(text.count(word) for word in NEGATIVE_KEYWORDS)
    
    # Simple ratio to prevent absolute counts from dominating. Add smoothing (+1)
    pos_neg_ratio = (pos_count + 1) / (neg_count + 1)
    
    # 2. Fed Keyword Ratio
    fed_count = sum(text.count(word) for word in FED_ECONOMIC_KEYWORDS)
    total_words = len(text.split())
    
    # Ratio of Fed keywords to total words
    fed_ratio = fed_count / total_words if total_words > 0 else 0
    
    return pos_count, neg_count, pos_neg_ratio, fed_ratio

# 네이버 API 함수 수정: Description 추가
def get_naver_news_api(query, display=100, start=1, sort="date"): 
    """
    Naver News Search API에서 데이터를 가져옵니다. Description(기사 스니펫)을 추가로 가져옵니다.
    """
    try:
        # 마찬가지로 secrets는 실제 환경에 맞게 설정해야 합니다.
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError:
        # st.error("❌ 네이버 API 키가 Streamlit Secrets의 [naver] 섹션에 설정되어 있지 않습니다.")
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
            # HTML 태그 제거
            title = re.sub('<[^<]+?>', '', item.get('title', ''))
            description = re.sub('<[^<]+?>', '', item.get('description', ''))
            pub_date = item.get('pubDate', '')
            try: pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception: pub_date_dt = None
            news_data.append({'Date': pub_date_dt, 'Title': title, 'Description': description})
        return pd.DataFrame(news_data)
    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ 네이버 API 요청 실패: {http_err} - 요청 설정(display/start)을 확인하세요.")
    except Exception as e:
        st.error(f"❌ 네이버 API 요청 실패: {e}")
        
    return pd.DataFrame(columns=['Date', 'Title', 'Description'])


# ------------------------
# 3. 피처 엔지니어링 함수
# ------------------------
def create_features(df_merge):
    """모든 팩터에 대해 시계열 피처를 생성하고 데이터를 정리합니다."""
    df = df_merge.copy()
    
    if 'NASDAQ_Close' in df.columns and 'SP500_Close' in df.columns:
        df['NASDAQ_SP500_Ratio'] = df['NASDAQ_Close'] / df['SP500_Close']
    
    # 🌟 타겟 변수를 10일 후 누적 수익률로 변경
    df['Return_10D'] = df['SP500_Close'].pct_change(periods=10).shift(-10) * 100
    df['Daily_Return'] = df['SP500_Close'].pct_change() * 100

    # ✨ 매크로 변수 정규화 방식 개선 (RAW, Pct Change, Z-score)
    MACRO_FEATURES_TO_ENHANCE = ['YIELD_CURVE', 'BBB_OAS', 'DXY', 'VIX', 'WTI', 'GOLD', 'COPPER']

    for col in MACRO_FEATURES_TO_ENHANCE:
        # 해당 컬럼이 데이터에 존재하는지 확인
        if col in df.columns: 
            # 1. Percentage Change (Pct Change)
            df[f'{col}_PCT_CHANGE'] = df[col].pct_change()
            
            # 2. Z-score Normalization (Over a 60-day window for recent deviation)
            window = 60
            std_dev = df[col].rolling(window=window).std()
            mean_val = df[col].rolling(window=window).mean()
            
            # 0으로 나누는 것을 방지
            df[f'{col}_ZSCORE_60D'] = np.where(std_dev != 0, (df[col] - mean_val) / std_dev, 0)
            
            # 3. Raw value suffix
            df.rename(columns={col: f'{col}_RAW'}, inplace=True)
    
    # 🌟 뉴스 피처에 이동 평균(MA) 및 변동성(Volatility) 추가
    news_agg_features = [
        'Sentiment_Score',      # 기존 감성 점수 (일별 평균)
        'News_Count',           # 기사 수 (Feature 4)
        'Avg_Pos_Neg_Ratio',    # 긍정/부정 비율 (Feature 2)
        'Avg_Fed_Ratio'         # 연준/금리 키워드 비중 (Feature 3)
    ]
    ma_windows = [3, 5, 10]
    
    for feature in news_agg_features:
        if feature in df.columns:
            # Feature 1: 이동평균 추가 (3일, 5일, 10일)
            for window in ma_windows:
                df[f'{feature}_MA_{window}D'] = df[feature].rolling(window=window, min_periods=1).mean()
    
    # Feature 4: 기사 변동폭 추가 (News_Count의 1일 절대 변화량)
    if 'News_Count' in df.columns:
        df['News_Count_Vol_1D'] = df['News_Count'].diff(1).abs()
        df['News_Count_Change_1D'] = df['News_Count'].diff(1) # 기사 증감량도 피처로 사용

    # --- Lagging all relevant features ---
    
    lags = [1, 3, 5, 10] 
    
    lag_factors = [
        'Daily_Return', 'FGI', 'NASDAQ_SP500_Ratio', 'SP500_EPS', 'GDP', 'M2'
    ]
    
    # 새로 생성된 매크로 팩터 버전 추가 (RAW, PCT_CHANGE, ZSCORE)
    for col in MACRO_FEATURES_TO_ENHANCE:
        if f'{col}_RAW' in df.columns:
            lag_factors.append(f'{col}_RAW')
            lag_factors.append(f'{col}_PCT_CHANGE')
            lag_factors.append(f'{col}_ZSCORE_60D') # Use 60D ZSCORE

    # 새로 생성된 모든 뉴스 관련 피처 추가 (Raw, MA, Volatility 포함)
    new_news_factors = [col for col in df.columns if col.startswith(tuple(news_agg_features)) or col.startswith('News_Count_Vol') or col.startswith('News_Count_Change')]
    lag_factors.extend(new_news_factors)
    
    # 중복 제거 및 존재하는 컬럼만 선택
    lag_factors = list(set(f for f in lag_factors if f in df.columns))
    
    for factor in lag_factors:
        for lag in lags:
            df[f'{factor}_Lag_{lag}'] = df[factor].shift(lag)
            
    # 보조 지표 추가
    if 'VIX_RAW' in df.columns:
        df['VIX_RAW_Change_5D'] = df['VIX_RAW'].diff(5)
    if 'SP500_Close' in df.columns:
        df['SP500_SMA_20'] = df['SP500_Close'].rolling(window=20).mean()
    
    # 🌟 타겟 변수가 NaN이 되는 마지막 10일을 제거, 그 외 모든 NaN 행 제거
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Target (Return_10D)이 NaN이 아닌 행만 남기고, 나머지 NaN 행 제거 (주로 Lagged features)
    df = df.dropna(subset=['Return_10D']).dropna()
    
    # 최종 피처 목록 구성
    base_features = [col for col in df.columns if not col.endswith(('Return', 'Close', '10D', '_2Y', '_10Y')) and 'SP500_' not in col and 'NASDAQ_' not in col]
    
    # 최종적으로 Lagged, Change, SMA, GDP/M2/EPS/DXY/Ratio 피처만 남깁니다.
    features = [f for f in base_features if ('Lag' in f or 'Change' in f or 'SMA' in f or f in ['GDP', 'M2', 'SP500_EPS', 'NASDAQ_SP500_Ratio'])]
    # 매크로 피처의 RAW/PCT/ZSCORE 버전 전체 포함
    features.extend([f for f in base_features if any(f.startswith(macro) and ('RAW' in f or 'PCT' in f or 'ZSCORE' in f) for macro in MACRO_FEATURES_TO_ENHANCE)])
    # 뉴스 피처 전체 포함
    features.extend([f for f in base_features if any(f.startswith(news_f) for news_f in news_agg_features) or f.startswith('News_Count_Vol') or f.startswith('News_Count_Change')])

    features = list(set(features))
    
    return df, features

# ------------------------
# 4. Optuna 목적 함수 (NEW)
# ------------------------

# Optuna 설정 상수
N_TRIALS = 20 # Streamlit 성능을 위해 트라이얼 수 제한
N_FOLDS_OPTUNA = 3 # Optuna 하이퍼파라미터 검색을 위한 TimeSeriesSplit 폴드 수

# @st.cache_data(show_spinner=False) # 캐싱은 Optuna의 동적 검색을 방해할 수 있으므로 제거
def objective_lgbm(trial, X, y):
    """Optuna가 최소화할 목적 함수: TimeSeriesSplit의 평균 RMSE"""
    tscv = TimeSeriesSplit(n_splits=N_FOLDS_OPTUNA)
    rmse_list = []

    # Optuna 탐색 공간 (과소적합 완화를 위해 규제(lambda)와 트리 깊이/잎 수 조정 포함)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000, # 조기 종료를 위해 충분히 크게 설정
        # loguniform 대신 suggest_float with log=True 사용 (Optuna 3.x 권장)
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True), # L1 규제
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True), # L2 규제
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    # TimeSeriesSplit 교차 검증
    for train_index, val_index in tscv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] # 조기 종료 적용
        )

        y_pred = model.predict(X_val_fold)
        # squared=False를 통해 RMSE 계산
        # rmse = mean_squared_error(y_val_fold, y_pred, squared=False) 
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)


# 5. 앙상블 모델 훈련 함수
@st.cache_resource(show_spinner="🚀 Soft Voting 앙상블 모델 훈련 중/로드 중...")
def train_voting_model(_X_train_df, _y_train, _lgbm_params, _xgb_params, _rf_params):
    """
    앙상블 모델(LGBM, XGBoost, RF)을 훈련하고, SHAP/Importance를 위한
    별도의 LGBM 모델 인스턴스(최종 훈련 세트로 학습)를 반환합니다.
    """
    # LGBM은 Optuna 최적화 파라미터를 사용합니다.
    lgbm_model = lgb.LGBMRegressor(**_lgbm_params)
    xgb_model = xgb.XGBRegressor(**_xgb_params)
    rf_model = RandomForestRegressor(**_rf_params)
    
    # 앙상블 모델 설정 (Soft Voting)
    voting_model = VotingRegressor(
        estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('rf', rf_model)],
        weights=[1, 1, 1] # 균등 가중치
    )
    
    # Voting Regressor 훈련
    voting_model.fit(_X_train_df, _y_train) 
    
    # VotingRegressor 내부의 LGBM 모델을 추출하여 SHAP/Importance에 사용
    # fit 이후 estimators_ 리스트에서 접근 가능
    final_lgbm_model = voting_model.estimators_[0] 
    
    return voting_model, final_lgbm_model

st.markdown("---")
# UI 입력 요소
col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    news_query = st.text_input(
        "📰 뉴스 감성 분석 키워드", 
        value="미국 증시 전망|금리 인상|연준|경기 침체", 
        help="네이버 뉴스 검색에 사용될 키워드를 '|' (파이프 기호)로 구분하여 입력하세요. (예: S&P 500|경기 침체)"
    )
with col2:
    start_date = st.date_input("분석 시작일", datetime.now() - timedelta(days=365 * 2)) 
with col3:
    end_date = st.date_input("분석 종료일", datetime.now().date())
    
if st.button("🚀 데이터 로드, 분석 및 예측 시작 (10일 추세 예측)", type="primary", use_container_width=True):
    
    # 1. 데이터 로드
    market_df = load_market_data(start_date, end_date)
    fred_data = get_fred_data()
    fg_df = get_fear_greed_index(limit=365 * 3)
    
    if market_df.empty or 'SP500_Close' not in market_df.columns:
        st.error("❌ S&P 500 데이터 로드에 실패했습니다. 유효한 기간을 선택하거나 데이터 로드 함수를 확인해주세요.")
        st.stop()
        
    # 1-2. 뉴스 감성 및 키워드 분석
    if not sentiment_model:
        st.warning("⚠️ 감성 분석 모델 로드에 실패하여 뉴스 감성 피처는 0으로 채워집니다. 진행합니다.")
        
    with st.spinner(f"뉴스 크롤링 및 감성/키워드 분석 중... (키워드: {news_query})"):
        # Description을 포함하여 뉴스 데이터 크롤링
        news_batch_1 = get_naver_news_api(news_query, display=100, start=1) 
        news_batch_2 = get_naver_news_api(news_query, display=100, start=101)
        
        all_news = pd.concat([news_batch_1, news_batch_2]).drop_duplicates(subset=['Title']).reset_index(drop=True)
        
        if all_news.empty or 'Date' not in all_news.columns or all_news['Date'].isnull().all():
            st.warning("⚠️ 네이버 API로부터 유효한 기사 데이터를 수집하지 못했습니다. 뉴스 분석을 건너뜁니다.")
            news_grouped = pd.DataFrame()
        else:
            load_start_date = start_date - timedelta(days=50)
            filtered_news = all_news[(all_news['Date'] >= load_start_date) & (all_news['Date'] <= end_date)].copy()
            
            if not filtered_news.empty:
                # 기사에 대한 감성 점수 계산
                filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
                
                # 기사에 대한 키워드 비율 계산
                filtered_news[['Pos_Count', 'Neg_Count', 'Pos_Neg_Ratio', 'Fed_Ratio']] = filtered_news.apply(
                    lambda row: analyze_text_keywords(row['Title'], row['Description']), 
                    axis=1, result_type='expand'
                )
                
                # 일자별로 집계
                news_grouped = filtered_news.groupby('Date').agg(
                    Sentiment_Score=('Sentiment_Score', 'mean'), 
                    News_Count=('Title', 'count'), 
                    Avg_Pos_Neg_Ratio=('Pos_Neg_Ratio', 'mean'), 
                    Avg_Fed_Ratio=('Fed_Ratio', 'mean') 
                )
                st.success(f"✅ 뉴스 감성 및 키워드 분석 완료! (총 {len(filtered_news)}개 기사 분석)")
            else:
                st.warning("⚠️ 지정된 기간에 해당하는 기사가 없습니다. 뉴스 분석을 건너뜁니다.")
                news_grouped = pd.DataFrame()

    # 2. 데이터 병합
    df_merge = market_df
    if not fg_df.empty: df_merge = pd.merge(df_merge, fg_df, left_index=True, right_index=True, how='left')
    for name, df_fred in fred_data.items(): 
        # FRED 데이터는 인덱스 이름이 다를 수 있으므로, 이름을 통일하고 병합
        if not df_fred.empty:
            df_merge = pd.merge(df_merge, df_fred, left_index=True, right_index=True, how='left')
            
    if not news_grouped.empty:
        df_merge = pd.merge(df_merge, news_grouped, left_index=True, right_index=True, how='left')
    
    # 결측치 처리 (최근 데이터는 ffill, 나머지는 0)
    df_merge = df_merge.fillna(method='ffill').fillna(0)
    
    # 3. 피처 엔지니어링 및 데이터 준비
    df_ml, features_full = create_features(df_merge)
    
    # 분석 기간 필터링 및 데이터 부족 체크
    df_ml = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)]

    if len(df_ml) <= 100:
        st.error(f"데이터가 부족합니다. 분석 기간을 늘리세요. (현재 {len(df_ml)}일, 최소 100일 필요)")
        st.stop()
        
    X_full = df_ml[features_full]
    y = df_ml['Return_10D'] 
    
    # 4. 피처 선택: LightGBM 중요도 기반 (Optuna 전 1회 실행)
    st.subheader("⚙️ 피처 선택 (LightGBM 중요도 기반 Top 15)")
    
    # 임시 모델 파라미터 (Optuna 이전 사용)
    INITIAL_LGBM_PARAMS = {'objective': 'regression', 'metric': 'rmse', 'n_estimators': 300, 'learning_rate': 0.01, 'num_leaves': 21, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    # 최종 피처 선택을 위해 전체 데이터셋으로 임시 LGBM 훈련
    temp_model = lgb.LGBMRegressor(**INITIAL_LGBM_PARAMS) 
    temp_model.fit(X_full, y)

    feature_importances = pd.Series(temp_model.feature_importances_, index=X_full.columns)
    features = feature_importances.nlargest(15).index.tolist()
    
    st.info(f"선택된 피처 수: {len(features)}개. (전체 {len(features_full)}개 중 상위 15개, LGBM 기반)")
    
    # 선택된 피처 목록을 UI에 표시
    news_features = [f for f in features if 'Sentiment' in f or 'News_Count' in f or 'Fed_Ratio' in f or 'Avg_Pos_Neg_Ratio' in f]
    macro_features = [f for f in features if any(f.startswith(macro) for macro in ['YIELD_CURVE', 'BBB_OAS', 'DXY', 'VIX', 'WTI', 'GOLD', 'COPPER'])]
    other_features = [f for f in features if f not in news_features and f not in macro_features]
    
    st.markdown(f"**선택된 뉴스 피처:** `{'`, `'.join(news_features)}`")
    st.markdown(f"**선택된 매크로 피처 (RAW/PCT/ZSCORE 포함):** `{'`, `'.join(macro_features)}`")
    st.markdown(f"**기타 피처:** `{'`, `'.join(other_features)}`")
    
    X = df_ml[features] 
    
    # 5. Optuna 하이퍼파라미터 최적화 (LGBM)
    st.header("✨ Optuna 하이퍼파라미터 최적화 (LGBM)")
    with st.spinner(f"⏳ Optuna ({N_TRIALS} 트라이얼)를 사용하여 LGBM 최적화 중 (TSCV {N_FOLDS_OPTUNA} 폴드)..."):
        # X와 y는 선택된 피처만 포함합니다.
        study = optuna.create_study(direction='minimize')
        # objective_lgbm 함수를 직접 호출하지 않고, lambda를 통해 Optuna에 전달
        study.optimize(lambda trial: objective_lgbm(trial, X, y), n_trials=N_TRIALS)

        best_lgbm_params = study.best_params
        
        # Optuna 결과를 최종 LGBM 파라미터로 설정
        LGBM_PARAMS = {
            'objective': 'regression', 'metric': 'rmse', 'n_estimators': 500, # 최종 앙상블 모델에 사용될 n_estimators
            'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'boosting_type': 'gbdt',
            **best_lgbm_params 
        }
        
        st.success(f"✅ Optuna 최적화 완료! Best RMSE: {study.best_value:.4f}")
        st.json(LGBM_PARAMS)

    # 6. 데이터 분할 및 스케일링
    scaler = MinMaxScaler()
    X_scaled_all = scaler.fit_transform(X)
    X_scaled_all_df = pd.DataFrame(X_scaled_all, columns=X.columns, index=X.index)
    
    # 테스트 데이터셋 분리 (마지막 N일)
    test_size = max(30, int(0.2 * len(X_scaled_all_df)))
    
    X_train_df, X_test_df = X_scaled_all_df.iloc[:-test_size], X_scaled_all_df.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    test_start_date = X_test_df.index[0]

    st.markdown(f"**훈련 기간:** {X_train_df.index[0]} ~ {X_train_df.index[-1]}")
    st.markdown(f"**테스트 기간:** {test_start_date} ~ {X_test_df.index[-1]} (총 {test_size}일)")
    
    # 7. 시계열 교차검증 (TS Split) - 최종 LGBM 파라미터로 검증
    st.header("📊 시계열 교차검증 (TimeSeriesSplit)")
    
    n_splits = 3 # Optuna에서 3을 사용했으므로 통일
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    r2_scores_lgbm = []
    
    # XGBoost 및 RandomForestRegressor 파라미터 (Optuna 최적화 없이 고정)
    XGB_PARAMS = {'objective': 'reg:squarederror', 'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1}
    RF_PARAMS = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}

    with st.spinner(f"⏳ TimeSeriesSplit 교차검증 중 (폴드 {n_splits}개, Optuna 최적 파라미터 사용, Early Stopping=50 적용)..."):
        
        for i, (train_index, val_index) in enumerate(tscv.split(X_train_df)):
            X_train_fold, X_val_fold = X_train_df.iloc[train_index], X_train_df.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            lgbm_fold = lgb.LGBMRegressor(**LGBM_PARAMS)
            
            # 과소적합 방지: Early Stopping Rounds를 50으로 설정하여 최종 모델 성능 향상
            lgbm_fold.fit(X_train_fold, y_train_fold,
                          eval_set=[(X_val_fold, y_val_fold)],
                          eval_metric='rmse',
                          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            
            y_val_pred = lgbm_fold.predict(X_val_fold)
            r2_scores_lgbm.append(r2_score(y_val_fold, y_val_pred))
            
        avg_r2 = np.mean(r2_scores_lgbm)
        st.info(f"✅ TimeSeriesSplit 평균 R² (LGBM 기준): **{avg_r2:.4f}**")
        st.dataframe(pd.DataFrame({'Fold': range(1, n_splits + 1), 'R2 Score': r2_scores_lgbm}), use_container_width=True)
    st.markdown("---")

    # 8. 최종 앙상블 모델 훈련
    voting_model, final_lgbm_model = train_voting_model(
        X_train_df, 
        y_train, 
        LGBM_PARAMS, 
        XGB_PARAMS, 
        RF_PARAMS
    )
        
    # 훈련 잔차 계산 (CI용)
    y_train_pred_lgbm = final_lgbm_model.predict(X_train_df)
    residuals = y_train - y_train_pred_lgbm
    residual_std = residuals.std()
    CI_FACTOR = 1.645 * residual_std # 90% 신뢰구간 (Z=1.645)
    
    # 테스트 기간 예측 (X_test_df 사용)
    y_test_pred = voting_model.predict(X_test_df)

    # 다음 10일 예측 및 CI 계산
    last_data_scaled = X_scaled_all_df.iloc[-1].values.reshape(1, -1)
    last_data_df = pd.DataFrame(last_data_scaled, columns=X_scaled_all_df.columns)
    
    next_day_return_pred = voting_model.predict(last_data_df)[0]
    low_ci = next_day_return_pred - CI_FACTOR
    high_ci = next_day_return_pred + CI_FACTOR
    
    # 9. 결과 출력
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
        current_vix = df_ml['VIX_RAW'].iloc[-1] if 'VIX_RAW' in df_ml.columns else df_ml['VIX'].iloc[-1]
        vix_trend = "하락 (강세) 🟢" if df_ml['VIX_RAW_Change_5D'].iloc[-1] < 0 else "상승 (약세) 🔴"
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

    # 10. SHAP 해석 추가
    st.header("💡 예측 해석: SHAP (10일 추세 예측에 기여)")
    st.markdown(f"**SHAP**을 사용하여 모델이 최종 $\mathbf{{10}}$일 예측(`{next_day_return_pred:+.2f}%`)을 산출하는 데 기여한 팩터의 영향력을 분석합니다. (**VotingRegressor 내부의 LightGBM 모델 기준**)")
    
    try:
        # SHAP Explainer는 훈련 데이터의 일부를 사용하여 학습해야 하지만, Streamlit 환경에서는 마지막 데이터를 사용해 현재 예측 분석
        explainer = shap.TreeExplainer(final_lgbm_model) 
        shap_values = explainer.shap_values(last_data_df)
        
        shap_df = pd.DataFrame({
            'Feature': last_data_df.columns,
            'SHAP Value': shap_values[0],
            'Scaled Feature Value': last_data_df.iloc[0].values
        })
        
        shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
        shap_df = shap_df.sort_values('Abs SHAP', ascending=False).head(5)

        fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                              color='SHAP Value', color_continuous_scale=px.colors.diverging.RdBu,
                              title=f"향후 10일 예측({next_day_return_pred:+.2f}%)에 기여한 Top 5 팩터",
                              hover_data={'Scaled Feature Value': True, 'SHAP Value': ':.4f'})
        fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ SHAP 해석 로드 중 오류 발생: {e}. SHAP Explainer가 예상치 못한 입력에 실패했습니다. (마지막 데이터: {last_data_df.head(1).to_dict('records')})")
    st.markdown("---")


    # 11. 피처 상관관계 히트맵 시각화
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
                                 title=f'LGBM 선택 {N_TOP_FEATURES}개 피처 간의 상관관계 히트맵',
                                 text_auto=".2f") # 상관계수 값을 자동으로 표시
        fig_heatmap.update_xaxes(side="top")
        
        # 텍스트 색상 대비 조정
        fig_heatmap.update_traces(textfont_color="black")
        
        fig_heatmap.update_layout(height=800)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ 히트맵 생성 중 오류: {e}")
        
    st.markdown("---")


    # 12. 주요 매크로 팩터 추이 시각화 (기존 코드의 누락된 부분)
    st.header("📊 주요 매크로 팩터 추이 (S&P 500과 비교)")
    
    # 분석 기간에 맞게 데이터 필터링
    df_macro_plot = df_ml[df_ml.index >= start_date].copy()

    fig_macro = go.Figure()
    
    # Y1: S&P 500
    fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['SP500_Close'], name='S&P 500 (좌측 축)', line=dict(color='#1f77b4', width=2), yaxis='y1'))
    
    # Y2: 장단기 금리차 (YIELD_CURVE_RAW)
    if 'YIELD_CURVE_RAW' in df_macro_plot.columns:
        fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['YIELD_CURVE_RAW'], name='장단기 금리차 (우측 축)', line=dict(color='#ff7f0e', dash='dot'), yaxis='y2', opacity=0.8))
    
    # Y2: VIX (VIX_RAW)
    if 'VIX_RAW' in df_macro_plot.columns:
        # VIX는 반전 지표이므로 색상 대비를 주고 Y2 축을 공유
        fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['VIX_RAW'], name='VIX (우측 축)', line=dict(color='#2ca02c', dash='dash'), yaxis='y2', opacity=0.7))
    
    # Update layout for dual Y-axis
    fig_macro.update_layout(
        title_text='S&P 500 추이 및 주요 매크로 지표 비교',
        xaxis=dict(title="날짜"),
        yaxis=dict(title="S&P 500 종가", showgrid=True, zeroline=False),
        yaxis2=dict(
            title="매크로 지표 값 (금리차/VIX)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(x=0, y=1.1, orientation='h'),
        hovermode="x unified",
        height=600
    )
    st.plotly_chart(fig_macro, use_container_width=True)
    st.markdown("---")


    # 13. 예측 결과 시각화 (테스트 셋) - 추가된 부분
    st.header("✨ 예측 결과 시각화 (테스트 셋)")
    st.markdown(f"테스트 기간({test_start_date} ~ {X_test_df.index[-1]}) 동안의 실제 10일 수익률과 모델 예측 수익률을 비교합니다.")

    # 예측 결과 데이터프레임
    df_results = pd.DataFrame({
        'Date': y_test.index,
        'Actual 10D Return': y_test.values,
        'Predicted 10D Return': y_test_pred
    }).set_index('Date')

    # 최종 예측 값을 위한 미래 날짜 설정
    # 10거래일 후의 대략적인 날짜 (주말 및 휴일은 고려하지 않음)
    prediction_date = end_date + timedelta(days=15) 
    
    df_future_pred = pd.DataFrame({
        'Date': [prediction_date],
        'Predicted 10D Return': [next_day_return_pred],
        'Actual 10D Return': [np.nan] # 실제 값은 알 수 없음
    }).set_index('Date')
    
    # 신뢰 구간 데이터프레임
    df_ci = pd.DataFrame({
        'Date': [prediction_date],
        'Low CI': [low_ci],
        'High CI': [high_ci]
    }).set_index('Date')
    
    # 테스트 결과 + 미래 예측 데이터 합치기
    df_combined = pd.concat([df_results, df_future_pred])

    fig_test = go.Figure()
    
    # Actual
    fig_test.add_trace(go.Scatter(x=df_combined.index, y=df_combined['Actual 10D Return'], name='실제 10일 수익률', mode='lines+markers', line=dict(color='red', width=2), marker=dict(size=4)))
    
    # Predicted (Test Set)
    fig_test.add_trace(go.Scatter(x=df_results.index, y=df_results['Predicted 10D Return'], name='예측 10일 수익률 (테스트)', mode='lines+markers', line=dict(color='blue', width=2, dash='dot'), marker=dict(size=4)))
    
    # Predicted (Next Day)
    fig_test.add_trace(go.Scatter(x=df_future_pred.index, y=df_future_pred['Predicted 10D Return'], name=f'최종 예측값 ({format_pred_value(next_day_return_pred)})', mode='markers', 
                                  marker=dict(size=12, color='darkblue', symbol='star')))

    # Confidence Interval Area (for future prediction point only)
    fig_test.add_trace(go.Scatter(
        x=[prediction_date, prediction_date],
        y=[low_ci, high_ci],
        mode='lines',
        name='90% 신뢰구간',
        line=dict(color='rgba(0,0,255,0.4)', width=8),
        hoverinfo='skip'
    ))
    
    fig_test.update_layout(
        title='테스트 기간 예측 vs 실제 및 최종 예측값',
        xaxis_title="날짜",
        yaxis_title="10거래일 누적 수익률 (%)",
        legend=dict(x=0, y=1.05, orientation='h'),
        hovermode="x unified",
        height=600,
        shapes=[
            # 제로 라인
            dict(
                type='line',
                x0=df_combined.index.min(), 
                x1=df_combined.index.max(), 
                y0=0,
                y1=0,
                line=dict(color='gray', dash='dash', width=1)
            )
        ]
    )
    st.plotly_chart(fig_test, use_container_width=True)

    st.markdown("---")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# from datetime import datetime, timedelta
# import plotly.graph_objects as go
# import plotly.express as px
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import TimeSeriesSplit
# import urllib.parse
# from json.decoder import JSONDecodeError
# import FinanceDataReader as fdr
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor, VotingRegressor
# from sklearn.preprocessing import MinMaxScaler
# import time
# from concurrent.futures import ThreadPoolExecutor
# import re
# import shap
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ------------------------
# # ✨ 상수 및 페이지 설정
# # ------------------------
# st.set_page_config(page_title="🇺🇸 미국 증시 중단기 추세 예측", layout="wide")
# st.title("🦅 미국 증시 추세 예측 모델 (10일 누적 수익률 예측)")

# st.markdown("""
# **S&P 500**의 **향후 $\mathbf{10}$거래일 누적 수익률**을 예측합니다. $\text{LGBM}$ 중요도 기반으로 피처를 선택하고, 네이버 뉴스 감성 분석을 활용합니다.
# """)

# # NEW: 뉴스 키워드 상수 정의 (Feature 2, 3)
# # ------------------------
# POSITIVE_KEYWORDS = ['긍정', '상승', '호재', '기대', '강세', '돌파', '매수', '낙관', '수혜', '성장', '회복', '최고', '상향']
# NEGATIVE_KEYWORDS = ['부정', '하락', '악재', '우려', '약세', '침체', '매도', '비관', '리스크', '경고', '인하', '폭락', '충격', '경색']
# # 연준/금리/수급 관련 키워드 (Feature 3)
# FED_ECONOMIC_KEYWORDS = ['연준', '금리', 'FOMC', '인상', '인하', '테이퍼링', '수급', '유동성', '물가', '인플레이션', '경기둔화']


# # 0. 매크로 데이터 수집 함수 (변경 없음)
# # ------------------------
# @st.cache_data(show_spinner="⏳ FRED 데이터 (금리차, M2, BBB OAS, SP500 EPS) 로드 중...")
# def get_fred_data():
#     try:
#         fred_api_key = st.secrets["fred"]["FRED_API_KEY"]
#     except KeyError:
#         st.error("❌ FRED API 키 설정 오류: Streamlit Secrets의 'fred' 섹션과 'FRED_API_KEY' 이름을 확인해주세요.")
#         st.stop()
#         return {}
#     TICKERS = {
#         "DGS10": "10Y", "DGS2": "2Y", 
#         "BAMLC0A4CBBB": "BBB_OAS", "M2SL": "M2", "GDPC1": "GDP",
#         "SP500PE": "SP500_EPS"
#     }
#     BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
#     def fetch_single_fred(ticker, observation_start):
#         params = {
#             "series_id": ticker, "api_key": fred_api_key, "file_type": "json", 
#             "observation_start": observation_start.strftime("%Y-%m-%d")
#         }
#         try:
#             response = requests.get(BASE_URL, params=params)
#             response.raise_for_status()
#             data = response.json().get('observations', [])
#             df = pd.DataFrame(data)
#             df['date'] = pd.to_datetime(df['date']).dt.date
#             df['value'] = pd.to_numeric(df['value'], errors='coerce')
#             df = df.dropna(subset=['value'])
#             return ticker, df[['date', 'value']].rename(columns={'value': TICKERS[ticker]}).set_index('date')
#         except Exception as e:
#             st.warning(f"⚠️ FRED 데이터 로드 실패 ({ticker}): {e}")
#             return ticker, pd.DataFrame()

#     start_date = datetime.now().date() - timedelta(days=365 * 3)
#     results = {}
#     total_tickers = len(TICKERS)
#     progress_bar = st.empty()
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = {executor.submit(fetch_single_fred, ticker, start_date): ticker for ticker in TICKERS.keys()}
#         loaded_count = 0
#         for future in futures:
#             ticker, df = future.result()
#             if not df.empty: results[TICKERS[ticker]] = df
#             loaded_count += 1
#             progress_bar.progress(loaded_count / total_tickers, text=f"FRED 지표 로드 중... ({loaded_count}/{total_tickers})")
#     progress_bar.empty()

#     if '10Y' in results and '2Y' in results:
#         df_yield = pd.merge(results['10Y'], results['2Y'], left_index=True, right_index=True, how='inner')
#         # 기존 'YIELD_CURVE' 이름으로 금리차 저장
#         results['YIELD_CURVE'] = (df_yield['10Y'] - df_yield['2Y']).rename('YIELD_CURVE').to_frame() 
#     return results

# @st.cache_data(show_spinner="⏳ Fear & Greed Index 로드 중...")
# def get_fear_greed_index(limit=1095): 
#     """Alternative.me에서 Fear & Greed Index를 가져옵니다."""
#     url = f"https://api.alternative.me/fng/?limit={limit}"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json().get("data", [])
#         df = pd.DataFrame(data)
#         df["value"] = df["value"].astype(float)
#         df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
#         df = df.rename(columns={"value": "FGI", "timestamp": "Date"})
#         return df[["Date", "FGI"]].sort_values("Date").set_index('Date')
#     except Exception as e:
#         st.warning(f"⚠️ Fear & Greed Index 로드 오류: {e}")
#         return pd.DataFrame()

# # ------------------------
# # 1. 팩터 및 증시 데이터 로드 (변경 없음)
# # ------------------------
# @st.cache_data(show_spinner="⏳ 주가, 원자재, DXY, NASDAQ 데이터 로드 중...")
# def load_market_data(start_date, end_date):
#     """S&P 500, NASDAQ, VIX, WTI, Copper, Gold, DXY 데이터를 로드합니다."""
#     load_start_date = start_date - timedelta(days=50) 
#     tickers = {
#         '^GSPC': 'SP500_Close', '^IXIC': 'NASDAQ_Close', '^VIX': 'VIX', 
#         'CL=F': 'WTI', 'GC=F': 'GOLD', 'HG=F': 'COPPER', 'DX-Y.NYB': 'DXY'
#     }
#     all_data = []
#     total_tickers = len(tickers)
#     progress_bar = st.progress(0, text="시장 데이터 로드 중...")
#     for i, (ticker, name) in enumerate(tickers.items()):
#         try:
#             progress_bar.progress((i + 1) / total_tickers, text=f"{name} ({ticker}) 로드 중...")
#             df = fdr.DataReader(ticker, start=load_start_date, end=end_date)
#             df = df[['Close']].rename(columns={'Close': name})
#             df.index = df.index.date
#             all_data.append(df)
#             time.sleep(0.05)
#         except Exception as e:
#             st.warning(f"⚠️ {name} ({ticker}) 데이터 로드 실패: {e}")
#             continue
#     progress_bar.empty()
#     st.success("✅ 시장 데이터 로드 완료!")
#     if not all_data: return pd.DataFrame()
#     df_merged = pd.concat(all_data, axis=1, join='outer').sort_index()
#     df_merged.index.name = 'Date'
#     return df_merged

# # ------------------------
# # 2. 감성 분석 모델 로드 및 함수 (변경 없음)
# # ------------------------
# @st.cache_resource
# def load_sentiment_model():
#     """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
#     # Note: st.secrets.get("HF_TOKEN") is assumed to be defined externally in the environment
#     # hf_token = st.secrets.get("HF_TOKEN") # Commented out as secrets are not provided in this context
#     hf_token = "" # Placeholder for execution environment
#     model_name = "snunlp/KR-FinBert-SC"
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
#         model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         return tokenizer, model, device
#     except Exception as e:
#         # st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
#         # st.info("Hugging Face 토큰 설정 또는 라이브러리 버전을 확인해주세요.")
#         return None, None, None # Continue execution with sentiment features as 0
    
# tokenizer, sentiment_model, device = load_sentiment_model()

# def analyze_sentiment(text):
#     """Calculates sentiment score for the given text."""
#     if not text or not sentiment_model: return 0.0 # Return 0.0 if model fails to load
#     try:
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad(): outputs = sentiment_model(**inputs)
#         probabilities = torch.softmax(outputs.logits, dim=1)[0]
#         neg_idx, pos_idx = None, None
#         for idx, label in sentiment_model.config.id2label.items():
#             if 'negative' in label.lower() or '부정' in label: neg_idx = idx
#             elif 'positive' in label.lower() or '긍정' in label: pos_idx = idx
#         negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
#         positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0
#         return positive_score - negative_score
#     except Exception as e:
#         # st.warning(f"Sentiment analysis failed: {e}")
#         return 0.0

# # NEW: 기사 내용 기반 키워드 및 비중 분석 함수 (Feature 2, 3)
# # ------------------------
# def analyze_text_keywords(title, description):
#     """
#     기사 제목과 내용(Description)을 기반으로
#     1. 긍정/부정 단어 카운트 및 비율 (Feature 2)
#     2. 연준/금리/수급 관련 키워드 비중 (Feature 3)
#     을 계산합니다.
#     """
#     text = title + " " + description
    
#     # 1. Pos/Neg Counts
#     pos_count = sum(text.count(word) for word in POSITIVE_KEYWORDS)
#     neg_count = sum(text.count(word) for word in NEGATIVE_KEYWORDS)
    
#     # Simple ratio to prevent absolute counts from dominating. Add smoothing (+1)
#     pos_neg_ratio = (pos_count + 1) / (neg_count + 1)
    
#     # 2. Fed Keyword Ratio
#     fed_count = sum(text.count(word) for word in FED_ECONOMIC_KEYWORDS)
#     total_words = len(text.split())
    
#     # Ratio of Fed keywords to total words
#     fed_ratio = fed_count / total_words if total_words > 0 else 0
    
#     return pos_count, neg_count, pos_neg_ratio, fed_ratio

# # 네이버 API 함수 수정: Description 추가 (Feat 2, 3 분석을 위해)
# # ------------------------
# def get_naver_news_api(query, display=100, start=1, sort="date"): 
#     try:
#         # Note: st.secrets.get("naver") is assumed to be defined externally in the environment
#         client_id = st.secrets["naver"]["client_id"]
#         client_secret = st.secrets["naver"]["client_secret"]
#     except KeyError:
#         # st.error("❌ 네이버 API 키가 Streamlit Secrets의 [naver] 섹션에 설정되어 있지 않습니다.")
#         return pd.DataFrame(columns=['Date', 'Title', 'Description']) 

#     enc_query = urllib.parse.quote(query)
#     url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"
#     headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status() 
#         data = response.json()
#         items = data.get('items', [])
#         news_data = []
#         for item in items:
#             title = re.sub('<[^<]+?>', '', item.get('title', ''))
#             description = re.sub('<[^<]+?>', '', item.get('description', '')) # Description 추가
#             pub_date = item.get('pubDate', '')
#             try: pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
#             except Exception: pub_date_dt = None
#             news_data.append({'Date': pub_date_dt, 'Title': title, 'Description': description})
#         return pd.DataFrame(news_data)
#     except requests.exceptions.HTTPError as http_err:
#         st.error(f"❌ 네이버 API 요청 실패: {http_err} - 요청 설정(display/start)을 확인하세요.")
#     except Exception as e:
#         st.error(f"❌ 네이버 API 요청 실패: {e}")
        
#     return pd.DataFrame(columns=['Date', 'Title', 'Description'])

# # 3. 피처 엔지니어링 함수 (개선된 뉴스 피처 로직 포함)
# # ------------------------
# def create_features(df_merge):
#     """모든 팩터에 대해 시계열 피처를 생성하고 데이터를 정리합니다."""
#     df = df_merge.copy()
    
#     if 'NASDAQ_Close' in df.columns and 'SP500_Close' in df.columns:
#         df['NASDAQ_SP500_Ratio'] = df['NASDAQ_Close'] / df['SP500_Close']
    
#     # 🌟 타겟 변수를 10일 후 누적 수익률로 변경
#     df['Return_10D'] = df['SP500_Close'].pct_change(periods=10).shift(-10) * 100
#     df['Daily_Return'] = df['SP500_Close'].pct_change() * 100

#     # ✨ NEW: 매크로 변수 정규화 방식 개선 (RAW, Pct Change, Z-score)
#     # -----------------------------------------------
#     # 원본 값을 유지하고 다양한 정규화 방식을 적용할 매크로/원자재/지수 목록
#     MACRO_FEATURES_TO_ENHANCE = ['YIELD_CURVE', 'BBB_OAS', 'DXY', 'VIX', 'WTI', 'GOLD', 'COPPER']

#     for col in MACRO_FEATURES_TO_ENHANCE:
#         if col in df.columns:
#             # 1. Percentage Change (Pct Change)
#             df[f'{col}_PCT_CHANGE'] = df[col].pct_change()
            
#             # 2. Z-score Normalization
#             # 시장 레벨 변화를 볼 때 적합한 Z-score: (값 - 평균) / 표준편차
#             if df[col].std() != 0:
#                 df[f'{col}_ZSCORE'] = (df[col] - df[col].mean()) / df[col].std()
#             else:
#                 df[f'{col}_ZSCORE'] = 0
            
#             # 3. Raw value suffix (원본 값도 피처로 활용)
#             df.rename(columns={col: f'{col}_RAW'}, inplace=True)
    
#     # 🌟 NEW: 뉴스 피처에 이동 평균(MA) 및 변동성(Volatility) 추가 (Feature 1, 4)
#     # -----------------------------------------------
#     # 일별 집계된 뉴스 피처 목록
#     news_agg_features = [
#         'Sentiment_Score',      # 기존 감성 점수 (일별 평균)
#         'News_Count',           # 기사 수 (Feature 4)
#         'Avg_Pos_Neg_Ratio',    # 긍정/부정 비율 (Feature 2)
#         'Avg_Fed_Ratio'         # 연준/금리 키워드 비중 (Feature 3)
#     ]
#     ma_windows = [3, 5, 10]
    
#     for feature in news_agg_features:
#         if feature in df.columns:
#             # Feature 1: 이동평균 추가 (3일, 5일, 10일)
#             for window in ma_windows:
#                 df[f'{feature}_MA_{window}D'] = df[feature].rolling(window=window, min_periods=1).mean()
    
#     # Feature 4: 기사 변동폭 추가 (News_Count의 1일 절대 변화량)
#     if 'News_Count' in df.columns:
#         df['News_Count_Vol_1D'] = df['News_Count'].diff(1).abs()
#         df['News_Count_Change_1D'] = df['News_Count'].diff(1) # 기사 증감량도 피처로 사용

#     # --- Lagging all relevant features ---
    
#     lags = [1, 3, 5, 10] 
    
#     # 기존 시장 및 경제 팩터 (업데이트된 이름)
#     # GDP, M2, SP500_EPS, FGI, NASDAQ_SP500_Ratio는 변경 없음
#     lag_factors = [
#         'Daily_Return', 'FGI', 'NASDAQ_SP500_Ratio', 'SP500_EPS', 'GDP', 'M2'
#     ]
    
#     # 새로 생성된 매크로 팩터 버전 추가 (RAW, PCT_CHANGE, ZSCORE)
#     for col in MACRO_FEATURES_TO_ENHANCE:
#         if f'{col}_RAW' in df.columns:
#             lag_factors.append(f'{col}_RAW')
#             lag_factors.append(f'{col}_PCT_CHANGE')
#             lag_factors.append(f'{col}_ZSCORE')

#     # 새로 생성된 모든 뉴스 관련 피처 추가 (Raw, MA, Volatility 포함)
#     new_news_factors = [col for col in df.columns if col.startswith(tuple(news_agg_features)) or col.startswith('News_Count_Vol') or col.startswith('News_Count_Change')]
#     lag_factors.extend(new_news_factors)
    
#     # 중복 제거 및 존재하는 컬럼만 선택
#     lag_factors = list(set(f for f in lag_factors if f in df.columns))
    
#     for factor in lag_factors:
#         for lag in lags:
#             df[f'{factor}_Lag_{lag}'] = df[factor].shift(lag)
            
#     # 보조 지표 추가 (VIX는 이제 VIX_RAW, VIX_PCT_CHANGE, VIX_ZSCORE 형태로 존재)
#     if 'VIX_RAW' in df.columns:
#         df['VIX_RAW_Change_5D'] = df['VIX_RAW'].diff(5)
#     df['SP500_SMA_20'] = df['SP500_Close'].rolling(window=20).mean()
    
#     # 🌟 타겟 변수가 NaN이 되는 마지막 10일을 제거, 그 외 모든 NaN 행 제거
#     df.replace([np.inf, -np.inf], np.nan, inplace=True) # 무한대 값 제거 (pct_change 시 분모 0인 경우 등)
#     df = df.dropna()
    
#     base_features = [col for col in df.columns if not col.endswith(('Return', 'Close', '10D')) and 'SP500_' not in col and 'NASDAQ_' not in col]
#     # 최종적으로 Lagged, Change, SMA, GDP/M2/EPS/DXY/Ratio 피처만 남깁니다.
#     features = [f for f in base_features if ('Lag' in f or 'Change' in f or 'SMA' in f or f in ['GDP', 'M2', 'SP500_EPS', 'NASDAQ_SP500_Ratio'])]
#     # 매크로 피처의 RAW/PCT/ZSCORE 버전 전체 포함
#     features.extend([f for f in base_features if any(f.startswith(macro) for macro in MACRO_FEATURES_TO_ENHANCE)])
#     # 뉴스 피처 전체 포함
#     features.extend([f for f in base_features if any(f.startswith(news_f) for news_f in news_agg_features)])

#     features = list(set(features))
    
#     return df, features


# # ------------------------
# # 4. Streamlit 실행 로직 (뉴스 분석 및 피처 개선 로직 적용)
# # ------------------------

# # 🌟 개선: LGBM 모델도 명시적으로 반환하여 SHAP 및 Importance 플롯에 사용되는 모델의 일관성을 높입니다.
# @st.cache_resource(show_spinner="🚀 Soft Voting 앙상블 모델 훈련 중/로드 중...")
# def train_voting_model(_X_train_df, _y_train, _lgbm_params, _xgb_params, _rf_params, _features):
#     """
#     앙상블 모델(LGBM, XGBoost, RF)을 훈련하고, SHAP/Importance를 위한
#     별도의 LGBM 모델 인스턴스(최종 훈련 세트로 학습)를 반환합니다.
#     """
#     lgbm_model = lgb.LGBMRegressor(**_lgbm_params)
#     xgb_model = xgb.XGBRegressor(**_xgb_params)
#     rf_model = RandomForestRegressor(**_rf_params)
    
#     voting_model = VotingRegressor(
#         estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('rf', rf_model)],
#         weights=[1, 1, 1] 
#     )
    
#     # Voting Regressor 훈련
#     voting_model.fit(_X_train_df, _y_train) 
    
#     final_lgbm_model = voting_model.estimators_[0] 
    
#     return voting_model, final_lgbm_model

# st.markdown("---")
# # UI 입력 요소
# col1, col2, col3 = st.columns([1.5, 1, 1])
# with col1:
#     news_query = st.text_input(
#         "📰 뉴스 감성 분석 키워드", 
#         value="미국 증시 전망|금리 인상|연준|경기 침체", 
#         help="네이버 뉴스 검색에 사용될 키워드를 '|' (파이프 기호)로 구분하여 입력하세요. (예: S&P 500|경기 침체)"
#     )
# with col2:
#     start_date = st.date_input("분석 시작일", datetime.now() - timedelta(days=365 * 2)) 
# with col3:
#     end_date = st.date_input("분석 종료일", datetime.now())
    
# if st.button("🚀 데이터 로드, 분석 및 예측 시작 (10일 추세 예측)", type="primary", use_container_width=True):
    
#     # 1. 데이터 로드
#     market_df = load_market_data(start_date, end_date)
#     fred_data = get_fred_data()
#     fg_df = get_fear_greed_index(limit=365 * 3)
#     trends_df = pd.DataFrame() 
    
#     # 1-2. 뉴스 감성 및 키워드 분석 (2회 요청 로직 적용)
#     # 모델 로드 실패 시에도 코드 실행을 위해 sentiment_model 로드 여부 체크
#     if not sentiment_model:
#         st.warning("⚠️ 감성 분석 모델 로드에 실패하여 뉴스 감성 피처는 0으로 채워집니다. 진행합니다.")
        
#     with st.spinner(f"뉴스 크롤링 및 감성/키워드 분석 중... (키워드: {news_query})"):
#         # Description을 포함하여 뉴스 데이터 크롤링
#         news_batch_1 = get_naver_news_api(news_query, display=100, start=1) 
#         news_batch_2 = get_naver_news_api(news_query, display=100, start=101)
        
#         all_news = pd.concat([news_batch_1, news_batch_2]).drop_duplicates(subset=['Title']).reset_index(drop=True)
        
#         if all_news.empty or 'Date' not in all_news.columns or all_news['Date'].isnull().all():
#             st.warning("⚠️ 네이버 API로부터 유효한 기사 데이터를 수집하지 못했습니다. 뉴스 분석을 건너뜁니다.")
#             news_grouped = pd.DataFrame()
#         else:
#             load_start_date = start_date - timedelta(days=50)
#             filtered_news = all_news[(all_news['Date'] >= load_start_date) & (all_news['Date'] <= end_date)].copy()
            
#             if not filtered_news.empty:
#                 # 1. 감성 점수 (기존)
#                 filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
                
#                 # 2. Pos/Neg/Fed Keyword Analysis (NEW)
#                 filtered_news[['Pos_Count', 'Neg_Count', 'Pos_Neg_Ratio', 'Fed_Ratio']] = filtered_news.apply(
#                     lambda row: analyze_text_keywords(row['Title'], row['Description']), 
#                     axis=1, result_type='expand'
#                 )
                
#                 # 3. Daily Aggregation (News Count, Avg Sentiments, Avg Ratios)
#                 # News_Count (Feature 4), Avg_Pos_Neg_Ratio (Feature 2), Avg_Fed_Ratio (Feature 3) 포함
#                 news_grouped = filtered_news.groupby('Date').agg(
#                     Sentiment_Score=('Sentiment_Score', 'mean'), 
#                     News_Count=('Title', 'count'), 
#                     Avg_Pos_Neg_Ratio=('Pos_Neg_Ratio', 'mean'), 
#                     Avg_Fed_Ratio=('Fed_Ratio', 'mean') 
#                 )
#                 st.success(f"✅ 뉴스 감성 및 키워드 분석 완료! (총 {len(filtered_news)}개 기사 분석)")
#             else:
#                 st.warning("⚠️ 지정된 기간에 해당하는 기사가 없습니다. 뉴스 분석을 건너뜁니다.")
#                 news_grouped = pd.DataFrame()


#     # 2. 데이터 병합
#     df_merge = market_df
#     if not fg_df.empty: df_merge = pd.merge(df_merge, fg_df, left_index=True, right_index=True, how='left')
#     for name, df_fred in fred_data.items(): df_merge = pd.merge(df_merge, df_fred, left_index=True, right_index=True, how='left')
#     if not news_grouped.empty:
#         df_merge = pd.merge(df_merge, news_grouped, left_index=True, right_index=True, how='left')
    
#     # 결측치 처리 (FFILL 후 0으로 채우기 - 주말/공휴일 등)
#     df_merge = df_merge.fillna(method='ffill').fillna(0)
    
#     # 3. 피처 엔지니어링 및 데이터 준비
#     df_ml, features_full = create_features(df_merge)
    
#     df_ml = df_ml.tail(500)
#     df_ml = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)]

#     if len(df_ml) <= 100:
#         st.error("데이터가 부족합니다. 분석 기간을 늘리세요. (최소 100일 필요)")
#         st.stop()
        
#     X_full = df_ml[features_full]
#     y = df_ml['Return_10D'] 
    
#     # 4. 피처 선택: LightGBM 중요도 기반
#     st.subheader("⚙️ 피처 선택 (LightGBM 중요도 기반 Top 15)")
    
#     LGBM_PARAMS = {'objective': 'regression', 'metric': 'rmse', 'n_estimators': 300, 'learning_rate': 0.01, 'num_leaves': 21, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
#     XGB_PARAMS = {'objective': 'reg:squarederror', 'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1}
#     RF_PARAMS = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}

#     # 최종 피처 선택을 위해 전체 데이터셋으로 임시 LGBM 훈련
#     temp_model = lgb.LGBMRegressor(**LGBM_PARAMS) 
#     temp_model.fit(X_full, y)

#     feature_importances = pd.Series(temp_model.feature_importances_, index=X_full.columns)
#     features = feature_importances.nlargest(15).index.tolist()
    
#     st.info(f"선택된 피처 수: {len(features)}개. (전체 {len(features_full)}개 중 상위 15개, LGBM 기반)")
#     st.markdown(f"**선택된 뉴스 피처:** `{'`, `'.join([f for f in features if 'Sentiment' in f or 'News_Count' in f or 'Fed_Ratio' in f])}`")
#     st.markdown(f"**선택된 매크로 피처 (RAW/PCT/ZSCORE 포함):** `{'`, `'.join([f for f in features if any(f.startswith(macro) and ('RAW' in f or 'PCT' in f or 'ZSCORE' in f) for macro in ['YIELD_CURVE', 'BBB_OAS', 'DXY', 'VIX', 'WTI', 'GOLD', 'COPPER'])])}`")
#     X = df_ml[features] 
    
#     # 전체 데이터 스케일링 준비
#     scaler = MinMaxScaler()
#     X_scaled_all = scaler.fit_transform(X)
#     X_scaled_all_df = pd.DataFrame(X_scaled_all, columns=X.columns, index=X.index)
    
#     # 테스트 데이터셋 분리 (마지막 N일)
#     test_size = max(30, int(0.2 * len(X_scaled_all_df)))
    
#     # 🌟 X_test_df를 위한 명확한 인덱싱 및 분리
#     X_train_df, X_test_df = X_scaled_all_df.iloc[:-test_size], X_scaled_all_df.iloc[-test_size:]
#     y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
#     # 테스트 기간 시작일 정의 (시각화에 사용)
#     test_start_date = X_test_df.index[0]

#     st.markdown(f"**훈련 기간:** {X_train_df.index[0]} ~ {X_train_df.index[-1]}")
#     st.markdown(f"**테스트 기간:** {test_start_date} ~ {X_test_df.index[-1]} (총 {test_size}일)")
    
#     # 5. 앙상블 모델 훈련 및 시계열 교차검증 (TS Split)
#     st.header("📊 시계열 교차검증 (TimeSeriesSplit)")
    
#     n_splits = 2 
#     tscv = TimeSeriesSplit(n_splits=n_splits)
    
#     r2_scores_lgbm = []
    
#     with st.spinner(f"⏳ TimeSeriesSplit 교차검증 중 (폴드 {n_splits}개, n_estimators=300, Early Stopping=30 적용)..."):
        
#         for i, (train_index, val_index) in enumerate(tscv.split(X_train_df)):
#             X_train_fold, X_val_fold = X_train_df.iloc[train_index], X_train_df.iloc[val_index]
#             y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

#             lgbm_fold = lgb.LGBMRegressor(**LGBM_PARAMS)
            
#             lgbm_fold.fit(X_train_fold, y_train_fold,
#                           eval_set=[(X_val_fold, y_val_fold)],
#                           eval_metric='rmse',
#                           callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
            
#             y_val_pred = lgbm_fold.predict(X_val_fold)
#             r2_scores_lgbm.append(r2_score(y_val_fold, y_val_pred))
            
#         avg_r2 = np.mean(r2_scores_lgbm)
#         st.info(f"✅ TimeSeriesSplit 평균 R² (LGBM 기준): **{avg_r2:.4f}**")
#         st.dataframe(pd.DataFrame({'Fold': range(1, n_splits + 1), 'R2 Score': r2_scores_lgbm}), use_container_width=True)
#     st.markdown("---")

#     voting_model, final_lgbm_model = train_voting_model(
#         X_train_df, 
#         y_train, 
#         LGBM_PARAMS, 
#         XGB_PARAMS, 
#         RF_PARAMS, 
#         tuple(features) 
#     )
        
#     # 훈련 잔차 계산 (CI용)
#     y_train_pred_lgbm = final_lgbm_model.predict(X_train_df)
#     residuals = y_train - y_train_pred_lgbm
#     residual_std = residuals.std()
#     CI_FACTOR = 1.645 * residual_std 
    
#     # 테스트 기간 예측 (X_test_df 사용)
#     y_test_pred = voting_model.predict(X_test_df)

#     # 다음 10일 예측 및 CI 계산
#     # X_scaled_all_df의 마지막 행은 가장 최신의 데이터 (오늘 날짜)를 포함
#     last_data_scaled = X_scaled_all_df.iloc[-1].values.reshape(1, -1)
#     last_data_df = pd.DataFrame(last_data_scaled, columns=X_scaled_all_df.columns)
    
#     next_day_return_pred = voting_model.predict(last_data_df)[0]
#     low_ci = next_day_return_pred - CI_FACTOR
#     high_ci = next_day_return_pred + CI_FACTOR
    
#     # 6. 결과 출력
#     mse = mean_squared_error(y_test, y_test_pred)
#     r2 = r2_score(y_test, y_test_pred)

#     st.markdown("---")
#     st.header("📈 최종 예측 결과 및 모델 성능")
    
#     col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

#     def format_pred_value(value): return f"{value:+.2f}%"

#     with col_pred1:
#         st.metric(label="📊 향후 10거래일 S&P 500 예측 수익률", value=format_pred_value(next_day_return_pred), delta=f"90% CI: {low_ci:+.2f}% ~ {high_ci:+.2f}%")
#     with col_pred2:
#         st.metric(label="✅ 테스트 R² (앙상블)", value=f"{r2:.2f}", help=f"MSE: {mse:.4f}. 1에 가까울수록 적합도가 높음.")
#     with col_pred3:
#         # VIX_RAW가 이제 원본 VIX
#         current_vix = df_ml['VIX_RAW'].iloc[-1] if 'VIX_RAW' in df_ml.columns else df_ml['VIX'].iloc[-1]
#         vix_trend = "하락 (강세) 🟢" if df_ml['VIX_RAW_Change_5D'].iloc[-1] < 0 else "상승 (약세) 🔴"
#         st.metric(label="🔥 현재 VIX 지수", value=f"{current_vix:.2f}", delta=vix_trend)
#     with col_pred4:
#         action = "강력 매수/추세 추종" if next_day_return_pred > 1.0 and low_ci > 0.0 else ("매도/리스크 관리" if next_day_return_pred < -1.0 else "관망/중립")
#         action_color = "#D4EDDA" if "매수" in action else ("#F8D7DA" if "매도" in action else "#FFF3CD")
#         st.markdown(f"""
#         <div style='padding: 10px; border-radius: 5px; text-align: center; 
#             background-color: {action_color}; color: {"#155724" if "매수" in action else ("#721C24" if "매도" in action else "#856404")}; 
#             font-weight: bold; margin-top: 15px;'>
#             최종 투자 시그널: {action}
#         </div>
#         """, unsafe_allow_html=True)
        
#     st.markdown("---")

#     # 7. SHAP 해석 추가
#     st.header("💡 예측 해석: SHAP (10일 추세 예측에 기여)")
#     st.markdown(f"**SHAP**을 사용하여 모델이 최종 $\mathbf{{10}}$일 예측(`{next_day_return_pred:+.2f}%`)을 산출하는 데 기여한 팩터의 영향력을 분석합니다. (**VotingRegressor 내부의 LightGBM 모델 기준**)")
    
#     try:
#         # final_lgbm_model을 사용하여 SHAP 해석 (일관성 확보)
#         explainer = shap.TreeExplainer(final_lgbm_model) 
#         # last_data_df는 이미 피처 이름을 가지고 있으므로 바로 사용
#         shap_values = explainer.shap_values(last_data_df)
        
#         shap_df = pd.DataFrame({
#             'Feature': last_data_df.columns,
#             'SHAP Value': shap_values[0],
#             # Feature Value는 스케일링된 값 (0-1)입니다. 해석의 편의를 위해 원본 데이터를 보여줄 수도 있으나,
#             # 모델 입력은 스케일링된 값이므로 일단 Scaled Value를 표시합니다.
#             'Scaled Feature Value': last_data_df.iloc[0].values
#         })
        
#         shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
#         shap_df = shap_df.sort_values('Abs SHAP', ascending=False).head(5)

#         fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
#                               color='SHAP Value', color_continuous_scale=px.colors.diverging.RdBu,
#                               title=f"향후 10일 예측({next_day_return_pred:+.2f}%)에 기여한 Top 5 팩터",
#                               hover_data={'Scaled Feature Value': True, 'SHAP Value': ':.4f'})
#         fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
#         st.plotly_chart(fig_shap, use_container_width=True)

#     except Exception as e:
#         st.warning(f"⚠️ SHAP 해석 로드 중 오류 발생: {e}.")
#     st.markdown("---")

#     # 8. 피처 상관관계 히트맵 시각화 추가
#     st.header("🔗 피처 상관관계 히트맵")
#     st.markdown("훈련에 사용된 **LightGBM 중요도 기반 Top 15 피처**와 타겟(`Return_10D`) 간의 상관관계를 시각적으로 확인합니다.")

#     correlation_df = df_ml[features + ['Return_10D']].copy().rename(columns={'Return_10D': 'Target_10D_Return'})
#     N_TOP_FEATURES = len(features) 
    
#     try:
#         corr_matrix = correlation_df.corr()
#         fig_heatmap = px.imshow(corr_matrix, 
#                                  x=corr_matrix.columns, 
#                                  y=corr_matrix.columns,
#                                  color_continuous_scale='RdBu_r', 
#                                  title=f'LGBM 선택 {N_TOP_FEATURES}개 피처 간의 상관관계 히트맵')
#         fig_heatmap.update_xaxes(side="top")
        
#         annotations = []
#         for i, row in enumerate(corr_matrix.values):
#             for j, val in enumerate(row):
#                 annotations.append(
#                     dict(x=corr_matrix.columns[j], y=corr_matrix.columns[i], 
#                           text=f"{val:.2f}", showarrow=False, font=dict(color="black" if abs(val) < 0.6 else "white"))
#                 )
#         fig_heatmap.update_layout(annotations=annotations, height=800)
#         st.plotly_chart(fig_heatmap, use_container_width=True)

#     except Exception as e:
#         st.warning(f"⚠️ 히트맵 생성 중 오류: {e}")
        
#     st.markdown("---")

#     # 9. 주요 매크로 팩터 추이 시각화
#     st.header("📊 주요 매크로 팩터 추이 (S&P 500과 비교)")
    
#     df_macro_plot = df_ml[df_ml.index >= start_date].copy()

#     fig_macro = go.Figure()
    
#     fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['SP500_Close'], name='S&P 500 (좌측 축)', line=dict(color='#1f77b4', width=2), yaxis='y1'))
#     # RAW 값 사용
#     if 'YIELD_CURVE_RAW' in df_macro_plot.columns:
#         fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['YIELD_CURVE_RAW'], name='장단기 금리차 (10Y-2Y)', line=dict(color='red', width=1.5), yaxis='y2', opacity=0.8))
#         fig_macro.add_hline(y=0, line_dash="dash", line_color="red", yref="y2")     
#     if 'BBB_OAS_RAW' in df_macro_plot.columns:
#         fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['BBB_OAS_RAW'], name='BBB 회사채 스프레드', line=dict(color='green', width=1.5), yaxis='y3', opacity=0.8))
#     if 'DXY_RAW' in df_macro_plot.columns:
#         fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['DXY_RAW'], name='USD Index (DXY)', line=dict(color='purple', width=1.5), yaxis='y4', opacity=0.8))

#     fig_macro.update_layout(title="S&P 500 vs. 경기/신용 리스크 지표", xaxis_title="날짜",
#         yaxis=dict(title=dict(text='S&P 500 종가', font=dict(color="#1f77b4")), domain=[0, 1]),
#         yaxis2=dict(title=dict(text='금리차 (%)', font=dict(color="red")), overlaying='y', side='right', position=0.90, showgrid=False),
#         yaxis3=dict(title=dict(text='BBB OAS', font=dict(color="green")), overlaying='y', side='right', position=0.95, showgrid=False),
#         yaxis4=dict(title=dict(text='DXY', font=dict(color="purple")), overlaying='y', side='right', position=1.0, showgrid=False),
#         hovermode="x unified", height=600, legend=dict(x=0, y=1.05, orientation="h"))
    
#     st.plotly_chart(fig_macro, use_container_width=True)

#     # 10. 예측 vs. 실제 수익률 시각화 (앙상블 모델)
#     st.subheader("📈 Soft Voting 앙상블 예측 vs. 실제 수익률 (90% 신뢰구간)")
    
#     y_test_df = pd.DataFrame({
#         'Actual': y_test, 'Predicted': y_test_pred,
#         'Low_CI': y_test_pred - CI_FACTOR, 'High_CI': y_test_pred + CI_FACTOR
#     }, index=X_test_df.index) # X_test_df.index 사용

#     fig_pred = go.Figure()

#     fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['High_CI'], mode='lines', line=dict(width=0), showlegend=False))
#     fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Low_CI'], fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', mode='lines', line=dict(width=0), name='90% 신뢰구간'))
#     fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Actual'], mode='markers', name='실제 10일 누적 수익률', marker=dict(color='blue', size=5, opacity=0.8)))
#     fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Predicted'], mode='lines', name='앙상블 예측 수익률 (Median)', line=dict(color='red', width=2)))

#     fig_pred.update_layout(title=f"테스트 기간 S&P 500 10일 누적 수익률 예측 결과", xaxis_title="날짜", yaxis_title="수익률(%)", hovermode="x unified", height=500)
#     st.plotly_chart(fig_pred, use_container_width=True)
    
#     # 11. 팩터 중요도 시각화
#     st.subheader("🔍 팩터 중요도 (LightGBM 기준)")
    
#     # final_lgbm_model을 사용하여 중요도 시각화 (일관성 확보)
#     importance_df = pd.DataFrame({
#         'Feature': features,
#         'Importance': final_lgbm_model.feature_importances_
#     }).sort_values('Importance', ascending=False).head(15)

#     fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
#                       title='LightGBM 모델 상위 15개 팩터 중요도',
#                       color='Importance', color_continuous_scale=px.colors.sequential.Viridis)
#     fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
#     st.plotly_chart(fig_imp, use_container_width=True)


# st.markdown("---")
# st.warning("⚠️ **면책 조항:** 이 모델은 교육 및 분석 목적으로만 제공됩니다. 실제 투자에 사용하기 전에 충분한 검증과 리스크 분석을 수행해야 합니다.")
