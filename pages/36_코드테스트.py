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
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import MinMaxScaler
import urllib.parse
import FinanceDataReader as fdr
import lightgbm as lgb
import xgboost as xgb
import time
from concurrent.futures import ThreadPoolExecutor
import re
import shap
from sklearn.inspection import permutation_importance
import logging

# 로깅 레벨 설정 (디버깅 정보 표시)
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
# 0. 매크로 데이터 수집 함수 (DatetimeIndex 유지 및 Naive Index 통일)
# ------------------------
@st.cache_data(show_spinner="⏳ FRED 데이터 (금리차, M2, BBB OAS, SP500 P/E) 로드 중...")
def get_fred_data():
    """FRED에서 여러 경제 지표를 병렬로 가져옵니다. (DatetimeIndex 유지)"""
    fred_api_key = st.secrets.get("fred", {}).get("FRED_API_KEY")
    if not fred_api_key:
        st.warning("⚠️ FRED API 키가 설정되지 않아 데이터를 로드할 수 없습니다.")
        return {}

    TICKERS = {
        "DGS10": "10Y", "DGS2": "2Y",  
        "BAMLC0A4CBBB": "BBB_OAS", "M2SL": "M2", "GDPC1": "GDP",
        "SP500PE": "SP500_PER" # SP500PE는 P/E Ratio (주가수익률)입니다.
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
            
            # DatetimeIndex로 변환하며 Timezone 제거 (Naive)
            # Series에 대해 .dt를 사용하는 것은 올바릅니다.
            df['date'] = pd.to_datetime(df['date']).dt.normalize().dt.tz_localize(None, errors='ignore')
            
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            # 1일 지연 피처를 위해 shift(1) 적용 (FRED 데이터는 주로 관측일 다음날 사용)
            df = df[['date', 'value']].rename(columns={'value': TICKERS[ticker]}).set_index('date')
            return ticker, df.shift(1).ffill() 
        except Exception as e:
            logging.error(f"FRED data load failed ({ticker}): {e}")
            return ticker, pd.DataFrame()

    start_date = datetime.now().date() - timedelta(days=365 * 3)
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_single_fred, ticker, start_date): ticker for ticker in TICKERS.keys()}
        for future in futures:
            ticker, df = future.result()
            if not df.empty: 
                results[TICKERS[ticker]] = df

    if '10Y' in results and '2Y' in results:
        df_yield = pd.merge(results['10Y'], results['2Y'], left_index=True, right_index=True, how='inner')
        results['YIELD_CURVE'] = (df_yield['10Y'] - df_yield['2Y']).rename('YIELD_CURVE').to_frame()
    return results

@st.cache_data(show_spinner="⏳ Fear & Greed Index 로드 중...")
def get_fear_greed_index(limit=1095): 
    """Alternative.me에서 Fear & Greed Index를 가져옵니다. (DatetimeIndex 유지, 1일 shift 적용)"""
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(float)
        
        # DatetimeIndex 유지, Naive Index로 변환
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize().dt.tz_localize(None, errors='ignore') 
        
        df = df.rename(columns={"value": "FGI", "timestamp": "Date"})
        df = df[["Date", "FGI"]].sort_values("Date").set_index('Date')
        
        # 1일 지연 피처를 위해 shift(1) 적용
        return df.shift(1).ffill()
    except Exception as e:
        logging.error(f"Fear & Greed Index load error: {e}")
        return pd.DataFrame()

# ------------------------
# 1. 팩터 및 증시 데이터 로드
# ------------------------
@st.cache_data(show_spinner="⏳ 주가, 원자재, DXY, NASDAQ 데이터 로드 중...")
def load_market_data(start_date, end_date):
    """S&P 500, NASDAQ, VIX, WTI, Copper, Gold, DXY 데이터를 로드합니다. (DatetimeIndex 유지, Naive Index 통일)"""
    # 롤링 피처 생성을 위해 충분한 과거 데이터 확보
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
            # FinanceDataReader는 Naive DatetimeIndex를 반환
            df = fdr.DataReader(ticker, start=load_start_date, end=end_date)
            df = df[['Close']].rename(columns={'Close': name})
            
            # [CRITICAL FIX]: DatetimeIndex에 .dt를 사용할 수 없습니다. Index 자체의 메서드를 사용합니다.
            # 명시적으로 Naive DatetimeIndex로 변환
            df.index = df.index.tz_localize(None, errors='ignore').normalize() 
            
            all_data.append(df)
            time.sleep(0.05)
        except Exception as e:
            logging.error(f"{name} ({ticker}) data load failed: {e}")
            continue
            
    progress_bar.empty()
    st.success("✅ 시장 데이터 로드 완료!")
    if not all_data: return pd.DataFrame()
    
    # 모든 데이터를 외부 조인하여 S&P 500 거래일 기준으로 통일
    df_merged = pd.concat(all_data, axis=1, join='outer').sort_index()
    df_merged.index.name = 'Date'
    
    # DXY만 1일 지연 피처를 적용 (FRED/FGI와 유사)
    if 'DXY' in df_merged.columns:
        df_merged['DXY'] = df_merged['DXY'].shift(1).ffill()
        
    return df_merged

# ------------------------
# 2. 감성 분석 모델 로드 및 함수
# ------------------------

@st.cache_resource
def load_sentiment_model():
    """Hugging Face에서 한국어 감성 분석 모델을 로드합니다. 실패 시 None 반환."""
    hf_token = st.secrets.get("HF_TOKEN")
    model_name = "snunlp/KR-FinBert-SC"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        st.success("✅ 감성 분석 모델 로드 완료!")
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Sentiment analysis model '{model_name}' load failed: {e}")
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    """주어진 텍스트에 대한 감성 점수 계산 (긍정 - 부정)."""
    if sentiment_model is None or not text: return 0.0
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(): outputs = sentiment_model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    
    # 모델의 라벨 ID를 기반으로 긍정/부정 인덱스 찾기
    neg_idx, pos_idx = None, None
    for idx, label in sentiment_model.config.id2label.items():
        if 'negative' in label.lower() or '부정' in label: neg_idx = idx
        elif 'positive' in label.lower() or '긍정' in label: pos_idx = idx
    
    negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0
    return positive_score - negative_score

def get_naver_news_api(query, display=100, start=1, sort="date"): 
    """Naver News Search API에서 데이터를 가져옵니다."""
    client_id = st.secrets.get("naver", {}).get("client_id")
    client_secret = st.secrets.get("naver", {}).get("client_secret")
    if not client_id or not client_secret:
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
            try: 
                # KST Timezone을 제거하고 Naive Datetime으로 정규화
                pub_date_dt = pd.to_datetime(datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")).normalize().tz_localize(None)
            except Exception: 
                pub_date_dt = None
            news_data.append({'Date': pub_date_dt, 'Title': title})
        return pd.DataFrame(news_data)
    except Exception as e:
        logging.error(f"Naver API call failed: {e}")
        return pd.DataFrame(columns=['Date', 'Title'])

# ------------------------
# 3. 피처 엔지니어링 함수
# ------------------------
def create_features(df_merge):
    """모든 팩터에 대해 시계열 피처를 생성하고 데이터를 정리합니다."""
    df = df_merge.copy()
    
    if 'SP500_Close' not in df.columns or df['SP500_Close'].empty:
        st.error("❌ S&P 500 데이터가 없어 피처를 생성할 수 없습니다.")
        return pd.DataFrame(), []

    # S&P 500 종가 결측치 처리 (거래일이 아닌 경우 발생 가능)
    df['SP500_Close'] = df['SP500_Close'].ffill().bfill()
    
    if 'NASDAQ_Close' in df.columns and 'SP500_Close' in df.columns:
        # NASDAQ/SP500 비율의 일일 변화율
        df['NASDAQ_SP500_Ratio'] = (df['NASDAQ_Close'] / df['SP500_Close']).ffill()
    
    # Target Variable: 향후 10일 수익률
    df['Return_10D'] = df['SP500_Close'].pct_change(periods=10).shift(-10) * 100
    df['Daily_Return'] = df['SP500_Close'].pct_change() * 100

    lags = [1, 3, 5, 10] 
    
    # 팩터 목록을 명확히 정의 (SP500_EPS -> SP500_PER 로 변경)
    lag_factors = ['Daily_Return', 'VIX', 'FGI', 'Sentiment_Score', 
                   'YIELD_CURVE', 'BBB_OAS', 'WTI', 'GOLD', 'COPPER',
                   'DXY', 'NASDAQ_SP500_Ratio', 'SP500_PER']
    
    for factor in lag_factors:
        if factor in df.columns:
            for lag in lags:
                df[f'{factor}_Lag_{lag}'] = df[factor].shift(lag)
            
    # 변화율 피처 추가
    df['VIX_Change_5D'] = df['VIX'].diff(5)
    df['SP500_SMA_20'] = df['SP500_Close'].rolling(window=20).mean()
    df['SP500_MOM_20'] = df['SP500_Close'] / df['SP500_SMA_20']

    # 타겟 변수가 NaN이 되는 마지막 10일 제거 및 피처 결측치 포함 행 제거
    df = df.dropna()
    
    # 최종 사용할 피처 목록 정의 (SP500_EPS -> SP500_PER 로 변경)
    features = [col for col in df.columns if 'Lag' in col or 'Change' in col or 'Ratio' in col or 'MOM' in col or col in ['GDP', 'M2', 'SP500_PER', 'DXY', 'YIELD_CURVE', 'BBB_OAS', 'FGI', 'VIX', 'Sentiment_Score']]
    features = list(set(features)) # 중복 제거
    
    return df, features

# ------------------------
# 4. Rolling Feature Importance 함수
# ------------------------
@st.cache_data(show_spinner="🛞 Rolling Feature Importance 계산 중...")
def get_rolling_feature_importance(_X_train_df, _y_train, _lgbm_params, window_size_months=6, top_n_features=15):
    """
    롤링 윈도우 방식으로 LGBM Feature Importance를 계산하고,
    빈도수와 평균 중요도를 기반으로 최종 피처를 선택합니다.
    """
    X_train_df = _X_train_df.copy()
    y_train = _y_train.copy()
    
    min_window_days = 100 # 최소 윈도우 크기 (약 4개월)
    window_days = 21 * window_size_months # 6개월 윈도우 크기 (거래일 기준)
    
    all_importances = {}
    
    # 롤링 윈도우 생성 (50% 겹침)
    start_idx = 0 
    
    progress_bar = st.progress(0, text="Rolling Feature Importance 계산 중...")
    
    # 윈도우 시작 인덱스가 전체 데이터 크기를 넘지 않고, 최소 윈도우 크기를 확보할 수 있을 때까지 반복
    while start_idx + min_window_days <= len(X_train_df): 
        end_idx = min(start_idx + window_days, len(X_train_df))
        
        # 실제 훈련에 사용할 데이터 윈도우
        X_roll = X_train_df.iloc[start_idx:end_idx]
        y_roll = y_train.iloc[start_idx:end_idx]
        
        try:
            # 롤링 윈도우 훈련
            lgbm_roll = lgb.LGBMRegressor(**_lgbm_params)
            lgbm_roll.fit(X_roll, y_roll)
            
            # 중요도 추출 및 저장
            importance = pd.Series(lgbm_roll.feature_importances_, index=X_roll.columns)
            for feature, imp in importance.items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(imp)
                
        except Exception as e:
            # 훈련 실패 시 건너뛰기
            logging.warning(f"Rolling Window training failed at {X_roll.index[0].date()} ~ {X_roll.index[-1].date()}: {e}")
        
        start_idx += int(window_days * 0.5) # 50%씩 겹치도록 윈도우를 이동
        progress_bar.progress(start_idx / len(X_train_df), text=f"Rolling Window 처리 중: {X_roll.index[0].date()} ~ {X_roll.index[-1].date()}")
            
    progress_bar.empty()
    
    if not all_importances:
        return None, None
        
    # 결과 집계: 빈도수와 평균 중요도를 합산
    summary = []
    for feature, imps in all_importances.items():
        summary.append({
            'Feature': feature,
            'Mean_Importance': np.mean(imps),
            'Frequency': len(imps) # 등장 빈도 (윈도우 수)
        })
        
    df_summary = pd.DataFrame(summary).sort_values(['Frequency', 'Mean_Importance'], ascending=False)
    
    # 최종 선택 로직: 빈도수 > 평균 중요도 순
    selected_features = df_summary['Feature'].tolist()[:top_n_features]
    
    return selected_features, df_summary.head(top_n_features)

# ------------------------
# 5. Streamlit 실행 로직
# ------------------------

@st.cache_resource(show_spinner="🚀 Soft Voting 앙상블 모델 훈련 중/로드 중...")
def train_voting_model(_X_train_df, _y_train, _lgbm_params, _xgb_params, _rf_params):
    """선택된 피처로 앙상블 모델을 훈련하고, SHAP 분석을 위한 LGBM 모델도 함께 반환합니다."""
    lgbm_model = lgb.LGBMRegressor(**_lgbm_params)
    xgb_model = xgb.XGBRegressor(**_xgb_params)
    rf_model = RandomForestRegressor(**_rf_params)
    
    voting_model = VotingRegressor(
        estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('rf', rf_model)],
        weights=[1, 1, 1] 
    )
    # SHAP 분석에 사용될 LGBM 모델은 별도로 훈련 (앙상블 모델은 SHAP Explainer가 지원하지 않음)
    lgbm_shap_model = lgb.LGBMRegressor(**_lgbm_params)

    # 훈련 (같은 데이터로 훈련해야 일관성 유지)
    voting_model.fit(_X_train_df, _y_train) 
    lgbm_shap_model.fit(_X_train_df, _y_train)
    
    return voting_model, lgbm_shap_model

LGBM_PARAMS = {'objective': 'regression', 'metric': 'rmse', 'n_estimators': 300, 'learning_rate': 0.01, 'num_leaves': 21, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
XGB_PARAMS = {'objective': 'reg:squarederror', 'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1}
RF_PARAMS = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}


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
    start_date = st.date_input("분석 시작일", datetime.now().date() - timedelta(days=365 * 2)) 
with col3:
    end_date = st.date_input("분석 종료일", datetime.now().date())
    
if st.button("🚀 데이터 로드, 분석 및 예측 시작 (피처 안정화 적용)", type="primary", width='stretch'):
    
    # 1. 데이터 로드
    market_df = load_market_data(start_date, end_date)
    fred_data = get_fred_data()
    fg_df = get_fear_greed_index(limit=365 * 3)
    
    if market_df.empty:
        st.error("❌ 시장 데이터(S&P 500 포함) 로드에 실패했습니다. API 키 또는 네트워크 상태를 확인하세요.")
        st.stop()
        
    # 1-2. 뉴스 감성 분석 (모델 로드 여부 확인)
    is_sentiment_available = sentiment_model is not None and tokenizer is not None
    if not is_sentiment_available:
        st.warning("⚠️ 감성 분석 모델 로드에 실패했습니다. 해당 피처는 0으로 채워지며 분석에서 제외됩니다. Hugging Face 토큰을 확인해주세요.")

    with st.spinner(f"뉴스 크롤링 및 감성 분석 중... (키워드: {news_query})"):
        if is_sentiment_available:
            news_batch_1 = get_naver_news_api(news_query, display=100, start=1) 
            news_batch_2 = get_naver_news_api(news_query, display=100, start=101)
            
            all_news = pd.concat([news_batch_1, news_batch_2]).drop_duplicates(subset=['Title']).reset_index(drop=True)
            
            if all_news.empty or 'Date' not in all_news.columns or all_news['Date'].isnull().all():
                st.warning("⚠️ 네이버 API로부터 유효한 기사 데이터를 수집하지 못했습니다. 감성 분석을 건너뜁니다.")
                news_grouped = pd.DataFrame(columns=['Sentiment_Score'])
            else:
                # 필터링 및 분석
                all_news['Date'] = pd.to_datetime(all_news['Date']).dt.normalize().dt.tz_localize(None, errors='ignore')
                load_start_date = start_date - timedelta(days=50)

                filtered_news = all_news[
                    (all_news['Date'] >= pd.to_datetime(load_start_date)) & 
                    (all_news['Date'] <= pd.to_datetime(end_date))
                ]
                
                if not filtered_news.empty:
                    # 감성 분석 점수 계산
                    filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
                    # Naive DatetimeIndex를 기준으로 평균 집계
                    news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().to_frame() 
                    st.success(f"✅ 뉴스 감성 분석 완료! (총 {len(filtered_news)}개 기사 분석)")
                else:
                    st.warning("⚠️ 지정된 기간에 해당하는 기사가 없습니다. 감성 분석을 건너뜁니다.")
                    news_grouped = pd.DataFrame(columns=['Sentiment_Score'])
        else:
            news_grouped = pd.DataFrame(columns=['Sentiment_Score'])


    # 2. 데이터 병합
    df_merge = market_df.copy()

    # 2-1. 매크로 데이터 병합 (FRED, FGI)
    for name, df_fred in fred_data.items():
        if not df_fred.empty:
            df_merge = pd.merge(df_merge, df_fred, left_index=True, right_index=True, how='left')
            
    if not fg_df.empty:
        df_merge = pd.merge(df_merge, fg_df, left_index=True, right_index=True, how='left')
        
    # 2-2. 뉴스 감성 분석 데이터 병합
    if not news_grouped.empty:
        df_merge = pd.merge(df_merge, news_grouped, left_index=True, right_index=True, how='left')
    
    # 2-3. 최종 결측치 처리 (ffill 후 bfill로 초기 결측치도 처리)
    df_merge = df_merge.ffill().bfill().fillna(0) # 마지막으로 남은 결측치는 0으로 채움
        
    # 3. 피처 엔지니어링 및 데이터 준비
    df_ml, features_full = create_features(df_merge)
    
    if df_ml.empty:
        st.error("❌ 데이터 병합 및 피처 생성 후 유효한 데이터가 없습니다. 데이터 로드 기간을 확인하세요.")
        st.stop()
        
    # 날짜 범위 조정 및 데이터 축소
    df_ml = df_ml[(df_ml.index >= pd.to_datetime(start_date)) & (df_ml.index <= pd.to_datetime(end_date))]
    df_ml = df_ml.tail(500) # 최근 500개만 사용 (계산 시간 절약)

    if len(df_ml) <= 100:
        st.error("데이터가 부족합니다. 분석 기간을 늘리세요. (최소 100일 필요)")
        st.stop()
        
    # 선택된 피처만 포함하는 X_full 구성
    features_full = [f for f in features_full if f in df_ml.columns] # 데이터에 없는 피처는 제거
    X_full = df_ml[features_full]
    y = df_ml['Return_10D'] 

    # 4. 데이터 스케일링 준비 및 분할
    scaler = MinMaxScaler()
    X_scaled_all = scaler.fit_transform(X_full)
    X_scaled_all_df = pd.DataFrame(X_scaled_all, columns=X_full.columns, index=X_full.index)
    
    # 마지막 날짜는 다음 예측에 사용
    test_size = max(30, int(0.2 * len(X_scaled_all_df)))
    X_train_df, X_test_df = X_scaled_all_df.iloc[:-test_size], X_scaled_all_df.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    
    # 5. 🌟 Rolling Feature Importance로 피처 선택
    st.header("⚙️ 피처 선택 (Rolling Feature Importance 기반 Top 15)")
    
    selected_features, df_importance_summary = get_rolling_feature_importance(
        X_train_df, y_train, LGBM_PARAMS, window_size_months=6, top_n_features=15
    )
    
    if selected_features is None:
        # Rolling Importance 실패 시, 전체 기간 LGBM 중요도로 대체
        temp_model = lgb.LGBMRegressor(**LGBM_PARAMS) 
        temp_model.fit(X_train_df, y_train)
        feature_importances = pd.Series(temp_model.feature_importances_, index=X_train_df.columns)
        selected_features = feature_importances.nlargest(15).index.tolist()
        df_importance_summary = pd.DataFrame({
            'Feature': selected_features,
            'Mean_Importance': feature_importances.loc[selected_features].values,
            'Frequency': 1
        })
        st.warning("⚠️ Rolling Importance 실패. 전체 기간 LGBM 중요도로 대체 선택했습니다.")
        
    features = selected_features
    # 선택된 피처로 데이터셋 재구성
    X_train_df = X_train_df[features]
    X_test_df = X_test_df[features]
    
    st.info(f"✅ Rolling Importance로 **선택된 피처 수: {len(features)}개**.")
    if df_importance_summary is not None:
        st.markdown("**Top 15 피처 (빈도수 > 평균 중요도 순)**")
        st.dataframe(df_importance_summary.set_index('Feature')[['Frequency', 'Mean_Importance']].style.format({'Mean_Importance': '{:.4f}'}), width='stretch')

    # 6. 앙상블 모델 훈련 및 시계열 교차검증 (TS Split)
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
        st.dataframe(pd.DataFrame({'Fold': range(1, n_splits + 1), 'R2 Score': r2_scores_lgbm}).style.format({'R2 Score': '{:.4f}'}), width='stretch')
    st.markdown("---")

    # 최종 앙상블 모델 훈련
    voting_model, lgbm_model = train_voting_model(
        X_train_df, 
        y_train, 
        LGBM_PARAMS, 
        XGB_PARAMS, 
        RF_PARAMS
    )
        
    y_train_pred_lgbm = lgbm_model.predict(X_train_df)
    residuals = y_train - y_train_pred_lgbm
    residual_std = residuals.std()
    # 90% 신뢰구간 계수 (Z-score 1.645)
    CI_FACTOR = 1.645 * residual_std 
    
    # 테스트 기간 예측
    y_test_pred = voting_model.predict(X_test_df)
    
    # 다음 10일 예측 (마지막 거래일 데이터 사용)
    last_data_full = X_scaled_all_df.iloc[-1].to_frame().T 
    last_data_df = last_data_full[features] # 선택된 피처만 사용
    
    next_day_return_pred = voting_model.predict(last_data_df)[0]
    low_ci = next_day_return_pred - CI_FACTOR
    high_ci = next_day_return_pred + CI_FACTOR
    
    # 7. 결과 출력
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
        # VIX_Change_5D 피처는 Lag 5가 아닌 차분 5일
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

    # 8. SHAP + Permutation Importance 비교
    st.header("💡 예측 해석: SHAP + Permutation Importance")
    
    col_shap, col_perm = st.columns(2)

    # SHAP 분석
    with col_shap:
        st.subheader("1. SHAP (LightGBM)")
        st.markdown(f"모델의 **국소적 기여도** 분석 (최종 예측치 `{next_day_return_pred:+.2f}%` 기여)")
        try:
            # SHAP explainer는 LGBM 모델을 사용 (앙상블 모델은 지원하지 않음)
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
        st.markdown("테스트 데이터셋에서 **전역적 중요도** 분석")
        try:
            # Permutation Importance는 LGBM 모델을 사용
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


    # 9. 예측 vs. 실제 수익률 시각화 (앙상블 모델)
    st.subheader("📈 Soft Voting 앙상블 예측 vs. 실제 수익률 (90% 신뢰구간)")
    
    y_test_df = pd.DataFrame({
        'Actual': y_test, 'Predicted': y_test_pred,
        'Low_CI': y_test_pred - CI_FACTOR, 'High_CI': y_test_pred + CI_FACTOR
    }, index=y_test.index) # y_test의 인덱스를 사용하여 정확히 매핑

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['High_CI'], mode='lines', line=dict(width=0), showlegend=False))
    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Low_CI'], fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', mode='lines', line=dict(width=0), name='90% 신뢰구간'))
    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Actual'], mode='markers', name='실제 10일 누적 수익률', marker=dict(color='blue', size=5, opacity=0.8)))
    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Predicted'], mode='lines', name='앙상블 예측 수익률 (Median)', line=dict(color='red', width=2)))

    fig_pred.update_layout(title=f"테스트 기간 S&P 500 10일 누적 수익률 예측 결과", xaxis_title="날짜", yaxis_title="수익률(%)", hovermode="x unified", height=500)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # 10. 팩터 중요도 시각화
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
