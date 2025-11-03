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
from pytrends.request import TrendReq # 사용하지 않으므로 주석 처리
import re
import shap 
import matplotlib.pyplot as plt 
import seaborn as sns 

# ------------------------
# ✨ 상수 및 페이지 설정 (코스피 맞춤 수정)
# ------------------------
st.set_page_config(page_title="🇰🇷 코스피 매크로 추세 예측", layout="wide")
st.title("🐯 코스피 추세 예측 모델 (TS Split 시간 최적화 적용)")

st.markdown("""
**KOSPI**의 다음 날 수익률을 예측합니다. **시계열 교차검증(TS Split)** 속도 최적화를 위해 모델 간소화 및 $\text{Early Stopping}$을 적용했습니다.
""")

# ------------------------
# 0. 매크로 데이터 수집 함수 (한국 증시에 맞게 수정)
# ------------------------
# FRED 데이터는 그대로 사용 (미국 금리, 유동성 등은 여전히 한국 증시에 중요)
@st.cache_data(show_spinner="⏳ FRED 데이터 (미국 금리차, M2, BBB OAS, S&P 500 EPS) 로드 중...")
def get_fred_data():
    """FRED에서 미국 주요 경제 지표를 병렬로 가져옵니다."""
    try:
        fred_api_key = st.secrets["fred"]["FRED_API_KEY"]
    except KeyError:
        st.error("❌ FRED API 키 설정 오류: Streamlit Secrets의 'fred' 섹션과 'FRED_API_KEY' 이름을 확인해주세요.")
        st.stop()
        return {}
    TICKERS = {
        "DGS10": "10Y", "DGS2": "2Y", "BAMLC0A4CBBB": "BBB_OAS", 
        "M2SL": "M2", "GDPC1": "GDP", "SP500PE": "SP500_EPS"
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
    total_tickers = len(TICKERS)
    progress_bar = st.empty()
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
        results['US_YIELD_CURVE'] = (df_yield['10Y'] - df_yield['2Y']).rename('US_YIELD_CURVE').to_frame() # 이름 변경
    return results

@st.cache_data(show_spinner="⏳ Fear & Greed Index 로드 중...")
def get_fear_greed_index(limit=1095): 
    """Alternative.me에서 Fear & Greed Index를 가져옵니다. (글로벌 센티먼트 지표로 유지)"""
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
# 1. 팩터 및 증시 데이터 로드 (코스피 맞춤 수정)
# ------------------------

@st.cache_data(show_spinner="⏳ KOSPI, KOSDAQ, V-KOSPI, 원자재, 환율 데이터 로드 중...")
def load_market_data(start_date, end_date):
    """KOSPI, KOSDAQ, V-KOSPI, WTI, Copper, Gold, 원/달러 환율 데이터를 로드합니다."""
    load_start_date = start_date - timedelta(days=50) 
    
    # 🌟 KOSPI, KOSDAQ, V-KOSPI, 국고채, 환율, 원자재로 변경
    tickers = {
        '^KS11': 'KOSPI_Close', '^KQ11': 'KOSDAQ_Close', '^VKOSPI': 'VKOSPI',
        'KR10YT=RR': 'KR_Bond_10Y', # 한국 국고채 10년물
        'KRW=X': 'KR_FX_KRWUSD',    # 원/달러 환율
        'CL=F': 'WTI', 'GC=F': 'GOLD', 'HG=F': 'COPPER' 
    }
    
    all_data = []
    total_tickers = len(tickers)
    progress_bar = st.progress(0, text="시장 데이터 로드 중...")
    
    for i, (ticker, name) in enumerate(tickers.items()):
        try:
            progress_value = (i + 1) / total_tickers
            progress_bar.progress(progress_value, text=f"{name} ({ticker}) 로드 중...")
            
            # 🌟 원/달러 환율(KRW=X)은 YFinance에서 제공하는 Ticker이므로 fdr.DataReader로 로드
            df = fdr.DataReader(ticker, start=load_start_date, end=end_date)
            
            if 'Close' in df.columns:
                 df = df[['Close']].rename(columns={'Close': name})
            elif 'Price' in df.columns: # 채권/환율 데이터는 'Price' 컬럼일 수 있음
                 df = df[['Price']].rename(columns={'Price': name})
            else:
                 # 종가가 없는 경우 첫 번째 컬럼 사용 (예: 국고채 수익률)
                 df = df.iloc[:, 0].to_frame(name=name)

            df.index = df.index.date
            all_data.append(df)
            time.sleep(0.05)
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
# 2. 감성 분석 모델 로드 및 함수 (네이버 API)
# ------------------------
# KR-FinBert 모델과 함수는 한국어로 동일하게 사용 (수정 없음)
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

tokenizer, sentiment_model, device = load_sentiment_model()

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

def get_naver_news_api(query, display=30, start=1, sort="date"):
    """Fetches data from Naver News Search API (코스피 관련 키워드 검색)."""
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError:
        st.error("❌ 네이버 API 키가 Streamlit Secrets의 [naver] 섹션에 설정되어 있지 않습니다.")
        return pd.DataFrame()

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
    except Exception as e:
        st.error(f"❌ 네이버 API 요청 실패: {e}")
        return pd.DataFrame()

# ------------------------
# 3. 피처 엔지니어링 함수 (코스피 맞춤 수정)
# ------------------------
def create_features(df_merge):
    """모든 팩터에 대해 시계열 피처를 생성하고 데이터를 정리합니다."""
    df = df_merge.copy()
    
    # 🌟 KOSDAQ/KOSPI 비율로 변경
    if 'KOSDAQ_Close' in df.columns and 'KOSPI_Close' in df.columns:
        df['KOSDAQ_KOSPI_Ratio'] = df['KOSDAQ_Close'] / df['KOSPI_Close']
    
    # 🌟 타겟 변수 및 일일 수익률을 KOSPI 기준으로 변경
    df['Next_Day_Return'] = df['KOSPI_Close'].pct_change().shift(-1) * 100
    df['Daily_Return'] = df['KOSPI_Close'].pct_change() * 100

    lags = [1, 3, 5]
    
    # 🌟 팩터 목록을 코스피 관련 지표로 업데이트
    lag_factors = ['Daily_Return', 'VKOSPI', 'FGI', 'Sentiment_Score',  
                   'US_YIELD_CURVE', 'KR_Bond_10Y', 'BBB_OAS', 
                   'WTI', 'GOLD', 'COPPER', 
                   'KR_FX_KRWUSD', 'KOSDAQ_KOSPI_Ratio', 'SP500_EPS'] 
    
    for factor in lag_factors:
        if factor in df.columns:
            for lag in lags:
                df[f'{factor}_Lag_{lag}'] = df[factor].shift(lag)
                
    # 🌟 VIX_Change_5D를 VKOSPI_Change_5D로 변경
    if 'VKOSPI' in df.columns:
        df['VKOSPI_Change_5D'] = df['VKOSPI'].diff(5)
    
    # 🌟 SP500_SMA_20을 KOSPI_SMA_20로 변경
    df['KOSPI_SMA_20'] = df['KOSPI_Close'].rolling(window=20).mean()
    
    df = df.dropna()
    
    # 🌟 피처 목록 업데이트
    base_features = [col for col in df.columns if not col.endswith(('Return', 'Close')) and 'KOSPI_' not in col and 'KOSDAQ_' not in col]
    features = [f for f in base_features + ['KOSPI_Close'] if f in df.columns and ('Lag' in f or 'Change' in f or 'SMA' in f or f in ['GDP', 'M2', 'SP500_EPS', 'KR_FX_KRWUSD', 'KOSDAQ_KOSPI_Ratio'])]
    features = list(set(features))
    
    return df, features

# ------------------------
# 4. Streamlit 실행 로직
# ------------------------

# 앙상블 모델을 캐싱하는 함수 (재훈련 방지)
@st.cache_resource(show_spinner="🚀 Soft Voting 앙상블 모델 훈련 중/로드 중...")
def train_voting_model(_X_train_df, _y_train, _lgbm_params, _xgb_params, _rf_params):
    lgbm_model = lgb.LGBMRegressor(**_lgbm_params)
    xgb_model = xgb.XGBRegressor(**_xgb_params)
    rf_model = RandomForestRegressor(**_rf_params)
    
    voting_model = VotingRegressor(
        estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('rf', rf_model)],
        weights=[1, 1, 1] 
    )
    voting_model.fit(_X_train_df, _y_train) 
    
    lgbm_shap_model = lgb.LGBMRegressor(**_lgbm_params)
    lgbm_shap_model.fit(_X_train_df, _y_train)
    
    return voting_model, lgbm_shap_model

st.markdown("---")
# UI 입력 요소
col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    # 🌟 코스피 관련 키워드로 변경
    news_query = st.text_input("📰 뉴스 감성 분석 키워드", 
                               value="코스피 전망|반도체 전망|삼성전자 실적|SK하이닉스 실적", 
                               help="네이버 뉴스 검색에 사용될 키워드 (예: 코스피, 반도체, 삼성전자)")
with col2:
    start_date = st.date_input("분석 시작일", datetime.now() - timedelta(days=365 * 2)) 
with col3:
    end_date = st.date_input("분석 종료일", datetime.now())
    
if st.button("🚀 데이터 로드, 분석 및 예측 시작 (최적화)", type="primary", use_container_width=True):
    
    # 1. 데이터 로드
    market_df = load_market_data(start_date, end_date)
    fred_data = get_fred_data()
    fg_df = get_fear_greed_index(limit=365 * 3)
    # 🌟 Google Trends는 한국어 키워드나 'kospi'로 변경 필요하지만, 일단 제외 (PyTrends의 한계)
    
    # 1-2. 뉴스 감성 분석
    with st.spinner(f"뉴스 크롤링 및 감성 분석 중... (키워드: {news_query})"):
        all_news = get_naver_news_api(news_query, display=100)
        load_start_date = start_date - timedelta(days=50)
        filtered_news = all_news[(all_news['Date'] >= load_start_date) & (all_news['Date'] <= end_date)].copy()
        
        if not filtered_news.empty:
            filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
            st.success("✅ 뉴스 감성 분석 완료!")
        else:
            st.warning("⚠️ 지정된 기간에 해당하는 기사가 없거나, 뉴스 로드에 실패했습니다. 감성 분석을 건너뜁니다.")

    # 2. 데이터 병합
    df_merge = market_df
    if not fg_df.empty: df_merge = pd.merge(df_merge, fg_df, left_index=True, right_index=True, how='left')
    for name, df_fred in fred_data.items(): df_merge = pd.merge(df_merge, df_fred, left_index=True, right_index=True, how='left')
    # if not trends_df.empty: df_merge = pd.merge(df_merge, trends_df, left_index=True, right_index=True, how='left') # Trends 제외
    if not filtered_news.empty:
        news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().to_frame()
        df_merge = pd.merge(df_merge, news_grouped, left_index=True, right_index=True, how='left')
    
    df_merge = df_merge.fillna(method='ffill').fillna(0)
    
    # 3. 피처 엔지니어링 및 데이터 준비
    df_ml, features_full = create_features(df_merge)
    
    df_ml = df_ml.tail(500)
    df_ml = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)]

    if len(df_ml) <= 100:
        st.error("데이터가 부족합니다. 분석 기간을 늘리세요. (최소 100일 필요)")
        st.stop()
        
    X_full = df_ml[features_full]
    y = df_ml['Next_Day_Return']
    
    # 4. 피처 선택 (상관도 기반 Top 15 피처 선별)
    st.subheader("⚙️ 피처 선택 (상관도 기반 Top 15)")
    corr_to_target = X_full.corrwith(y).abs().sort_values(ascending=False)
    features = corr_to_target.head(15).index.tolist()
    st.info(f"선택된 피처 수: {len(features)}개. (전체 {len(features_full)}개 중 상위 15개)")
    X = df_ml[features]
    
    # 전체 데이터 스케일링 준비
    scaler = MinMaxScaler()
    X_scaled_all = scaler.fit_transform(X)
    X_scaled_all_df = pd.DataFrame(X_scaled_all, columns=X.columns, index=X.index)
    
    # 테스트 데이터셋 분리 (마지막 30일)
    test_size = max(30, int(0.2 * len(X_scaled_all_df)))
    X_train_df, X_test_df = X_scaled_all_df.iloc[:-test_size], X_scaled_all_df.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    # 5. 앙상블 모델 훈련 및 시계열 교차검증 (TS Split)
    
    # 5-1. 모델 파라미터 정의
    LGBM_PARAMS = {'objective': 'regression', 'metric': 'rmse', 'n_estimators': 300, 'learning_rate': 0.01, 'num_leaves': 21, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    XGB_PARAMS = {'objective': 'reg:squarederror', 'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 7, 'random_state': 42, 'n_jobs': -1}
    RF_PARAMS = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    
    # 5-2. 시계열 교차검증 (TimeSeriesSplit)
    st.header("📊 시계열 교차검증 (TimeSeriesSplit)")
    n_splits = 3 
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    r2_scores_lgbm = []
    
    with st.spinner(f"⏳ TimeSeriesSplit 교차검증 중 (폴드 {n_splits}개, n_estimators=300, Early Stopping=30 적용)..."):
        
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

    # 5-3. 최종 앙상블 모델 훈련 (캐싱 적용)
    voting_model, lgbm_model = train_voting_model(X_train_df, y_train, LGBM_PARAMS, XGB_PARAMS, RF_PARAMS)
        
    # 잔차 기반 신뢰구간 계산 및 예측
    y_train_pred_lgbm = lgbm_model.predict(X_train_df)
    residuals = y_train - y_train_pred_lgbm
    residual_std = residuals.std()
    CI_FACTOR = 1.645 * residual_std 
    
    y_test_pred = voting_model.predict(X_test_df)

    # 다음 날 예측 및 CI 계산
    last_data_scaled = X_scaled_all_df.iloc[-1].values.reshape(1, -1)
    
    # 🌟 XGBoost 예측에 필요한 DataFrame 형태 변환 (피처 이름 포함)
    last_data_df = pd.DataFrame(last_data_scaled, columns=X_scaled_all_df.columns)
    
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
        # 🌟 KOSPI 예측으로 변경
        st.metric(label="📊 다음 거래일 KOSPI 예측 수익률", value=format_pred_value(next_day_return_pred), delta=f"90% CI: {low_ci:+.2f}% ~ {high_ci:+.2f}%")
    with col_pred2:
        st.metric(label="✅ 테스트 R² (앙상블)", value=f"{r2:.2f}", help=f"MSE: {mse:.4f}. 1에 가까울수록 적합도가 높음.")
    with col_pred3:
        # 🌟 VKOSPI 지수로 변경
        if 'VKOSPI' in df_ml.columns:
            current_vix = df_ml['VKOSPI'].iloc[-1]
            vix_trend = "하락 (강세) 🟢" if df_ml['VKOSPI_Change_5D'].iloc[-1] < 0 else "상승 (약세) 🔴"
            st.metric(label="🔥 현재 V-KOSPI 지수", value=f"{current_vix:.2f}", delta=vix_trend)
        else:
            st.metric(label="🔥 V-KOSPI 지수", value="데이터 없음")

    with col_pred4:
        action = "매수/추세 추종" if next_day_return_pred > 0.3 and low_ci > -0.1 else ("매도/리스크 관리" if next_day_return_pred < -0.3 else "관망/중립")
        action_color = "#D4EDDA" if "매수" in action else ("#F8D7DA" if "매도" in action else "#FFF3CD")
        st.markdown(f"""
        <div style='padding: 10px; border-radius: 5px; text-align: center; 
            background-color: {action_color}; color: {"#155724" if "매수" in action else ("#721C24" if "매도" in action else "#856404")}; 
            font-weight: bold; margin-top: 15px;'>
            최종 투자 시그널: {action}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")

    # 7. SHAP 해석 추가 
    st.header("💡 예측 해석: SHAP (SHapley Additive exPlanations)")
    st.markdown("**SHAP**을 사용하여 모델이 최종 예측(`{:.2f}%`)을 산출하는 데 기여한 팩터의 영향력을 분석합니다. (LightGBM 모델 기준)".format(next_day_return_pred))
    
    try:
        explainer = shap.TreeExplainer(lgbm_model) 
        # 🌟 last_data_df 사용 (XGBoost 오류 방지용)
        shap_values = explainer.shap_values(last_data_df)
        
        shap_df = pd.DataFrame({
            'Feature': last_data_df.columns,
            'SHAP Value': shap_values[0],
            'Feature Value': last_data_df.iloc[0].values
        })
        
        shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
        shap_df = shap_df.sort_values('Abs SHAP', ascending=False).head(5)

        fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                           color='SHAP Value', color_continuous_scale=px.colors.diverging.RdBu,
                           title=f"최종 예측({next_day_return_pred:+.2f}%)에 기여한 Top 5 팩터",
                           hover_data={'Feature Value': True, 'SHAP Value': ':.4f'})
        fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ SHAP 해석 로드 중 오류 발생: {e}. 피처 수 축소로 인해 SHAP 계산에 문제가 생길 수 있습니다.")
    st.markdown("---")


    # 8. 피처 상관관계 히트맵 시각화 추가 (수정 없음)
    st.header("🔗 피처 상관관계 히트맵")
    st.markdown("훈련에 사용된 **축소된 Top 15 피처**와 타겟 간의 상관관계를 시각적으로 확인합니다.")

    correlation_df = df_ml[features + ['Next_Day_Return']].copy()
    
    N_TOP_FEATURES = len(features) 
    
    try:
        corr_matrix = correlation_df.corr()
        fig_heatmap = px.imshow(corr_matrix, 
                                 x=corr_matrix.columns, 
                                 y=corr_matrix.columns,
                                 color_continuous_scale='RdBu_r', 
                                 title=f'훈련에 사용된 {N_TOP_FEATURES}개 피처 간의 상관관계 히트맵')
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


    # 9. 주요 매크로 팩터 추이 시각화 (코스피 맞춤 수정)
    st.header("📊 주요 매크로 팩터 추이 (KOSPI와 비교)")
    
    df_macro_plot = df_ml[df_ml.index >= start_date].copy()

    fig_macro = go.Figure()
    
    # 🌟 KOSPI 지표 사용
    fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['KOSPI_Close'], name='KOSPI (좌측 축)', line=dict(color='#1f77b4', width=2), yaxis='y1'))
    # 🌟 미국 금리차 사용
    fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['US_YIELD_CURVE'], name='미국 장단기 금리차 (10Y-2Y)', line=dict(color='red', width=1.5), yaxis='y2', opacity=0.8))
    fig_macro.add_hline(y=0, line_dash="dash", line_color="red", yref="y2")     
    # 🌟 원/달러 환율 사용
    if 'KR_FX_KRWUSD' in df_macro_plot.columns:
        fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['KR_FX_KRWUSD'], name='원/달러 환율', line=dict(color='purple', width=1.5), yaxis='y3', opacity=0.8))
    # 🌟 국고채 10년물 사용
    if 'KR_Bond_10Y' in df_macro_plot.columns:
        fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['KR_Bond_10Y'], name='국고채 10년물 수익률', line=dict(color='green', width=1.5), yaxis='y4', opacity=0.8))
    
    fig_macro.update_layout(title="KOSPI vs. 주요 매크로 지표", xaxis_title="날짜",
        yaxis=dict(title=dict(text='KOSPI 종가', font=dict(color="#1f77b4")), domain=[0, 1]),
        yaxis2=dict(title=dict(text='미국 금리차 (%)', font=dict(color="red")), overlaying='y', side='right', position=0.90, showgrid=False),
        yaxis3=dict(title=dict(text='원/달러 환율', font=dict(color="purple")), overlaying='y', side='right', position=0.95, showgrid=False),
        yaxis4=dict(title=dict(text='국고채 수익률', font=dict(color="green")), overlaying='y', side='right', position=1.0, showgrid=False),
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
    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Actual'], mode='markers', name='실제 수익률', marker=dict(color='blue', size=5, opacity=0.8)))
    fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Predicted'], mode='lines', name='앙상블 예측 수익률 (Median)', line=dict(color='red', width=2)))

    # 🌟 KOSPI 예측 결과로 변경
    fig_pred.update_layout(title=f"테스트 기간 KOSPI 수익률 예측 결과", xaxis_title="날짜", yaxis_title="수익률(%)", hovermode="x unified", height=500)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # 11. 팩터 중요도 시각화 (수정 없음)
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
