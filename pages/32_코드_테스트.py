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
import urllib.parse
from json.decoder import JSONDecodeError
import FinanceDataReader as fdr
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import MinMaxScaler
import time
from concurrent.futures import ThreadPoolExecutor
from pytrends.request import TrendReq
import re

# ------------------------
# ✨ 상수 및 페이지 설정
# ------------------------
st.set_page_config(page_title="🇺🇸 미국 증시 매크로 추세 예측", layout="wide")
st.title("🦅 미국 증시 추세 예측 모델 (통합 팩터 & 앙상블)")

st.markdown("""
**S&P 500**의 다음 날 수익률을 **VIX, 금리차, M2, 회사채 스프레드, F&G, 뉴스 감성** 및 **DXY, NASDAQ/S&P 비율** 등
다양한 매크로 및 시장 팩터를 통합하여 **Soft Voting 앙상블 모델**로 예측합니다.
""")

# ------------------------
# 0. 매크로 데이터 수집 함수
# ------------------------
@st.cache_data(show_spinner="⏳ FRED 데이터 (금리차, M2, BBB OAS, SP500 EPS) 로드 중...")
def get_fred_data():
    """FRED에서 여러 경제 지표를 병렬로 가져옵니다."""
    try:
        # 요청하신 대로 st.secrets["fred"]["FRED_API_KEY"]로 수정
        fred_api_key = st.secrets["fred"]["FRED_API_KEY"]
    except KeyError:
        st.error("❌ FRED API 키 설정 오류: Streamlit Secrets의 'fred' 섹션과 'FRED_API_KEY' 이름을 확인해주세요.")
        st.stop()
        return {}

    # FRED Tickers: SP500 EPS 추정치 추가 (SP500PE)
    TICKERS = {
        "DGS10": "10Y", "DGS2": "2Y", 
        "BAMLC0A4CBBB": "BBB_OAS", "M2SL": "M2", "GDPC1": "GDP",
        "SP500PE": "SP500_EPS" # S&P 500 EPS 추정치 추가
    }
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def fetch_single_fred(ticker, observation_start):
        params = {
            "series_id": ticker,
            "api_key": fred_api_key,
            "file_type": "json",
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
            ticker_name = futures[future]
            try:
                ticker, df = future.result()
                if not df.empty:
                    results[TICKERS[ticker]] = df
            except Exception as e:
                pass 
                
            loaded_count += 1
            progress_value = loaded_count / total_tickers
            progress_bar.progress(progress_value, text=f"FRED 지표 로드 중... ({loaded_count}/{total_tickers})")
    
    progress_bar.empty()

    # 장단기 금리차 계산 (10Y - 2Y)
    if '10Y' in results and '2Y' in results:
        df_yield = pd.merge(results['10Y'], results['2Y'], left_index=True, right_index=True, how='inner')
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


@st.cache_data(show_spinner="⏳ Google Trends 데이터 로드 중...")
def get_google_trends(keywords, start_date, end_date):
    """Google Trends에서 검색량을 가져옵니다."""
    try:
        pytrends = TrendReq(hl='en-US', tz=360) 
        timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='')
        
        # Rate Limiting 방지를 위해 10초 지연 유지
        time.sleep(10) 
        
        df = pytrends.interest_over_time()
        
        if df.empty or 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'], errors='ignore')
            
        df.index = df.index.date
        df.index.name = 'Date'
        df = df.rename(columns={col: f'Trend_{col}' for col in df.columns})
        return df
    except Exception as e:
        st.warning(f"⚠️ Google Trends 데이터 로드 오류: {e}. PyTrends 설치 상태 확인 필요.")
        return pd.DataFrame()


# ------------------------
# 1. 팩터 및 증시 데이터 로드 (DXY, NASDAQ 추가)
# ------------------------

@st.cache_data(show_spinner="⏳ 주가, 원자재, DXY, NASDAQ 데이터 로드 중...")
def load_market_data(start_date, end_date):
    """S&P 500, NASDAQ, VIX, WTI, Copper, Gold, DXY 데이터를 로드합니다."""
    load_start_date = start_date - timedelta(days=50) 
    
    # DXY와 NASDAQ(^IXIC) 추가
    tickers = {
        '^GSPC': 'SP500_Close', '^IXIC': 'NASDAQ_Close', '^VIX': 'VIX', 
        'CL=F': 'WTI', 'GC=F': 'GOLD', 'HG=F': 'COPPER', 
        'DX-Y.NYB': 'DXY' # USD Index 추가 (Yahoo Finance Ticker)
    }
    
    all_data = []
    total_tickers = len(tickers)
    
    progress_bar = st.progress(0, text="시장 데이터 로드 중...")
    
    for i, (ticker, name) in enumerate(tickers.items()):
        try:
            progress_value = (i + 1) / total_tickers
            progress_bar.progress(progress_value, text=f"{name} ({ticker}) 로드 중...")
            
            df = fdr.DataReader(ticker, start=load_start_date, end=end_date)
            df = df[['Close']].rename(columns={'Close': name})
            df.index = df.index.date
            all_data.append(df)
            time.sleep(0.05)
        except Exception as e:
            st.warning(f"⚠️ {name} ({ticker}) 데이터 로드 실패: {e}")
            continue

    progress_bar.empty()
    st.success("✅ 시장 데이터 로드 완료!")
        
    if not all_data:
        return pd.DataFrame()
        
    df_merged = pd.concat(all_data, axis=1, join='outer').sort_index()
    df_merged.index.name = 'Date'
    return df_merged

# ------------------------
# 2. 감성 분석 모델 로드 및 함수 
# ------------------------
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
    """Fetches data from Naver News Search API (미국 증시 관련 키워드 검색)."""
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
            title = re.sub('<[^<]+?>', '', item.get('title', '')) # HTML 태그 제거
            pub_date = item.get('pubDate', '')
            try: pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception: pub_date_dt = None
            news_data.append({'Date': pub_date_dt, 'Title': title})
        return pd.DataFrame(news_data)
    except Exception as e:
        st.error(f"❌ 네이버 API 요청 실패: {e}")
        return pd.DataFrame()

# ------------------------
# 3. 피처 엔지니어링 함수
# ------------------------
def create_features(df_merge):
    """모든 팩터에 대해 시계열 피처를 생성하고 데이터를 정리합니다."""
    df = df_merge.copy()
    
    # 0. NASDAQ/S&P 비율 피처 추가
    if 'NASDAQ_Close' in df.columns and 'SP500_Close' in df.columns:
        df['NASDAQ_SP500_Ratio'] = df['NASDAQ_Close'] / df['SP500_Close']
    
    # 1. 타겟 변수: S&P 500 다음 날의 수익률 (%)
    df['Next_Day_Return'] = df['SP500_Close'].pct_change().shift(-1) * 100
    df['Daily_Return'] = df['SP500_Close'].pct_change() * 100

    # 2. 시계열 지연(Lag) 피처 생성
    lags = [1, 3, 5]
    
    # Lag 피처를 생성할 팩터 목록 (새로운 팩터 포함)
    lag_factors = ['Daily_Return', 'VIX', 'FGI', 'Sentiment_Score', 
                   'YIELD_CURVE', 'BBB_OAS', 'WTI', 'GOLD', 'COPPER',
                   'DXY', 'NASDAQ_SP500_Ratio', 'SP500_EPS'] 
    
    for factor in lag_factors:
        if factor in df.columns:
            for lag in lags:
                df[f'{factor}_Lag_{lag}'] = df[factor].shift(lag)
                
    # 3. 변화율 (Rate of Change) 및 기술적 지표 추가
    df['VIX_Change_5D'] = df['VIX'].diff(5)
    df['SP500_SMA_20'] = df['SP500_Close'].rolling(window=20).mean()
    
    df = df.dropna()
    
    # 최종 피처 목록
    base_features = [col for col in df.columns if not col.endswith(('Return', 'Close')) and 'SP500_' not in col and 'NASDAQ_' not in col]
    features = [f for f in base_features + ['SP500_Close'] if f in df.columns and ('Lag' in f or 'Change' in f or 'SMA' in f or f in ['GDP', 'M2', 'SP500_EPS', 'DXY', 'NASDAQ_SP500_Ratio'])]
    features = list(set(features)) # 중복 제거
    
    return df, features

# ------------------------
# 4. Streamlit 실행 로직
# ------------------------

st.markdown("---")
# UI 입력 요소
col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    news_query = st.text_input("📰 뉴스 감성 분석 키워드", value="미국 증시 전망", help="네이버 뉴스 검색에 사용될 키워드 (예: S&P 500, 미국 주식, 연준)")
with col2:
    start_date = st.date_input("분석 시작일", datetime.now() - timedelta(days=365))
with col3:
    end_date = st.date_input("분석 종료일", datetime.now())
    
if st.button("🚀 데이터 로드, 분석 및 예측 시작", type="primary", use_container_width=True):
    
    # 1. 데이터 로드
    market_df = load_market_data(start_date, end_date)
    fred_data = get_fred_data()
    fg_df = get_fear_greed_index(limit=365 * 3)
    
    # 1-1. Google Trends 로드
    trends_keywords = ["S&P 500", "Recession"]
    trends_df = get_google_trends(trends_keywords, start_date, end_date)
    
    # 1-2. 뉴스 감성 분석
    with st.spinner(f"뉴스 크롤링 및 감성 분석 중... (키워드: {news_query})"):
        all_news = get_naver_news_api(news_query, display=100)
        
        load_start_date = start_date - timedelta(days=50)
        filtered_news = all_news[(all_news['Date'] >= load_start_date) & (all_news['Date'] <= end_date)].copy()
        
        if filtered_news.empty:
            st.warning("⚠️ 뉴스 데이터를 충분히 가져오지 못했습니다. 감성 점수는 0으로 처리될 수 있습니다.")
        else:
            filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
            st.success("✅ 뉴스 감성 분석 완료!")
            
            st.subheader("📰 분석에 사용된 뉴스 기사 및 감성 점수")
            st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values('Date', ascending=False).head(200), 
                         use_container_width=True,
                         column_config={
                             "Sentiment_Score": st.column_config.NumberColumn("감성 점수", format="%.4f", help="-1.0 (부정) ~ 1.0 (긍정)")
                         })
            st.markdown("---")

    # 2. 데이터 병합 (날짜 기준으로 Outer Join 후 FFILL)
    df_merge = market_df
    
    if not fg_df.empty:
        df_merge = pd.merge(df_merge, fg_df, left_index=True, right_index=True, how='left')
    
    for name, df_fred in fred_data.items():
        df_merge = pd.merge(df_merge, df_fred, left_index=True, right_index=True, how='left')
        
    if not trends_df.empty:
        df_merge = pd.merge(df_merge, trends_df, left_index=True, right_index=True, how='left')

    if not filtered_news.empty:
        news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().to_frame()
        df_merge = pd.merge(df_merge, news_grouped, left_index=True, right_index=True, how='left')
    
    df_merge = df_merge.fillna(method='ffill').fillna(0)
    
    # 3. 피처 엔지니어링 및 데이터 준비
    df_ml, features = create_features(df_merge)
    df_ml = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)]

    if len(df_ml) <= 50:
        st.error("데이터가 부족합니다. 분석 기간을 늘리거나, 데이터 로드 오류를 확인하세요.")
        st.stop()
        
    X = df_ml[features].values
    y = df_ml['Next_Day_Return'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    test_size = max(30, int(0.2 * len(X_scaled))) # 최소 30일 테스트
    X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    # 4. 앙상블 모델 훈련 (Soft Voting)
    LGBM_TUNED_PARAMS = {
        'objective': 'regression', 'metric': 'rmse', 'n_estimators': 700, 
        'learning_rate': 0.01, 'num_leaves': 21, 'max_depth': 7,
        'colsample_bytree': 0.8, 'subsample': 0.8, 'random_state': 42, 
        'n_jobs': -1, 'verbose': -1
    }
    XGB_TUNED_PARAMS = {
        'objective': 'reg:squarederror', 'n_estimators': 700, 'learning_rate': 0.01,
        'max_depth': 7, 'colsample_bytree': 0.8, 'subsample': 0.8,
        'random_state': 42, 'n_jobs': -1
    }
    RF_TUNED_PARAMS = {
        'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1
    }
    
    with st.spinner("🚀 Soft Voting 앙상블 모델 훈련 중 (LGBM, XGBoost, RF)..."):
        # 개별 모델 정의
        lgbm_model = lgb.LGBMRegressor(**LGBM_TUNED_PARAMS)
        xgb_model = xgb.XGBRegressor(**XGB_TUNED_PARAMS)
        rf_model = RandomForestRegressor(**RF_TUNED_PARAMS)

        # Soft Voting 앙상블 모델 정의
        voting_model = VotingRegressor(
            estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('rf', rf_model)],
            weights=[1, 1, 1] 
        )
        
        # 모델 훈련 (VotingRegressor는 fit 시 개별 모델 모두 훈련)
        voting_model.fit(X_train, y_train) 
        
    # 잔차 기반 신뢰구간 계산 및 예측 (LGBM 모델의 잔차 사용)
    lgbm_model.fit(X_train, y_train,
                   eval_set=[(X_test, y_test)],
                   callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])

    y_train_pred_lgbm = lgbm_model.predict(X_train) # 잔차 계산용
    residuals = y_train - y_train_pred_lgbm
    residual_std = residuals.std()
    CI_FACTOR = 1.645 * residual_std 
    
    # 앙상블 모델 예측
    y_test_pred = voting_model.predict(X_test)

    # 다음 날 예측
    last_data = df_ml[features].iloc[-1].values.reshape(1, -1)
    last_data_scaled = scaler.transform(last_data)
    next_day_return_pred = voting_model.predict(last_data_scaled)[0]
    low_ci = next_day_return_pred - CI_FACTOR
    high_ci = next_day_return_pred + CI_FACTOR
    
    # 5. 결과 시각화 및 출력
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    st.markdown("---")
    st.header("📈 최종 예측 결과 및 모델 성능")
    
    # --- A. 예측 결과 카드형 출력 ---
    col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

    def format_pred_value(value): return f"{value:+.2f}%"

    with col_pred1:
        st.metric(label="📊 다음 거래일 S&P 500 예측 수익률", 
                  value=format_pred_value(next_day_return_pred), 
                  delta=f"90% CI: {low_ci:+.2f}% ~ {high_ci:+.2f}%")

    with col_pred2:
        st.metric(label="✅ 테스트 R² (예측 신뢰도)", 
                  value=f"{r2:.2f}", 
                  help=f"MSE: {mse:.4f}. 1에 가까울수록 모델의 적합도가 높음. (Soft Voting 결과)")
        
    with col_pred3:
        current_vix = df_ml['VIX'].iloc[-1]
        vix_trend = "하락 (강세) 🟢" if df_ml['VIX_Change_5D'].iloc[-1] < 0 else "상승 (약세) 🔴"
        st.metric(label="🔥 현재 VIX 지수", 
                  value=f"{current_vix:.2f}", 
                  delta=vix_trend)

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
    
    # --- B. 주요 매크로 팩터 추이 시각화 (DXY 추가) ---
    st.subheader("📊 주요 매크로 팩터 추이 (S&P 500과 비교)")
    
    df_macro_plot = df_ml[df_ml.index >= start_date].copy()

    fig_macro = go.Figure()
    
    # S&P 500 (1차 축)
    fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['SP500_Close'], name='S&P 500 (좌측 축)', line=dict(color='#1f77b4', width=2), yaxis='y1'))

    # YIELD_CURVE (2차 축)
    fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['YIELD_CURVE'], name='장단기 금리차 (10Y-2Y)', line=dict(color='red', width=1.5), yaxis='y2', opacity=0.8))
    fig_macro.add_hline(y=0, line_dash="dash", line_color="red", yref="y2")     
    
    # BBB_OAS (3차 축)
    fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['BBB_OAS'], name='BBB 회사채 스프레드', line=dict(color='green', width=1.5), yaxis='y3', opacity=0.8))

    # DXY (4차 축)
    if 'DXY' in df_macro_plot.columns:
         fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['DXY'], name='USD Index (DXY)', line=dict(color='purple', width=1.5), yaxis='y4', opacity=0.8))


    fig_macro.update_layout(title="S&P 500 vs. 경기/신용 리스크 지표", xaxis_title="날짜",
        yaxis=dict(title=dict(text='S&P 500 종가', font=dict(color="#1f77b4")), domain=[0, 1]),
        yaxis2=dict(title=dict(text='금리차 (%)', font=dict(color="red")), overlaying='y', side='right', position=0.90, showgrid=False),
        yaxis3=dict(title=dict(text='BBB OAS', font=dict(color="green")), overlaying='y', side='right', position=0.95, showgrid=False),
        yaxis4=dict(title=dict(text='DXY', font=dict(color="purple")), overlaying='y', side='right', position=1.0, showgrid=False),
        hovermode="x unified", height=600, legend=dict(x=0, y=1.05, orientation="h"))
    
    st.plotly_chart(fig_macro, use_container_width=True)


    # --- C. 예측 vs. 실제 수익률 시각화 (앙상블 모델) ---
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

    fig_pred.update_layout(title=f"테스트 기간 S&P 500 수익률 예측 결과", xaxis_title="날짜", yaxis_title="수익률(%)", hovermode="x unified", height=500)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # --- D. 팩터 중요도 시각화 (LGBM 모델의 중요도 사용) ---
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
# import urllib.parse
# from json.decoder import JSONDecodeError
# import FinanceDataReader as fdr
# import lightgbm as lgb
# from sklearn.preprocessing import MinMaxScaler
# import time
# from concurrent.futures import ThreadPoolExecutor
# from pytrends.request import TrendReq
# import re 

# # ------------------------
# # ✨ 상수 및 페이지 설정
# # ------------------------
# st.set_page_config(page_title="🇺🇸 미국 증시 매크로 추세 예측", layout="wide")
# st.title("🦅 미국 증시 추세 예측 모델 (통합 팩터)")

# st.markdown("""
# **S&P 500**의 다음 날 수익률을 **VIX, 금리차, M2, 회사채 스프레드, F&G, 뉴스 감성** 등
# 다양한 매크로 및 시장 팩터를 통합하여 LightGBM으로 예측합니다.
# """)

# # ------------------------
# # 0. 매크로 데이터 수집 함수
# # ------------------------
# # ------------------------
# # 0. 매크로 데이터 수집 함수 (수정된 부분)
# # ------------------------

# @st.cache_data(show_spinner="⏳ FRED 데이터 (금리차, M2, BBB OAS) 로드 중...")
# def get_fred_data():
#     """FRED에서 여러 경제 지표를 병렬로 가져옵니다."""
#     try:
#         # FRED API 키 참조 방식은 이미 수정되었다고 가정 (예: st.secrets["fred"]["FRED_API_KEY"])
#         fred_api_key = st.secrets["fred"]["FRED_API_KEY"] 
#     except KeyError:
#         st.error("❌ FRED API 키 설정 오류: Streamlit Secrets의 'FRED' 섹션과 'FRED_API_KEY' 이름을 확인해주세요.")
#         st.stop()
#         return {}

#     # FRED Tickers:
#     TICKERS = {
#         "DGS10": "10Y", "DGS2": "2Y", 
#         "BAMLC0A4CBBB": "BBB_OAS", "M2SL": "M2", "GDPC1": "GDP"
#     }
    
#     BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
#     def fetch_single_fred(ticker, observation_start):
#         # ... (fetch_single_fred 함수 내용은 동일) ...
#         # 
#         params = {
#             "series_id": ticker,
#             "api_key": fred_api_key,
#             "file_type": "json",
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
    
#     # 1. 진행률 바 생성
#     progress_bar = st.empty()
    
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         # 2. Future 딕셔너리 생성
#         futures = {executor.submit(fetch_single_fred, ticker, start_date): ticker for ticker in TICKERS.keys()}
        
#         loaded_count = 0
        
#         # 3. Future 객체에서 결과를 하나씩 추출하며 진행률 업데이트
#         for future in futures: # 딕셔너리가 아닌 Future 객체의 리스트를 순회
#             ticker_name = futures[future] # 딕셔너리에서 키 이름 가져오기
#             try:
#                 ticker, df = future.result()
#                 if not df.empty:
#                     results[TICKERS[ticker]] = df
#             except Exception as e:
#                 # fetch_single_fred에서 이미 warning을 띄웠으므로 여기서는 pass
#                 pass 
                
#             loaded_count += 1
#             progress_value = loaded_count / total_tickers
#             progress_bar.progress(progress_value, text=f"FRED 지표 로드 중... ({loaded_count}/{total_tickers})")
    
#     # 4. 로드가 완료되면 진행률 바 제거
#     progress_bar.empty()

#     # 장단기 금리차 계산 (10Y - 2Y)
#     if '10Y' in results and '2Y' in results:
#         df_yield = pd.merge(results['10Y'], results['2Y'], left_index=True, right_index=True, how='inner')
#         results['YIELD_CURVE'] = (df_yield['10Y'] - df_yield['2Y']).rename('YIELD_CURVE').to_frame()

#     return results
# @st.cache_data(show_spinner="⏳ Fear & Greed Index 로드 중...")
# def get_fear_greed_index(limit=1095): 
#     """Alternative.me에서 Fear & Greed Index를 가져옵니다."""
#     # 이 API는 별도 키 필요 없음
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


# @st.cache_data(show_spinner="⏳ Google Trends 데이터 로드 중...")
# def get_google_trends(keywords, start_date, end_date):
#     """Google Trends에서 검색량을 가져옵니다."""
#     try:
#         pytrends = TrendReq(hl='en-US', tz=360) 
#         timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        
#         # 1. 페이로드(요청 내용) 구성
#         pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='')
        
#         # 2. ⚠️ 요청 후 5초 지연 추가 (Rate Limiting 방지)
#         time.sleep(10) 
        
#         # 3. 데이터 로드 (실제 서버 통신 발생)
#         df = pytrends.interest_over_time()
        
#         if df.empty or 'isPartial' in df.columns:
#             df = df.drop(columns=['isPartial'], errors='ignore')
            
#         df.index = df.index.date
#         df.index.name = 'Date'
#         df = df.rename(columns={col: f'Trend_{col}' for col in df.columns})
#         return df
#     except Exception as e:
#         # Code 429 오류가 발생하면 이 부분이 실행됩니다.
#         st.warning(f"⚠️ Google Trends 데이터 로드 오류: {e}. PyTrends 설치 상태 확인 필요.")
#         return pd.DataFrame()

# # ------------------------
# # 1. 팩터 및 증시 데이터 로드
# # ------------------------

# @st.cache_data(show_spinner="⏳ 주가 및 원자재 데이터 로드 중...")
# def load_market_data(start_date, end_date):
#     """S&P 500, VIX, WTI, Copper, Gold 데이터를 로드합니다."""
#     # 시계열 피처 생성을 위해 검색 기간보다 30일 정도 더 많은 데이터를 로드
#     load_start_date = start_date - timedelta(days=50) 
    
#     tickers = {
#         '^GSPC': 'SP500_Close', '^VIX': 'VIX', 'CL=F': 'WTI', 
#         'GC=F': 'GOLD', 'HG=F': 'COPPER'
#     }
    
#     all_data = []
#     total_tickers = len(tickers)
    
#     # 1. 진행률 바 위젯 생성
#     progress_bar = st.progress(0, text="시장 데이터 로드 중...")
    
#     # 2. 루프를 돌면서 데이터 로드 및 진행률 업데이트
#     for i, (ticker, name) in enumerate(tickers.items()):
#         try:
#             # 진행률 업데이트: (현재 인덱 + 1) / 전체 개수
#             progress_value = (i + 1) / total_tickers
#             progress_bar.progress(progress_value, text=f"{name} ({ticker}) 로드 중...")
            
#             df = fdr.DataReader(ticker, start=load_start_date, end=end_date)
#             df = df[['Close']].rename(columns={'Close': name})
#             df.index = df.index.date
#             all_data.append(df)
#             time.sleep(0.05)
#         except Exception as e:
#             st.warning(f"⚠️ {name} ({ticker}) 데이터 로드 실패: {e}")
#             continue

#     # 3. 로드가 완료된 후 진행률 바 제거 또는 완료 표시
#     progress_bar.empty()
#     st.success("✅ 시장 데이터 로드 완료!")
        
#     if not all_data:
#         return pd.DataFrame()
        
#     df_merged = pd.concat(all_data, axis=1, join='outer').sort_index()
#     df_merged.index.name = 'Date'
#     return df_merged

# # ------------------------
# # 2. 감성 분석 모델 로드 및 함수 (API 키 적용)
# # ------------------------
# @st.cache_resource
# def load_sentiment_model():
#     """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
#     # 💡 HF_TOKEN 적용
#     hf_token = st.secrets.get("HF_TOKEN")
#     model_name = "snunlp/KR-FinBert-SC"
    
#     try:
#         # 모델 로드 시 token 매개변수 사용 (Secrets에 저장된 토큰 사용)
#         tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
#         model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         return tokenizer, model, device
#     except Exception as e:
#         st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
#         st.info("Hugging Face 토큰 설정 또는 라이브러리 버전을 확인해주세요.")
#         st.stop()
#         return None, None, None

# tokenizer, sentiment_model, device = load_sentiment_model()

# def analyze_sentiment(text):
#     """Calculates sentiment score for the given text."""
#     if not text: return 0.0
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad(): outputs = sentiment_model(**inputs)
#     probabilities = torch.softmax(outputs.logits, dim=1)[0]
#     neg_idx, pos_idx = None, None
#     for idx, label in sentiment_model.config.id2label.items():
#         if 'negative' in label.lower() or '부정' in label: neg_idx = idx
#         elif 'positive' in label.lower() or '긍정' in label: pos_idx = idx
#     negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
#     positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0
#     return positive_score - negative_score

# def get_naver_news_api(query, display=30, start=1, sort="date"):
#     """Fetches data from Naver News Search API (미국 증시 관련 키워드 검색)."""
#     # 💡 네이버 API 키 적용
#     try:
#         client_id = st.secrets["naver"]["client_id"]
#         client_secret = st.secrets["naver"]["client_secret"]
#     except KeyError:
#         st.error("❌ 네이버 API 키가 Streamlit Secrets의 [naver] 섹션에 설정되어 있지 않습니다.")
#         return pd.DataFrame()

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
#             title = re.sub('<[^<]+?>', '', item.get('title', '')) # HTML 태그 제거
#             pub_date = item.get('pubDate', '')
#             try: pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
#             except Exception: pub_date_dt = None
#             news_data.append({'Date': pub_date_dt, 'Title': title})
#         return pd.DataFrame(news_data)
#     except Exception as e:
#         st.error(f"❌ 네이버 API 요청 실패: {e}")
#         return pd.DataFrame()

# # ------------------------
# # 3. 피처 엔지니어링 함수
# # ------------------------
# def create_features(df_merge):
#     """모든 팩터에 대해 시계열 피처를 생성하고 데이터를 정리합니다."""
#     df = df_merge.copy()
    
#     # 1. 타겟 변수: S&P 500 다음 날의 수익률 (%)
#     df['Next_Day_Return'] = df['SP500_Close'].pct_change().shift(-1) * 100
#     df['Daily_Return'] = df['SP500_Close'].pct_change() * 100

#     # 2. 시계열 지연(Lag) 피처 생성
#     lags = [1, 3, 5]
    
#     # Lag 피처를 생성할 팩터 목록
#     lag_factors = ['Daily_Return', 'VIX', 'FGI', 'Sentiment_Score', 
#                    'YIELD_CURVE', 'BBB_OAS', 'WTI', 'GOLD', 'COPPER']
    
#     for factor in lag_factors:
#         if factor in df.columns:
#             for lag in lags:
#                 df[f'{factor}_Lag_{lag}'] = df[factor].shift(lag)
                
#     # 3. 변화율 (Rate of Change) 및 기술적 지표 추가
#     df['VIX_Change_5D'] = df['VIX'].diff(5)
#     df['SP500_SMA_20'] = df['SP500_Close'].rolling(window=20).mean()
    
#     df = df.dropna()
    
#     # 최종 피처 목록
#     base_features = [col for col in df.columns if not col.endswith(('Return', 'Close')) and 'SP500_' not in col]
#     features = [f for f in base_features + ['SP500_Close'] if f in df.columns and ('Lag' in f or 'Change' in f or 'SMA' in f or f in ['GDP', 'M2'])]
#     features = list(set(features)) # 중복 제거
    
#     return df, features

# # ------------------------
# # 4. Streamlit 실행 로직
# # ------------------------

# st.markdown("---")
# # UI 입력 요소
# col1, col2, col3 = st.columns([1.5, 1, 1])
# with col1:
#     news_query = st.text_input("📰 뉴스 감성 분석 키워드", value="미국 증시 전망", help="네이버 뉴스 검색에 사용될 키워드 (예: S&P 500, 미국 주식, 연준)")
# with col2:
#     start_date = st.date_input("분석 시작일", datetime.now() - timedelta(days=365))
# with col3:
#     end_date = st.date_input("분석 종료일", datetime.now())
    
# if st.button("🚀 데이터 로드, 분석 및 예측 시작", type="primary", use_container_width=True):
    
#     # 1. 데이터 로드
#     market_df = load_market_data(start_date, end_date)
#     fred_data = get_fred_data()
#     fg_df = get_fear_greed_index(limit=365 * 3)
    
#     # 1-1. Google Trends 로드
#     trends_keywords = ["S&P 500", "Recession"]
#     trends_df = get_google_trends(trends_keywords, start_date, end_date)
    
#     # 1-2. 뉴스 감성 분석
#     with st.spinner(f"뉴스 크롤링 및 감성 분석 중... (키워드: {news_query})"):
#         all_news = get_naver_news_api(news_query, display=100)
        
#         load_start_date = start_date - timedelta(days=50)
#         filtered_news = all_news[(all_news['Date'] >= load_start_date) & (all_news['Date'] <= end_date)].copy()
        
#         if filtered_news.empty:
#             st.warning("⚠️ 뉴스 데이터를 충분히 가져오지 못했습니다. 감성 점수는 0으로 처리될 수 있습니다.")
#         else:
#             filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
#             st.success("✅ 뉴스 감성 분석 완료!")
            
#             # 뉴스 기사 및 점수 출력
#             st.subheader("📰 분석에 사용된 뉴스 기사 및 감성 점수")
#             st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values('Date', ascending=False).head(200), 
#                          use_container_width=True,
#                          column_config={
#                              "Sentiment_Score": st.column_config.NumberColumn("감성 점수", format="%.4f", help="-1.0 (부정) ~ 1.0 (긍정)")
#                          })
#             st.markdown("---")

#     # 2. 데이터 병합 (날짜 기준으로 Outer Join 후 FFILL)
#     df_merge = market_df
    
#     # F&G Index 병합
#     if not fg_df.empty:
#         df_merge = pd.merge(df_merge, fg_df, left_index=True, right_index=True, how='left')
    
#     # FRED 지표 병합
#     for name, df_fred in fred_data.items():
#         df_merge = pd.merge(df_merge, df_fred, left_index=True, right_index=True, how='left')
        
#     # Google Trends 병합
#     if not trends_df.empty:
#         df_merge = pd.merge(df_merge, trends_df, left_index=True, right_index=True, how='left')

#     # 뉴스 감성 점수 평균 병합
#     if not filtered_news.empty:
#         news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().to_frame()
#         df_merge = pd.merge(df_merge, news_grouped, left_index=True, right_index=True, how='left')
    
#     # 결측치 처리: 대부분의 매크로 지표는 FFILL (Forward Fill) 후 0으로 채우기
#     df_merge = df_merge.fillna(method='ffill').fillna(0)
    
#     # 3. 피처 엔지니어링 및 데이터 준비
#     df_ml, features = create_features(df_merge)
#     df_ml = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)]

#     if len(df_ml) <= 50:
#         st.error("데이터가 부족합니다. 분석 기간을 늘리거나, 데이터 로드 오류를 확인하세요.")
#         st.stop()
        
#     X = df_ml[features].values
#     y = df_ml['Next_Day_Return'].values
    
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     test_size = max(30, int(0.2 * len(X_scaled))) # 최소 30일 테스트
#     X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
#     y_train, y_test = y[:-test_size], y[-test_size:]
    
#     # 4. LightGBM 모델 훈련
#     LGBM_TUNED_PARAMS = {
#         'objective': 'regression', 'metric': 'rmse',
#         'n_estimators': 700, 'learning_rate': 0.01, 
#         'num_leaves': 21, 'max_depth': 7,
#         'colsample_bytree': 0.8, 'subsample': 0.8,
#         'random_state': 42, 'n_jobs': -1, 'verbose': -1
#     }
    
#     with st.spinner("🚀 LightGBM 모델 훈련 중 (통합 팩터 기반)..."):
#         lgbm_model = lgb.LGBMRegressor(**LGBM_TUNED_PARAMS)
#         lgbm_model.fit(X_train, y_train,
#                         eval_set=[(X_test, y_test)],
#                         callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])

#     # 잔차 기반 신뢰구간 계산 및 예측
#     y_train_pred = lgbm_model.predict(X_train)
#     residuals = y_train - y_train_pred
#     residual_std = residuals.std()
#     CI_FACTOR = 1.645 * residual_std 
#     y_test_pred = lgbm_model.predict(X_test)

#     # 다음 날 예측
#     last_data = df_ml[features].iloc[-1].values.reshape(1, -1)
#     last_data_scaled = scaler.transform(last_data)
#     next_day_return_pred = lgbm_model.predict(last_data_scaled)[0]
#     low_ci = next_day_return_pred - CI_FACTOR
#     high_ci = next_day_return_pred + CI_FACTOR
    
#     # 5. 결과 시각화 및 출력
#     mse = mean_squared_error(y_test, y_test_pred)
#     r2 = r2_score(y_test, y_test_pred)

#     st.markdown("---")
#     st.header("📈 최종 예측 결과 및 모델 성능")
    
#     # --- A. 예측 결과 카드형 출력 ---
#     col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

#     def format_pred_value(value): return f"{value:+.2f}%"

#     with col_pred1:
#         st.metric(label="📊 다음 거래일 S&P 500 예측 수익률", 
#                   value=format_pred_value(next_day_return_pred), 
#                   delta=f"90% CI: {low_ci:+.2f}% ~ {high_ci:+.2f}%")

#     with col_pred2:
#         st.metric(label="✅ 테스트 R² (예측 신뢰도)", 
#                   value=f"{r2:.2f}", 
#                   help=f"MSE: {mse:.4f}. 1에 가까울수록 모델의 적합도가 높음.")
        
#     with col_pred3:
#         current_vix = df_ml['VIX'].iloc[-1]
#         vix_trend = "하락 (강세) 🟢" if df_ml['VIX_Change_5D'].iloc[-1] < 0 else "상승 (약세) 🔴"
#         st.metric(label="🔥 현재 VIX 지수", 
#                   value=f"{current_vix:.2f}", 
#                   delta=vix_trend)

#     with col_pred4:
#         action = "매수/추세 추종" if next_day_return_pred > 0.3 and low_ci > -0.1 else ("매도/리스크 관리" if next_day_return_pred < -0.3 else "관망/중립")
#         action_color = "#D4EDDA" if "매수" in action else ("#F8D7DA" if "매도" in action else "#FFF3CD")
#         st.markdown(f"""
#         <div style='padding: 10px; border-radius: 5px; text-align: center; 
#             background-color: {action_color}; color: {"#155724" if "매수" in action else ("#721C24" if "매도" in action else "#856404")}; 
#             font-weight: bold; margin-top: 15px;'>
#             최종 투자 시그널: {action}
#         </div>
#         """, unsafe_allow_html=True)
        
#     st.markdown("---")
    
#     # --- B. 주요 매크로 팩터 추이 시각화 ---
#     st.subheader("📊 주요 매크로 팩터 추이 (S&P 500과 비교)")
    
#     df_macro_plot = df_ml[df_ml.index >= start_date].copy()

#     # YIELD_CURVE (장단기 금리차) 및 BBB_OAS 시각화
#     fig_macro = go.Figure()
    
#     # S&P 500 (1차 축)
#     fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['SP500_Close'], name='S&P 500 (좌측 축)', line=dict(color='#1f77b4', width=2), yaxis='y1'))

#     # YIELD_CURVE (2차 축)
#     fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['YIELD_CURVE'], name='장단기 금리차 (10Y-2Y)', line=dict(color='red', width=1.5), yaxis='y2', opacity=0.8))
#     fig_macro.add_hline(y=0, line_dash="dash", line_color="red", yref="y2")    
#     # BBB_OAS (3차 축)
#     fig_macro.add_trace(go.Scatter(x=df_macro_plot.index, y=df_macro_plot['BBB_OAS'], name='BBB 회사채 스프레드', line=dict(color='green', width=1.5), yaxis='y3', opacity=0.8))

#     fig_macro.update_layout(title="S&P 500 vs. 경기/신용 리스크 지표", xaxis_title="날짜",
#         yaxis=dict(title=dict(text='S&P 500 종가', font=dict(color="#1f77b4")), domain=[0, 1]),
#         yaxis2=dict(title=dict(text='금리차 (%)', font=dict(color="red")), overlaying='y', side='right', position=0.95, showgrid=False),
#         yaxis3=dict(title=dict(text='BBB OAS', font=dict(color="green")), overlaying='y', side='right', position=1.0, showgrid=False),
#         hovermode="x unified", height=600, legend=dict(x=0, y=1.05, orientation="h"))
    
#     st.plotly_chart(fig_macro, use_container_width=True)


#     # --- C. 예측 vs. 실제 수익률 시각화 ---
#     st.subheader("📈 LightGBM 예측 vs. 실제 수익률 (90% 신뢰구간)")
    
#     y_test_df = pd.DataFrame({
#         'Actual': y_test, 'Predicted': y_test_pred,
#         'Low_CI': y_test_pred - CI_FACTOR, 'High_CI': y_test_pred + CI_FACTOR
#     }, index=df_ml.index[-test_size:])

#     fig_pred = go.Figure()

#     fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['High_CI'], mode='lines', line=dict(width=0), showlegend=False))
#     fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Low_CI'], fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', mode='lines', line=dict(width=0), name='90% 신뢰구간'))
#     fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Actual'], mode='markers', name='실제 수익률', marker=dict(color='blue', size=5, opacity=0.8)))
#     fig_pred.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['Predicted'], mode='lines', name='예측 수익률 (Median)', line=dict(color='red', width=2)))

#     fig_pred.update_layout(title=f"테스트 기간 S&P 500 수익률 예측 결과", xaxis_title="날짜", yaxis_title="수익률(%)", hovermode="x unified", height=500)
#     st.plotly_chart(fig_pred, use_container_width=True)
    
#     # --- D. 팩터 중요도 시각화 ---
#     st.subheader("🔍 팩터 중요도 (Feature Importance)")
    
#     importance_df = pd.DataFrame({
#         'Feature': features,
#         'Importance': lgbm_model.feature_importances_
#     }).sort_values('Importance', ascending=False).head(15)

#     fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
#                      title='LightGBM 모델 상위 15개 팩터 중요도',
#                      color='Importance', color_continuous_scale=px.colors.sequential.Viridis)
#     fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
#     st.plotly_chart(fig_imp, use_container_width=True)


# st.markdown("---")
# st.warning("⚠️ **면책 조항:** 이 모델은 교육 및 분석 목적으로만 제공됩니다. 실제 투자에 사용하기 전에 충분한 검증과 리스크 분석을 수행해야 합니다.")
