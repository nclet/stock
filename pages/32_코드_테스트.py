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
from sklearn.preprocessing import MinMaxScaler
import time

# ------------------------
# ✨ 상수 및 페이지 설정
# ------------------------
st.set_page_config(page_title="한국 주식 뉴스 감성 분석 전략", layout="wide")
st.title("📰 한국 주식 뉴스 감성 분석 전략 (Plotly UI 강화)")

st.markdown("""
네이버 뉴스를 크롤링하여 기술적 데이터와 결합,
과거 패턴 기반의 시계열 분석을 통해 주요 한국 상장 기업의 주가를 분석하고 예측합니다.
""")

# ------------------------
# 0. 피처 엔지니어링 함수 (시계열 특성 추가)
# ------------------------
def create_features(df_merge):
    """
    주가 및 감성 데이터에 기술적 지표와 시계열 지연(Lag) 피처를 추가합니다.
    """
    df = df_merge.copy()
    
    # 1. 기술적 지표 (기존 유지)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

    # 2. 타겟 변수: 다음 날의 수익률 (%)
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1) * 100

    # 3. 시계열 지연(Lag) 피처 추가
    lags = [1, 3, 5]
    
    # 3-1. 감성 점수 지연 피처
    for lag in lags:
        df[f'Sentiment_Lag_{lag}'] = df['Sentiment_Score'].shift(lag)
        
    # 3-2. 종가 수익률 지연 피처
    df['Daily_Return'] = df['Close'].pct_change() * 100
    for lag in lags:
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

    # 3-3. 거래량 지연 피처
    for lag in lags:
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

    df = df.dropna()
    
    base_features = ['Close', 'Volume', 'Open', 'High', 'Low', 'Sentiment_Score', 'SMA_20', 'Volatility']
    lag_features = [col for col in df.columns if 'Lag' in col or 'Daily_Return' in col]
    
    features = [f for f in base_features + lag_features if f in df.columns]
    
    return df, features

# ------------------------
# ✨ 감성 분석 모델 로드
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
        
        st.sidebar.success(f"✅ 감성 분석 모델 로드 완료 (장치: {device})")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰 설정 또는 라이브러리 버전을 확인해주세요.")
        st.stop()
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    """Calculates sentiment score for the given text."""
    if not text:
        return 0.0
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    neg_idx = None
    pos_idx = None
    for idx, label in sentiment_model.config.id2label.items():
        if 'negative' in label.lower() or '부정' in label:
            neg_idx = idx
        elif 'positive' in label.lower() or '긍정' in label:
            pos_idx = idx
    
    negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

    sentiment_score = positive_score - negative_score
    
    return sentiment_score

# ------------------------
# ✨ 종목 목록 로드 (FinanceDataReader)
# ------------------------
@st.cache_data(show_spinner="⏳ 종목 리스트를 로드 중입니다...")
def get_stock_list():
    """Loads KRX stock list using FinanceDataReader."""
    try:
        df_krx = fdr.StockListing('KRX')
        df_krx = df_krx[~df_krx['Name'].str.contains('리츠|스팩|ETN|ETF|인버스|곱버스|레버리지|선물|상장지수|지수', case=False, na=False)]
        return df_krx
    except Exception as e:
        st.error(f"❌ 종목 리스트 로드 중 오류 발생: {e}")
        st.stop()
        return pd.DataFrame()

df_krx = get_stock_list()
company_names = df_krx['Name'].tolist()

# ------------------------
# ✨ UI 입력 요소
# ------------------------
col_select, col_date_start, col_date_end, col_max_news = st.columns([2, 1, 1, 1])

with col_select:
    default_company = "삼성전자"
    if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
        st.session_state.selected_company = default_company if default_company in company_names else company_names[0]

    company_name = st.selectbox(
        "✅ 분석할 기업 선택",
        company_names,
        index=company_names.index(st.session_state.selected_company),
        key="selected_company"
    )

stock_code = df_krx[df_krx['Name'] == company_name]['Code'].iloc[0]

with col_date_start:
    start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=90))
with col_date_end:
    end_date = st.date_input("뉴스 검색 종료일", datetime.now())
with col_max_news:
    max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=200, value=100, step=10)


# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(query, display=30, start=1, sort="date"):
    """Fetches data from Naver News Search API."""
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError as e:
        st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
        st.info("Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
        return pd.DataFrame()

    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        data = response.json()
        items = data.get('items', [])
        news_data = []
        for item in items:
            title = item.get('title', '')
            pub_date = item.get('pubDate', '')
            try:
                pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception:
                pub_date_dt = None
            news_data.append({
                'Date': pub_date_dt,
                'Title': title
            })
        df = pd.DataFrame(news_data)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 실패: {e}")
        return pd.DataFrame()
    except JSONDecodeError as e:
        st.error(f"API 응답 파싱 실패: {e}")
        return pd.DataFrame()

# ------------------------
# ✨ 주가 데이터 로드 (FinanceDataReader)
# ------------------------
@st.cache_data(show_spinner="⏳ 주가 데이터를 로드 중입니다...")
def get_stock_data(code, start_date, end_date):
    """Loads daily stock data using FinanceDataReader."""
    # 시계열 피처 생성을 위해 검색 기간보다 20일 정도 더 많은 데이터를 로드합니다.
    load_start_date = start_date - timedelta(days=30) 
    try:
        df = fdr.DataReader(code, start=load_start_date, end=end_date)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ 주가 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# ------------------------
# ✨ 실행 로직
# ------------------------
st.markdown("---")
if st.button("🚀 크롤링 및 분석 시작", type="primary", use_container_width=True):
    
    # 1. 뉴스 크롤링 및 감성 분석
    with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
        all_news = pd.DataFrame()
        for start_idx in range(1, max_news + 1, 100):
            count = min(100, max_news - start_idx + 1)
            df_part = get_naver_news_api(company_name, display=count, start=start_idx)
            all_news = pd.concat([all_news, df_part], ignore_index=True)
            if len(df_part) < count:
                break
            time.sleep(0.5) 

        all_news = all_news.dropna(subset=['Date'])
        load_start_date = start_date - timedelta(days=30) 
        filtered_news = all_news[(all_news['Date'] >= load_start_date) & (all_news['Date'] <= end_date)].copy()

        if filtered_news.empty:
            st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 기업명을 확인해주세요.")
            st.stop()
        
        filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
        st.success("✅ 뉴스 크롤링 및 감성 분석 완료!")
        
    # 2. 주가 데이터 로드 및 병합
    df_stock = get_stock_data(stock_code, start_date, end_date)
    
    if df_stock.empty:
        st.error("❌ 주가 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
        st.stop()
    else:
        st.success("✅ 주가 데이터 로드 완료 (FinanceDataReader)!")
        
        df_stock.reset_index(inplace=True)
        filtered_news['Date'] = pd.to_datetime(filtered_news['Date']).dt.date
        
        filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
        df_merge = pd.merge(df_stock, filtered_news_grouped, on='Date', how='left')
        df_merge = df_merge.set_index('Date')
        
        df_merge['Sentiment_Score'] = df_merge['Sentiment_Score'].fillna(method='ffill').fillna(0) 

        # 3. 피처 엔지니어링 및 데이터 준비
        df_ml, features = create_features(df_merge)
        df_ml = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)]


        if len(df_ml) <= 100:
            st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 최소 100개 이상의 데이터가 필요합니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")
            st.stop()

        X = df_ml[features].values
        y = df_ml['Next_Day_Return'].values
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        test_size = max(1, int(0.2 * len(X_scaled)))
        X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # 4. 모델 훈련
        LGBM_TUNED_PARAMS = {
            'objective': 'regression', 'metric': 'rmse',
            'n_estimators': 700, 'learning_rate': 0.01, 
            'num_leaves': 21, 'max_depth': 7,
            'colsample_bytree': 0.8, 'subsample': 0.8,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }
        
        st.info("모델 훈련 중... (LGBM 파라미터 튜닝 및 시계열 특성 반영)")
        lgbm_model = lgb.LGBMRegressor(**LGBM_TUNED_PARAMS)
        
        lgbm_model.fit(X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])

        # 잔차 기반 신뢰구간 계산
        y_train_pred = lgbm_model.predict(X_train)
        residuals = y_train - y_train_pred
        residual_std = residuals.std()
        CI_FACTOR = 1.645 * residual_std # 90% CI
        
        # 예측 수행
        y_test_pred = lgbm_model.predict(X_test)
        
        # 5. 모델 성능 평가
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        # 6. 다음 날 예측
        last_data = df_ml[features].iloc[-1].values.reshape(1, -1)
        last_data_scaled = scaler.transform(last_data)
        next_day_return_pred = lgbm_model.predict(last_data_scaled)[0]
        
        low_ci = next_day_return_pred - CI_FACTOR
        high_ci = next_day_return_pred + CI_FACTOR

        st.markdown("---")
        st.subheader(f"✨ 최종 분석 및 예측 결과: {company_name} ({stock_code})")
        
        # --- A. 예측 결과 카드형 출력 ---
        col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

        def get_pred_color(pred):
            if pred > 0.5:
                return "green"
            elif pred < -0.5:
                return "red"
            else:
                return "orange"
                
        def format_pred_value(value):
            return f"{value:+.2f}%"

        with col_pred1:
            st.metric(label="📈 다음 날 예측 수익률", 
                      value=format_pred_value(next_day_return_pred), 
                      delta=f"CI 범위: {low_ci:+.2f}% ~ {high_ci:+.2f}%")

        with col_pred2:
            st.metric(label="✅ 예측 신뢰도 (R²)", 
                      value=f"{r2:.2f}", 
                      help=f"MSE: {mse:.4f}. 1에 가까울수록 모델의 적합도가 높음.")
            
        with col_pred3:
            sentiment_summary = filtered_news_grouped['Sentiment_Score'].iloc[-30:].mean()
            sentiment_trend = "긍정적 🟢" if sentiment_summary > 0.1 else ("부정적 🔴" if sentiment_summary < -0.1 else "중립 🟡")
            st.metric(label="📰 최근 30일 감성 점수 평균", 
                      value=f"{sentiment_summary:+.2f}", 
                      delta=sentiment_trend)

        with col_pred4:
            action = "매수 신호" if next_day_return_pred > 0.5 and low_ci > 0 else ("매도/관망" if next_day_return_pred < -0.5 else "관망")
            st.markdown(f"""
            <div style='
                padding: 10px; border-radius: 5px; text-align: center; 
                background-color: {"#D4EDDA" if action == "매수 신호" else ("#F8D7DA" if action == "매도/관망" else "#FFF3CD")}; 
                color: {"#155724" if action == "매수 신호" else ("#721C24" if action == "매도/관망" else "#856404")}; 
                font-weight: bold; margin-top: 15px;'>
                최종 액션: {action}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # --- B. Plotly: 주가 및 감성 점수 시각화 ---
        st.subheader("📊 주가 및 감성 점수 추이")

        df_plot = df_merge.copy()
        
        # 주가 데이터만 필터링 기간 내로 자릅니다. (감성 점수 lag를 위해 merge는 더 길게 진행)
        df_plot = df_plot[(df_plot.index >= start_date) & (df_plot.index <= end_date)]

        fig_price = go.Figure()

        # 1. 주가 (봉차트)
        fig_price.add_trace(go.Candlestick(x=df_plot.index,
                                           open=df_plot['Open'],
                                           high=df_plot['High'],
                                           low=df_plot['Low'],
                                           close=df_plot['Close'],
                                           name='주가 (OHLC)',
                                           yaxis='y1'))
        
        # 2. 감성 점수 (보조축)
        sentiment_color = df_plot['Sentiment_Score'].apply(lambda x: 'red' if x < 0 else 'green')
        fig_price.add_trace(go.Bar(x=df_plot.index, 
                                   y=df_plot['Sentiment_Score'], 
                                   name='감성 점수 평균', 
                                   yaxis='y2',
                                   marker_color=sentiment_color,
                                   opacity=0.5))

        # ⭐️ 오류가 발생했던 update_layout 부분 수정 ⭐️
        fig_price.update_layout(
            title=f"{company_name} 주가와 일일 감성 점수 비교",
            xaxis_title="날짜",
            # 주가 축: titlefont 대신 title.font 사용
            yaxis=dict(
                title=dict(text='종가', font=dict(color="#1f77b4")),
                tickfont=dict(color="#1f77b4"), 
                domain=[0.3, 1]
            ),
            # 감성 점수 축: titlefont 대신 title.font 사용
            yaxis2=dict(
                title=dict(text='감성 점수 (-1.0 ~ 1.0)', font=dict(color="#d62728")),
                tickfont=dict(color="#d62728"), 
                overlaying='y', 
                side='right', 
                domain=[0, 0.25]
            ),
            hovermode="x unified",
            height=600,
            legend=dict(x=0, y=1.1, orientation="h")
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
        
        
        # --- C. Plotly: 예측 vs. 실제 수익률 시각화 ---
        st.subheader("📈 모델 예측 vs. 실제 수익률 (90% 신뢰구간)")
        
        y_test_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred,
            'Low_CI': y_test_pred - CI_FACTOR,
            'High_CI': y_test_pred + CI_FACTOR
        }, index=df_ml.index[-test_size:])

        fig_pred = go.Figure()

        # 신뢰구간 (음영)
        fig_pred.add_trace(go.Scatter(
            x=y_test_df.index, y=y_test_df['High_CI'], 
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_pred.add_trace(go.Scatter(
            x=y_test_df.index, y=y_test_df['Low_CI'], 
            fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', 
            mode='lines', line=dict(width=0), name='90% 신뢰구간'
        ))
        
        # 실제 수익률 (마커만 표시)
        fig_pred.add_trace(go.Scatter(
            x=y_test_df.index, y=y_test_df['Actual'], 
            mode='markers', name='실제 수익률', marker=dict(color='blue', size=5, opacity=0.8)
        ))
        
        # 예측 수익률 (선으로 연결)
        fig_pred.add_trace(go.Scatter(
            x=y_test_df.index, y=y_test_df['Predicted'], 
            mode='lines', name='예측 수익률 (Median)', line=dict(color='red', width=2)
        ))

        fig_pred.update_layout(
            title=f"테스트 기간의 LightGBM 예측 결과 (수익률%)",
            xaxis_title="날짜",
            yaxis_title="수익률(%)",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")
    st.write("👉 **감성점수 계산 방식**: Hugging Face 모델에서 추출한 '긍정' 점수에서 '부정' 점수를 뺀 값이며, $\pm 1.0$ 범위를 가집니다.")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from sklearn.metrics import mean_squared_error, r2_score
# import urllib.parse
# from json.decoder import JSONDecodeError
# import FinanceDataReader as fdr
# import lightgbm as lgb
# from sklearn.preprocessing import MinMaxScaler
# import time

# # ------------------------
# # ✨ 상수 및 페이지 설정
# # ------------------------
# st.set_page_config(page_title="한국 주식 뉴스 감성 분석 전략", layout="wide")
# st.title("📰 한국 주식 뉴스 감성 분석 전략 (시계열 & CI 강화)")

# st.markdown("""
# 네이버 뉴스를 크롤링하여 기술적 데이터와 결합,
# 과거 패턴 기반의 시계열 분석을 통해 주요 한국 상장 기업의 주가를 분석하고 예측합니다.
# """)

# # ------------------------
# # 0. 피처 엔지니어링 함수 (시계열 특성 추가)
# # ------------------------
# def create_features(df_merge):
#     """
#     주가 및 감성 데이터에 기술적 지표와 시계열 지연(Lag) 피처를 추가합니다.
#     """
#     df = df_merge.copy()
    
#     # 1. 기술적 지표 (기존 유지)
#     df['SMA_20'] = df['Close'].rolling(window=20).mean()
#     df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

#     # 2. 타겟 변수: 다음 날의 수익률 (%)
#     # pct_change()는 NaN이 발생하므로, Target을 먼저 계산하고, 이후 dropna()를 통해 정리합니다.
#     df['Next_Day_Return'] = df['Close'].pct_change().shift(-1) * 100

#     # 3. 시계열 지연(Lag) 피처 추가 (핵심 개선 사항)
#     lags = [1, 3, 5]
    
#     # 3-1. 감성 점수 지연 피처
#     for lag in lags:
#         df[f'Sentiment_Lag_{lag}'] = df['Sentiment_Score'].shift(lag)
        
#     # 3-2. 종가 수익률 지연 피처
#     df['Daily_Return'] = df['Close'].pct_change() * 100
#     for lag in lags:
#         df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

#     # 3-3. 거래량 지연 피처
#     for lag in lags:
#         df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

#     # 필요한 모든 NaN을 제거합니다. (특히 초기 60일 데이터)
#     df = df.dropna()
    
#     # 최종적으로 사용할 피처 목록
#     base_features = ['Close', 'Volume', 'Open', 'High', 'Low', 'Sentiment_Score', 'SMA_20', 'Volatility']
#     lag_features = [col for col in df.columns if 'Lag' in col or 'Daily_Return' in col]
    
#     features = [f for f in base_features + lag_features if f in df.columns]
    
#     return df, features

# # ------------------------
# # ✨ 감성 분석 모델 로드
# # ------------------------
# @st.cache_resource
# def load_sentiment_model():
#     """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
#     # Load Hugging Face token from Streamlit secrets
#     hf_token = st.secrets.get("HF_TOKEN")
#     model_name = "snunlp/KR-FinBert-SC"
    
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
#         # Use 'auto' for device mapping to utilize GPU if available
#         model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
        
#         st.success(f"✅ 감성 분석 모델 : '{model_name}' (장치: {device})")
#         st.write(f"모델 라벨 맵핑: {model.config.id2label}")
        
#         return tokenizer, model, device
#     except Exception as e:
#         st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
#         st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
#         st.stop()
#         return None, None, None

# tokenizer, sentiment_model, device = load_sentiment_model()

# def analyze_sentiment(text):
#     """Calculates sentiment score for the given text."""
#     if not text:
#         return 0.0
    
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = sentiment_model(**inputs)
    
#     probabilities = torch.softmax(outputs.logits, dim=1)[0]

#     neg_idx = None
#     pos_idx = None
#     for idx, label in sentiment_model.config.id2label.items():
#         if 'negative' in label.lower() or '부정' in label:
#             neg_idx = idx
#         elif 'positive' in label.lower() or '긍정' in label:
#             pos_idx = idx
    
#     negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
#     positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

#     sentiment_score = positive_score - negative_score
    
#     return sentiment_score

# # ------------------------
# # ✨ 종목 목록 로드 (FinanceDataReader)
# # ------------------------
# @st.cache_data(show_spinner="⏳ 종목 리스트를 로드 중입니다...")
# def get_stock_list():
#     """Loads KRX stock list using FinanceDataReader."""
#     try:
#         df_krx = fdr.StockListing('KRX')
#         df_krx = df_krx[~df_krx['Name'].str.contains('리츠|스팩|ETN|ETF|인버스|곱버스|레버리지|선물|상장지수|지수', case=False, na=False)]
        
#         if df_krx.empty:
#             st.error("❌ FinanceDataReader에서 종목 리스트를 가져오지 못했습니다.")
#             st.stop()
            
#         return df_krx
#     except requests.exceptions.RequestException as e:
#         st.error(f"❌ 종목 리스트 로드 중 네트워크 오류 발생: {e}")
#         st.info("인터넷 연결 상태를 확인하거나 잠시 후 다시 시도해주세요.")
#         st.stop()
#     except JSONDecodeError as e:
#         st.error(f"❌ 종목 리스트 로드 중 데이터 파싱 오류 발생: {e}")
#         st.info("데이터 제공 서버가 일시적으로 불안정할 수 있습니다. 잠시 후 다시 시도하거나, `financedatareader` 라이브러리를 업데이트해 보세요.")
#         st.stop()
#     except Exception as e:
#         st.error(f"❌ 종목 리스트 로드 중 예기치 않은 오류 발생: {e}")
#         st.stop()
#     return pd.DataFrame()

# df_krx = get_stock_list()
# company_names = df_krx['Name'].tolist()

# # ------------------------
# # ✨ 주식 종목 선택 UI
# # ------------------------
# default_company = "삼성전자"
# if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
#     st.session_state.selected_company = default_company if default_company in company_names else company_names[0]

# company_name = st.selectbox(
#     "✅ 분석할 기업 선택",
#     company_names,
#     index=company_names.index(st.session_state.selected_company),
#     key="selected_company"
# )

# stock_code = df_krx[df_krx['Name'] == company_name]['Code'].iloc[0]

# # Date selection widgets
# start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=90))
# end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# # ------------------------
# # ✨ 네이버 뉴스 API 함수
# # ------------------------
# def get_naver_news_api(query, display=30, start=1, sort="date"):
#     """Fetches data from Naver News Search API."""
#     try:
#         client_id = st.secrets["naver"]["client_id"]
#         client_secret = st.secrets["naver"]["client_secret"]
#     except KeyError as e:
#         st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
#         st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
#         return pd.DataFrame()

#     enc_query = urllib.parse.quote(query)
#     url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

#     headers = {
#         "X-Naver-Client-Id": client_id,
#         "X-Naver-Client-Secret": client_secret
#     }

#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status() # Raise an exception for bad status codes
#         data = response.json()
#         items = data.get('items', [])
#         news_data = []
#         for item in items:
#             title = item.get('title', '')
#             pub_date = item.get('pubDate', '')
#             try:
#                 # Naver API date format: 'Fri, 23 Oct 2025 11:00:00 +0900'
#                 # Convert to date object
#                 pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
#             except Exception:
#                 pub_date_dt = None
#             news_data.append({
#                 'Date': pub_date_dt,
#                 'Title': title
#             })
#         df = pd.DataFrame(news_data)
#         return df
#     except requests.exceptions.RequestException as e:
#         st.error(f"API 요청 실패: {e}")
#         return pd.DataFrame()
#     except JSONDecodeError as e:
#         st.error(f"API 응답 파싱 실패: {e}")
#         return pd.DataFrame()

# # ------------------------
# # ✨ 주가 데이터 로드 (FinanceDataReader)
# # ------------------------
# @st.cache_data(show_spinner="⏳ 주가 데이터를 로드 중입니다...")
# def get_stock_data(code, start_date, end_date):
#     """Loads daily stock data using FinanceDataReader."""
#     # 시계열 피처 생성을 위해 검색 기간보다 20일 정도 더 많은 데이터를 로드합니다.
#     load_start_date = start_date - timedelta(days=30) 
#     try:
#         df = fdr.DataReader(code, start=load_start_date, end=end_date)
#         df.reset_index(inplace=True)
#         df['Date'] = pd.to_datetime(df['Date']).dt.date
#         df.set_index('Date', inplace=True)
#         return df
#     except Exception as e:
#         st.error(f"❌ 주가 데이터 로드 중 오류 발생: {e}")
#         return pd.DataFrame()

# # ------------------------
# # ✨ 실행 버튼
# # ------------------------
# max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=200, value=100, step=10)

# if st.button("🚀 크롤링 및 분석 시작"):
#     with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
#         all_news = pd.DataFrame()
#         # Naver API는 한 번에 최대 100건만 가능
#         for start_idx in range(1, max_news + 1, 100):
#             count = min(100, max_news - start_idx + 1)
#             df_part = get_naver_news_api(company_name, display=count, start=start_idx)
#             all_news = pd.concat([all_news, df_part], ignore_index=True)
#             if len(df_part) < count:
#                 break
#             time.sleep(0.5) # Add a small delay to avoid overwhelming the API

#     all_news = all_news.dropna(subset=['Date'])
#     # 주가 데이터 로드 시점을 고려하여 시작일보다 약간 이른 날짜부터 감성 분석 데이터를 확보
#     load_start_date = start_date - timedelta(days=30) 
#     filtered_news = all_news[(all_news['Date'] >= load_start_date) & (all_news['Date'] <= end_date)].copy()

#     if filtered_news.empty:
#         st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 기업명을 확인해주세요.")
#     else:
#         # 감성 분석
#         filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
#         st.success("✅ 뉴스 감성 분석 완료!")
#         st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False).head(10))

#         st.info(f"📈 {company_name} 주가 데이터를 로드 중입니다...")
#         df_stock = get_stock_data(stock_code, start_date, end_date)
            
#         if df_stock.empty:
#             st.error("❌ 주가 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
#             st.stop()
#         else:
#             st.success("✅ 주가 데이터 로드 완료 (FinanceDataReader)!")
            
#             # --- 데이터 병합 및 피처 엔지니어링 ---
#             df_stock.reset_index(inplace=True)
#             filtered_news['Date'] = pd.to_datetime(filtered_news['Date']).dt.date
            
#             # 날짜별 감성 점수 평균 계산
#             filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
            
#             # 주가 데이터와 감성 점수 병합
#             df_merge = pd.merge(df_stock, filtered_news_grouped, on='Date', how='left')
#             df_merge = df_merge.set_index('Date')
            
#             # 결측치 처리: 이전 값으로 채우고, 시작 부분은 0으로 채움
#             df_merge['Sentiment_Score'] = df_merge['Sentiment_Score'].fillna(method='ffill').fillna(0) 

#             # 피처 엔지니어링 실행 (시계열 특성 포함)
#             df_ml, features = create_features(df_merge)
            
#             # 주가 데이터 시작일-종료일 범위로 다시 필터링
#             df_ml = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)]


#             if len(df_ml) > 100:
#                 X = df_ml[features].values
#                 y = df_ml['Next_Day_Return'].values
                
#                 # Data normalization
#                 scaler = MinMaxScaler()
#                 X_scaled = scaler.fit_transform(X)
                
#                 # Split train/test data (Time-series split)
#                 test_size = max(1, int(0.2 * len(X_scaled)))
#                 X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
#                 y_train, y_test = y[:-test_size], y[-test_size:]
                
#                 # --- LightGBM 파라미터 튜닝 및 훈련 ---
#                 LGBM_TUNED_PARAMS = {
#                     'objective': 'regression',
#                     'metric': 'rmse',
#                     # 튜닝된 값: 정확도와 안정성 향상
#                     'n_estimators': 700, 
#                     'learning_rate': 0.01, 
#                     'num_leaves': 21,
#                     'max_depth': 7,
#                     'colsample_bytree': 0.8,
#                     'subsample': 0.8,
#                     'random_state': 42, 
#                     'n_jobs': -1, 
#                     'verbose': -1
#                 }
                
#                 st.info("모델 훈련 중... (LGBM 파라미터 튜닝 및 시계열 특성 반영)")
#                 lgbm_model = lgb.LGBMRegressor(**LGBM_TUNED_PARAMS)
                
#                 lgbm_model.fit(X_train, y_train,
#                                 eval_set=[(X_test, y_test)],
#                                 # 조기 종료 조건 강화 (과적합 방지)
#                                 callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])

#                 # 훈련 데이터셋 잔차 계산 (신뢰구간에 사용)
#                 y_train_pred = lgbm_model.predict(X_train)
#                 residuals = y_train - y_train_pred
#                 residual_std = residuals.std()
                
#                 # 90% 신뢰구간 계수 (Z-score for 90% CI is approx 1.645)
#                 CI_FACTOR = 1.645 * residual_std

#                 # 예측 수행
#                 y_test_pred = lgbm_model.predict(X_test)
                
#                 st.subheader("📊 모델 성능 평가")
#                 mse = mean_squared_error(y_test, y_test_pred)
#                 r2 = r2_score(y_test, y_test_pred)
#                 st.write(f"**평균 제곱 오차 (MSE)**: {mse:.4f}")
#                 st.write(f"**결정 계수 (R² Score)**: {r2:.4f}")
#                 st.write(f"**잔차 표준편차 (StdDev for CI)**: {residual_std:.4f}")
#                 st.write(f"**90% 신뢰구간 폭**: $\pm {CI_FACTOR:.4f}$ (%)")
                
#                 # --- 예측 결과 시각화 (Plotly로 변경: 신뢰구간 표현 용이) ---
#                 st.subheader("📈 예측 결과 시각화 (90% 신뢰구간 포함)")
                
#                 y_test_df = pd.DataFrame({
#                     'Actual': y_test,
#                     'Predicted': y_test_pred,
#                     'Low_CI': y_test_pred - CI_FACTOR,
#                     'High_CI': y_test_pred + CI_FACTOR
#                 }, index=df_ml.index[-test_size:])

#                 fig = go.Figure()

#                 # 신뢰구간 (음영)
#                 fig.add_trace(go.Scatter(
#                     x=y_test_df.index, y=y_test_df['High_CI'], 
#                     mode='lines', line=dict(width=0), showlegend=False
#                 ))
#                 fig.add_trace(go.Scatter(
#                     x=y_test_df.index, y=y_test_df['Low_CI'], 
#                     fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', 
#                     mode='lines', line=dict(width=0), name='90% 신뢰구간'
#                 ))
                
#                 # 실제 수익률 (마커만 표시)
#                 fig.add_trace(go.Scatter(
#                     x=y_test_df.index, y=y_test_df['Actual'], 
#                     mode='markers', name='실제 수익률', marker=dict(color='blue', size=5, opacity=0.8)
#                 ))
                
#                 # 예측 수익률 (선으로 연결)
#                 fig.add_trace(go.Scatter(
#                     x=y_test_df.index, y=y_test_df['Predicted'], 
#                     mode='lines', name='예측 수익률 (Median)', line=dict(color='red', width=2)
#                 ))

#                 fig.update_layout(
#                     title=f"{company_name} ({stock_code}) 예측 vs. 실제 수익률 (90% CI)",
#                     xaxis_title="날짜",
#                     yaxis_title="수익률(%)",
#                     hovermode="x unified"
#                 )
#                 st.plotly_chart(fig, use_container_width=True)

#                 st.markdown("---")
#                 st.subheader("💡 다음 날 주가 수익률 예측 및 신뢰구간")
                
#                 # Use the last data point to predict the next day's return
#                 last_data = df_ml[features].iloc[-1].values.reshape(1, -1)
#                 last_data_scaled = scaler.transform(last_data)
                
#                 next_day_return_pred = lgbm_model.predict(last_data_scaled)[0]
                
#                 # 신뢰구간 계산
#                 low_ci = next_day_return_pred - CI_FACTOR
#                 high_ci = next_day_return_pred + CI_FACTOR
                
#                 st.write(f"다음 영업일의 주가 수익률 예측: **{next_day_return_pred:.2f}%**")
#                 st.write(f"90% 신뢰구간 (CI): **{low_ci:.2f}%** 부터 **{high_ci:.2f}%** 까지")
                
#                 if next_day_return_pred > 0 and low_ci > 0:
#                     st.success("예측 수익률과 신뢰구간 하한 모두 긍정적입니다. **강한 매수 신호**로 고려해볼 수 있습니다. 🟢")
#                 elif next_day_return_pred > 0:
#                     st.info("예측 수익률은 긍정적이지만, 신뢰구간 하한은 0 이하입니다. **관망/약한 매수 신호**로 고려해볼 수 있습니다. 🟡")
#                 else:
#                     st.warning("예측 수익률이 부정적입니다. **매도 또는 관망 신호**로 고려해볼 수 있습니다. 🔴")
#             else:
#                 st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 최소 100개 이상의 데이터가 필요합니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")

#         st.markdown("---")
#         st.write("👉 **감성점수 계산 방식**: Hugging Face 모델에서 추출한 '긍정' 점수에서 '부정' 점수를 뺀 값입니다.")
