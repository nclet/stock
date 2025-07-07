import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.linear_model import LinearRegression
import urllib.parse

# ------------------------
# ✨ 페이지 설정
# ------------------------
st.set_page_config(page_title="뉴스 감성분석 + 모멘텀 + VIX 전략", layout="wide")
st.title("뉴스 감성 + 모멘텀 + VIX 결합 주가 예측 전략")

st.markdown("""
네이버 뉴스, VIX(변동성 지수), 모멘텀 데이터를 결합하여  
기업의 주가를 더 정교하게 예측하는 통합 전략 예제입니다.
""")

# ------------------------
# ✨ 감성 분석 모델 로드
# ------------------------
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base")
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

def analyze_sentiment(text):
    if not text:
        return 0.0
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return (score - 0.5) * 2  # -1 ~ 1

# ------------------------
# ✨ 종목 선택 UI (session_state로 문제 해결)
# ------------------------
@st.cache_resource
def get_company_list(market):
    return fdr.StockListing(market)

market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
company_list = get_company_list(market_option)
company_names = company_list['Name'].tolist()

if "selected_company" not in st.session_state:
    st.session_state.selected_company = "삼성전자" if "삼성전자" in company_names else company_names[0]

company_name = st.selectbox(
    "✅ 분석할 기업 선택",
    company_names,
    index=company_names.index(st.session_state.selected_company),
    key="selected_company"
)

stock_code = company_list.loc[company_list['Name'] == st.session_state.selected_company, 'Code'].values[0]

start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=30))
end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(company_name, display=30, start=1, sort="date"):
    client_id = st.secrets["naver"]["client_id"]
    client_secret = st.secrets["naver"]["client_secret"]

    enc_query = urllib.parse.quote(company_name)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
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
    else:
        st.error(f"API 요청 실패: 상태 코드 {response.status_code}")
        return pd.DataFrame()

# ------------------------
# ✨ 실행 버튼
# ------------------------
max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=100, value=30, step=10)

if st.button("🚀 크롤링 및 분석 시작"):
    with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
        all_news = pd.DataFrame()
        for start_idx in range(1, max_news + 1, 100):
            count = min(100, max_news - start_idx + 1)
            df_part = get_naver_news_api(company_name, display=count, start=start_idx)
            all_news = pd.concat([all_news, df_part], ignore_index=True)
            if len(df_part) < count:
                break

        all_news = all_news.dropna(subset=['Date'])
        filtered_news = all_news[(all_news['Date'] >= start_date) & (all_news['Date'] <= end_date)]

    if filtered_news.empty:
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다.")
    else:
        filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)

        st.success("✅ 뉴스 감성 분석 완료!")
        st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False))

        # ------------------------
        # ✨ 주가 데이터
        # ------------------------
        df_stock = fdr.DataReader(stock_code, start_date, end_date)
        if df_stock.empty:
            st.error("❌ 주가 데이터를 가져오지 못했습니다.")
        else:
            df_stock = df_stock.reset_index()[['Date', 'Close']]
            df_stock['Date'] = pd.to_datetime(df_stock['Date'])

            # VIX 데이터
            vix = yf.download('^VIX', start=start_date - timedelta(days=30), end=end_date + timedelta(days=1))
            vix = vix.reset_index()
            vix = vix[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})
            vix['Date'] = pd.to_datetime(vix['Date'])

            # 모멘텀
            df_stock['Momentum'] = df_stock['Close'].diff()

            # Date 컬럼 타입 통일
            df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            vix['Date'] = pd.to_datetime(vix['Date'])
            filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
            
            # 뉴스 그룹핑 후 reset_index
            filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
            
            # 병합
            df_merge = pd.merge(df_stock, vix, on='Date', how='left')
            df_merge = pd.merge(df_merge, filtered_news_grouped, on='Date', how='left').fillna(0)

            # ------------------------
            # ✨ 회귀 예측
            # ------------------------
            X = df_merge[['Sentiment_Score', 'Momentum', 'VIX_Close']].fillna(0).values
            y = df_merge['Close'].values

            if len(X) > 5:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                df_merge['Predicted_Close'] = y_pred

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_merge['Date'], df_merge['Close'], label='Actual Close')
                ax.plot(df_merge['Date'], df_merge['Predicted_Close'], label='Predicted Close', linestyle='--')
                ax.set_title(f"{company_name} Stock Prediction (NEWS + MOMENTUM + VIX)")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.metric("회귀계수 (감성)", f"{model.coef_[0]:.2f}")
                st.metric("회귀계수 (모멘텀)", f"{model.coef_[1]:.2f}")
                st.metric("회귀계수 (VIX)", f"{model.coef_[2]:.2f}")
            else:
                st.warning("데이터가 부족하여 예측을 수행할 수 없습니다.")

        st.markdown("---")
        st.write("👉 감성점수는 부정 뉴스에 -1, 긍정 뉴스에 1 점수를 대입합니다. 즉, -1(부정)~1(긍정)으로 점수가 계산됩니다.")
