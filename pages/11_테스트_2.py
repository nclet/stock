import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import yfinance as yf
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="뉴스+모멘텀+VIX 기반 주가 예측", layout="wide")
st.title("🇰🇷 뉴스 감성 + 모멘텀 + VIX 기반 고급 주가 예측")

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
    return (score - 0.5) * 2

# ------------------------
# ✨ UI
# ------------------------
market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
company_list = fdr.StockListing(market_option)
company_names = company_list['Name'].tolist()
company_name = st.selectbox("✅ 분석할 기업 선택", company_names, index=company_names.index("삼성전자") if "삼성전자" in company_names else 0)
stock_code = company_list.loc[company_list['Name'] == company_name, 'Code'].values[0]

start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=60))
end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(company_name, display=30, start=1, sort="date"):
    client_id = st.secrets["naver"]["client_id"]
    client_secret = st.secrets["naver"]["client_secret"]

    import urllib.parse
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
            title = item.get('title', '').replace("<b>", "").replace("</b>", "")
            pub_date = item.get('pubDate', '')
            try:
                pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception:
                pub_date_dt = None
            news_data.append({'Date': pub_date_dt, 'Title': title})
        df = pd.DataFrame(news_data)
        return df
    else:
        st.error(f"API 요청 실패: 상태 코드 {response.status_code}")
        return pd.DataFrame()

# ------------------------
# ✨ 버튼 실행
# ------------------------
max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=200, value=50, step=10)

if st.button("🚀 뉴스 수집 및 분석 시작"):
    with st.spinner("뉴스 수집 및 감성 분석 중..."):
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

        df_stock = fdr.DataReader(stock_code, start_date - timedelta(days=30), end_date)
        if df_stock.empty:
            st.error("❌ 주가 데이터를 가져오지 못했습니다.")
        else:
            df_stock = df_stock.reset_index()[['Date', 'Close']]
            df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])

            # 모멘텀 지표 추가
            df_stock['Return_10d'] = df_stock['Close'].pct_change(10)
            df_stock['MA20'] = df_stock['Close'].rolling(window=20).mean()
            df_stock['Diff_MA20'] = df_stock['Close'] - df_stock['MA20']

            # VIX 추가
            vix = yf.download('^VIX', start=start_date - timedelta(days=30), end=end_date)
            vix = vix.reset_index()[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})

            df_all = pd.merge(df_stock, vix, on='Date', how='left')
            df_all = pd.merge(df_all, filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index(),
                              on='Date', how='left').fillna(0)

            df_all = df_all.dropna(subset=['Return_10d', 'Diff_MA20', 'VIX_Close'])

            # 모델 학습
            X = df_all[['Sentiment_Score', 'Return_10d', 'Diff_MA20', 'VIX_Close']].values
            y = df_all['Close'].values

            if len(X) > 30:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                y_pred = model.predict(X)
                df_all['Predicted_Close'] = y_pred

                # 시각화
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_all['Date'], df_all['Close'], label='Actual Close')
                ax.plot(df_all['Date'], df_all['Predicted_Close'], label='Predicted Close', linestyle='--')
                ax.set_title(f"{company_name} 고급 주가 예측 (감성+모멘텀+VIX)")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.metric(label="Feature Importance (샘플)", value="랜덤포레스트 자동 계산됨")
            else:
                st.warning("데이터가 부족하여 고급 예측을 수행할 수 없습니다.")

        st.markdown("---")
        st.write("👉 감성 점수 (-1 ~ 1), 모멘텀 지표, VIX를 함께 활용한 고급 예측 데모입니다.")
