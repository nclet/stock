# ===============================
# 🇰🇷 뉴스 감성 분석 기반 주가 예측 앱 (최종본)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

# 딥러닝 감성 분석 라이브러리
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------------------
# ✨ 페이지 설정
# ------------------------
st.set_page_config(page_title="뉴스 감성 기반 주가 예측", layout="wide")

st.title("🇰🇷 한국 증시 뉴스 기반 감성 분석 & 주가 예측")
st.markdown("""
본 앱은 **네이버 뉴스 제목**을 크롤링하고, 딥러닝 감성 분석으로 점수를 추출한 뒤,  
과거 주가와 결합하여 단순 선형 회귀 기반 주가 예측을 시연합니다.
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
# ✨ 종목 선택 UI
# ------------------------
market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
company_list = fdr.StockListing(market_option)
company_names = company_list['Name'].tolist()
company_name = st.selectbox("✅ 분석할 기업 선택", company_names, index=company_names.index("삼성전자") if "삼성전자" in company_names else 0)
stock_code = company_list.loc[company_list['Name'] == company_name, 'Code'].values[0]

start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=30))
end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# ------------------------
# ✨ 네이버 뉴스 크롤링 함수 (최신 구조 반영 + User-Agent 추가)
# ------------------------
def get_naver_news_with_sentiment(company_name, start_date, end_date, max_pages=3):
    base_url = "https://search.naver.com/search.naver"
    news_data_list = []

    start_date_str = start_date.strftime('%Y.%m.%d')
    end_date_str = end_date.strftime('%Y.%m.%d')
    start_date_param = start_date.strftime('%Y%m%d')
    end_date_param = end_date.strftime('%Y%m%d')

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    start_date_dt = start_date if isinstance(start_date, datetime) else datetime.combine(start_date, datetime.min.time())
    end_date_dt = end_date if isinstance(end_date, datetime) else datetime.combine(end_date, datetime.min.time())

    for i in range(max_pages):
        start_idx = i * 10 + 1
        params = {
            'where': 'news',
            'query': company_name,
            'sort': 0,
            'ds': start_date_str,
            'de': end_date_str,
            'nso': f'so:r,p:from{start_date_param}to{end_date_param},a:all',
            'start': start_idx
        }
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.select('div.news_area')
            if not news_items:
                break

            for item in news_items:
                title_tag = item.select_one('a.news_tit')
                date_tag_list = item.select('div.info_group span.info')
                
                raw_date = None
                for span in date_tag_list:
                    text = span.get_text().strip()
                    if re.match(r'\d{4}\.\d{2}\.\d{2}\.', text) or "시간 전" in text or "분 전" in text or "일 전" in text:
                        raw_date = text
                        break

                if title_tag and raw_date:
                    title = title_tag['title']

                    if "시간 전" in raw_date or "분 전" in raw_date or "일 전" in raw_date:
                        news_date = datetime.now().date()
                    elif re.match(r'\d{4}\.\d{2}\.\d{2}\.', raw_date):
                        news_date = datetime.strptime(raw_date, '%Y.%m.%d.').date()
                    else:
                        continue

                    if start_date_dt.date() <= news_date <= end_date_dt.date():
                        sentiment = analyze_sentiment(title)
                        news_data_list.append({
                            'Date': news_date,
                            'Title': title,
                            'Sentiment_Score': sentiment
                        })

        except Exception as e:
            st.warning(f"뉴스 크롤링 오류: {e}")
            break

    if not news_data_list:
        return pd.DataFrame(columns=['Date', 'Sentiment_Score'])

    df_news = pd.DataFrame(news_data_list)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    df_daily = df_news.groupby('Date')['Sentiment_Score'].mean().reset_index()

    return df_daily

# ------------------------
# ✨ 버튼 실행
# ------------------------
if st.button("🚀 뉴스 크롤링 및 분석 시작"):
    with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
        df_news = get_naver_news_with_sentiment(company_name, start_date, end_date)

    if df_news.empty:
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다.")
    else:
        st.success("✅ 뉴스 감성 분석 완료!")
        st.dataframe(df_news)

        # ------------------------
        # ✨ 주가 데이터 로드
        # ------------------------
        df_stock = fdr.DataReader(stock_code, start_date, end_date)
        if df_stock.empty:
            st.error("❌ 주가 데이터를 가져오지 못했습니다.")
        else:
            df_stock = df_stock.reset_index()[['Date', 'Close']]

            # ------------------------
            # ✨ 주가 & 감성 점수 병합
            # ------------------------
            df_merged = pd.merge(df_stock, df_news, on='Date', how='left').fillna(0)

            # ------------------------
            # ✨ 단순 선형회귀 예측 (데모용)
            # ------------------------
            from sklearn.linear_model import LinearRegression

            X = df_merged[['Sentiment_Score']].values
            y = df_merged['Close'].values

            if len(X) > 2:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                df_merged['Predicted_Close'] = y_pred

                # ------------------------
                # ✨ 결과 시각화
                # ------------------------
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_merged['Date'], df_merged['Close'], label='Actual Close')
                ax.plot(df_merged['Date'], df_merged['Predicted_Close'], label='Predicted Close', linestyle='--')
                ax.set_title(f"{company_name} 주가 및 감성 기반 예측")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.metric(label="회귀계수 (감성 점수 → 주가)", value=f"{model.coef_[0]:.2f}")
            else:
                st.warning("데이터가 부족하여 예측을 수행할 수 없습니다.")

        st.markdown("---")
        st.write("감성 점수는 -1 (강한 부정) ~ 1 (강한 긍정) 범위이며, 단순 예측 데모용입니다.")

