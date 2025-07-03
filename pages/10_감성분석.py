import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# 딥러닝 감성 분석 관련 라이브러리
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    st.error("""
    **딥러닝 감성 분석 기능을 사용하려면 다음을 설치하세요:**
    `pip install transformers torch sentencepiece`
    """)
    st.stop()

from sklearn.linear_model import LinearRegression

# --- ✨ Streamlit 페이지 설정 ---
st.set_page_config(layout="wide", page_title="뉴스 감성 분석 및 주가 예측 데모")

st.title("📰 뉴스 감성 분석 및 주가 예측 데모")
st.markdown("네이버 뉴스를 기반으로 감성 분석을 수행하고, 그 결과를 활용해 주가를 예측합니다.")

# --- 딥러닝 기반 감성 분석 모델 로드 및 함수 ---

@st.cache_resource
def load_sentiment_model():
    st.info("AI 감성 분석 모델 로드 중입니다. 잠시만 기다려 주세요...")
    
    try:
        hf_token = st.secrets.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-finetuned-sentiment", use_auth_token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-BERT-finetuned-sentiment", use_auth_token=hf_token)
        st.success("✅ AI 감성 분석 모델 로드 완료!")
        return tokenizer, model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.stop()

tokenizer, sentiment_model = load_sentiment_model()

def analyze_sentiment_with_dl(text):
    if not text:
        return 0.0

    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = sentiment_model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)
        pos_score = probabilities[0][1].item()
        
        # 보다 구체적인 변환: (pos_score - 0.5) * 2 → -1 ~ 1
        standardized_score = (pos_score - 0.5) * 2
        standardized_score = np.clip(standardized_score, -1, 1)
        return standardized_score

    except Exception as e:
        st.warning(f"감성 분석 오류: {e}")
        return 0.0

# --- 뉴스 크롤링 함수 ---

@st.cache_data(ttl=3600)
def get_naver_news_with_sentiment(company_name, start_date, end_date, max_pages=3):
    base_url = "https://search.naver.com/search.naver"
    news_data_list = []

    st.info(f"'{company_name}' 관련 뉴스 크롤링 중...")

    start_date_str = start_date.strftime('%Y.%m.%d')
    end_date_str = end_date.strftime('%Y.%m.%d')
    start_date_param = start_date.strftime('%Y%m%d')
    end_date_param = end_date.strftime('%Y%m%d')

    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    total_crawled_news = 0
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
                date_tag = item.select_one('div.news_info > div.info_group > span.info')

                if title_tag and date_tag:
                    title = title_tag['title']
                    raw_date = date_tag.get_text().strip()

                    news_date = None
                    if "시간 전" in raw_date or "분 전" in raw_date or "일 전" in raw_date:
                        news_date = datetime.now().date()
                    elif re.match(r'\d{4}\.\d{2}\.\d{2}\.?', raw_date):
                        news_date = datetime.strptime(raw_date.rstrip('.'), '%Y.%m.%d').date()

                    if news_date and start_date.date() <= news_date <= end_date.date():
                        sentiment = analyze_sentiment_with_dl(title)
                        news_data_list.append({
                            'Date': news_date,
                            'Title': title,
                            'Sentiment_Score': sentiment
                        })
                        total_crawled_news += 1

            if len(news_items) < 10:
                break

        except Exception as e:
            st.warning(f"뉴스 크롤링 오류 (페이지 {i+1}): {e}")
            break

    if not news_data_list:
        return pd.DataFrame(columns=['Date', 'Title', 'Sentiment_Score'])

    df_news = pd.DataFrame(news_data_list)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    return df_news

# --- Streamlit UI ---

st.sidebar.header("🔍 검색 설정")
company_name = st.sidebar.text_input("기업 이름 또는 키워드 (예: 삼성전자)", "삼성전자")
today = datetime.now().date()
default_start = today - timedelta(days=7)

date_range = st.sidebar.date_input("뉴스 검색 기간", value=(default_start, today), max_value=today)

if len(date_range) == 2:
    start_date_input = datetime.combine(date_range[0], datetime.min.time())
    end_date_input = datetime.combine(date_range[1], datetime.max.time())
else:
    st.sidebar.warning("날짜를 선택해주세요.")
    st.stop()

max_pages_input = st.sidebar.slider("최대 페이지 수 (10개 뉴스/페이지)", 1, 10, 3)

if st.sidebar.button("🚀 뉴스 크롤링 및 감성 분석"):
    df_sentiment = get_naver_news_with_sentiment(company_name, start_date_input, end_date_input, max_pages_input)

    if not df_sentiment.empty:
        st.subheader("📰 감성 분석 결과")
        df_sentiment['Sentiment_Class'] = df_sentiment['Sentiment_Score'].apply(
            lambda x: "긍정 😊" if x > 0.3 else ("부정 😠" if x < -0.3 else "중립 😐")
        )
        st.dataframe(df_sentiment[['Date', 'Title', 'Sentiment_Score', 'Sentiment_Class']])

        # --- 일별 평균 감성 시각화 ---
        st.subheader("📊 일별 평균 감성 점수")
        daily_avg = df_sentiment.groupby('Date')['Sentiment_Score'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_avg['Date'], daily_avg['Sentiment_Score'], marker='o')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f"'{company_name}' 일별 평균 감성 점수")
        ax.set_ylabel("감성 점수 (-1 ~ 1)")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # --- 예제: 주가 예측 로직 추가 (간단 회귀) ---
        st.subheader("💹 간단한 뉴스 기반 주가 예측 예시")

        # 예제용 랜덤 주가 생성 (실제로는 API나 CSV에서 불러와야 함)
        np.random.seed(42)
        daily_avg['Close'] = 50000 + (daily_avg['Sentiment_Score'] * 5000) + np.random.normal(0, 2000, size=len(daily_avg))

        # Feature & Target
        X = daily_avg[['Sentiment_Score']]
        y = daily_avg['Close']

        model = LinearRegression()
        model.fit(X, y)
        daily_avg['Predicted_Close'] = model.predict(X)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(daily_avg['Date'], daily_avg['Close'], marker='o', label='실제 종가')
        ax2.plot(daily_avg['Date'], daily_avg['Predicted_Close'], marker='x', linestyle='--', label='예측 종가')
        ax2.set_title(f"뉴스 감성 기반 예측 종가 (예제)")
        ax2.legend()
        ax2.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        st.info("⚠️ 위 예측은 단순한 데모용 예제입니다. 실제 주가 데이터와 연동하면 더 정확한 분석이 가능합니다.")

    else:
        st.warning("뉴스 데이터를 가져오지 못했습니다.")

st.markdown("---")
st.write("### 참고")
st.write("""
- **뉴스 크롤링:** 네이버 뉴스 검색 결과를 사용합니다. 너무 많은 요청은 네이버 정책에 위배될 수 있습니다.
- **감성 분석:** `snunlp/KR-BERT-finetuned-sentiment` 모델은 영화 리뷰 기반으로 학습되었으며, 뉴스 도메인에 최적화되어 있지 않을 수 있습니다.
- **주가 예측 예제:** 실제 주가 API나 CSV를 연동하면 더 정밀한 예측이 가능합니다.
""")
