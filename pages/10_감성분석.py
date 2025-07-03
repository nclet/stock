import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="디버깅 뉴스 감성 기반 주가 예측", layout="wide")

st.title("🐞 네이버 뉴스 감성 분석 & 주가 예측 (디버깅 모드)")

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base")
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import urllib.parse

# 감성 분석 함수(예시, 본인이 가진 모델로 대체 가능)
def analyze_sentiment(text):
    # TODO: 본인의 딥러닝 감성 분석 모델 함수로 교체하세요.
    # 여기서는 임시로 긍정(1.0) 리턴
    return 1.0

# 네이버 뉴스 API 호출 함수
def get_naver_news_api(query, display=30, start=1, sort="date"):
    client_id = st.secrets["naver"]["client_id"]
    client_secret = st.secrets["naver"]["client_secret"]

    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        data = res.json()
        items = data.get('items', [])
        news_data = []
        for item in items:
            title = item.get('title', '')
            description = item.get('description', '')
            pub_date = item.get('pubDate', '')
            try:
                pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception:
                pub_date_dt = None
            news_data.append({
                'Date': pub_date_dt,
                'Title': title,
                'Description': description
            })
        df = pd.DataFrame(news_data)
        return df
    else:
        st.error(f"API 요청 실패: 상태 코드 {res.status_code}")
        return pd.DataFrame()

# Streamlit UI
st.title("네이버 뉴스 검색 및 감성 분석")

company_name = st.text_input("기업 이름 또는 키워드 입력", "삼성전자")

start_date = st.date_input("시작 날짜", datetime.today())
end_date = st.date_input("종료 날짜", datetime.today())

max_results = st.slider("최대 뉴스 건수 선택", min_value=10, max_value=100, value=30, step=10)

if st.button("뉴스 검색 및 감성 분석 실행"):
    with st.spinner("뉴스 검색 중..."):
        all_news = pd.DataFrame()
        for start_idx in range(1, max_results + 1, 100):
            count = min(100, max_results - start_idx + 1)
            df_part = get_naver_news_api(company_name, display=count, start=start_idx, sort="date")
            all_news = pd.concat([all_news, df_part], ignore_index=True)
            if len(df_part) < count:
                # 더 이상 뉴스가 없음
                break

        # 날짜 필터링
        if not all_news.empty:
            all_news = all_news.dropna(subset=['Date'])
            all_news = all_news[(all_news['Date'] >= start_date) & (all_news['Date'] <= end_date)]

            # 감성 분석 적용
            all_news['Sentiment_Score'] = all_news['Title'].apply(analyze_sentiment)

            st.write(f"총 {len(all_news)}건 뉴스 수집 및 감성 분석 완료")
            st.dataframe(all_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False))
        else:
            st.warning("검색된 뉴스가 없습니다.")
