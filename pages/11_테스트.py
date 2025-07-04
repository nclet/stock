# ===============================
# 🇰🇷 뉴스 요약 + 감성 분석 + SVM 예측 (Streamlit 데모)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import re

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Transformers for summarization
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# ------------------------
# ✨ Streamlit 기본 설정
# ------------------------
st.set_page_config(page_title="뉴스 요약 + 감성 분석 기반 주가 예측", layout="wide")
st.title("🇰🇷 뉴스 요약 + 감성 분석 기반 SVM 주가 예측 데모")
st.markdown("뉴스 전문을 **요약**한 뒤, 벡터화하여 SVM으로 주가 등락 예측을 시연합니다.")

# ------------------------
# ✨ Summarization 모델 로드
# ------------------------
@st.cache_resource
def load_summary_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")
    return tokenizer, model

summary_tokenizer, summary_model = load_summary_model()

def summarize_text(text, max_length=128):
    inputs = summary_tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')
    summary_ids = summary_model.generate(inputs['input_ids'], max_length=max_length, num_beams=4, early_stopping=True)
    summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# ------------------------
# ✨ 네이버 뉴스 크롤링 함수 (간단화)
# ------------------------
def crawl_news(company_name, start_date, end_date, max_pages=2):
    base_url = "https://search.naver.com/search.naver"
    news_list = []

    start_str = start_date.strftime('%Y.%m.%d')
    end_str = end_date.strftime('%Y.%m.%d')
    start_param = start_date.strftime('%Y%m%d')
    end_param = end_date.strftime('%Y%m%d')

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for page in range(max_pages):
        start_idx = page * 10 + 1
        params = {
            'where': 'news',
            'query': company_name,
            'sort': 0,
            'ds': start_str,
            'de': end_str,
            'nso': f'so:r,p:from{start_param}to{end_param},a:all',
            'start': start_idx
        }
        try:
            res = requests.get(base_url, params=params, headers=headers, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')
            news_items = soup.select('div.news_area')
            for item in news_items:
                title_tag = item.select_one('a.news_tit')
                if title_tag:
                    title = title_tag['title']
                    news_list.append(title)
        except Exception as e:
            st.warning(f"크롤링 오류: {e}")
            break

    return news_list

# ------------------------
# ✨ UI: 종목, 기간 선택
# ------------------------
market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
company_list = fdr.StockListing(market_option)
company_names = company_list['Name'].tolist()
company_name = st.selectbox("✅ 기업 선택", company_names, index=company_names.index("삼성전자") if "삼성전자" in company_names else 0)

start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=30))
end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# ------------------------
# ✨ 버튼 실행
# ------------------------
if st.button("🚀 뉴스 요약 & 예측 실행"):
    with st.spinner("뉴스 크롤링 중..."):
        news_titles = crawl_news(company_name, start_date, end_date)
    
    if not news_titles:
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다.")
    else:
        st.success("✅ 뉴스 크롤링 완료!")
        st.write(f"총 {len(news_titles)}건의 뉴스 제목을 수집했습니다.")

        # ------------------------
        # ✨ 뉴스 전문 요약
        # ------------------------
        summaries = []
        for news in news_titles:
            summary = summarize_text(news)
            summaries.append(summary)

        df_news = pd.DataFrame({
            'Original_Title': news_titles,
            'Summary': summaries
        })
        st.dataframe(df_news)

        # ------------------------
        # ✨ 벡터화
        # ------------------------
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(summaries).toarray()

        # ------------------------
        # ✨ 주가 데이터 준비
        # ------------------------
        stock_code = company_list.loc[company_list['Name'] == company_name, 'Code'].values[0]
        df_stock = fdr.DataReader(stock_code, start_date, end_date)
        df_stock = df_stock.reset_index()
        df_stock = df_stock[['Date', 'Close']]

        # ------------------------
        # ✨ 단순 예측용 레이블 생성 (종가가 어제보다 올랐으면 1, 아니면 0)
        # ------------------------
        df_stock['Target'] = (df_stock['Close'].diff() > 0).astype(int).shift(-1).fillna(0).astype(int)

        # 뉴스 개수와 주가 데이터 개수를 맞추기 위해 최소 길이만큼 자름
        min_len = min(len(X), len(df_stock))
        X = X[:min_len]
        y = df_stock['Target'].values[:min_len]

        if len(X) > 5:
            # ------------------------
            # ✨ SVM 모델 학습
            # ------------------------
            model = SVC(kernel='linear', probability=True)
            model.fit(X, y)

            # ------------------------
            # ✨ 예측
            # ------------------------
            predictions = model.predict(X)
            acc = (predictions == y).mean()

            st.metric("SVM 예측 정확도", f"{acc * 100:.2f}%")
            st.write("예측 결과 (1=상승, 0=하락):", predictions.tolist())

        else:
            st.warning("데이터가 부족하여 학습 및 예측을 수행할 수 없습니다.")

    st.markdown("---")
    st.write("⚠️ 본 데모는 간단한 흐름 설명용이며, 실제 사용 시 더 많은 데이터와 정교한 파이프라인이 필요합니다.")
