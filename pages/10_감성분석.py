import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import urllib.parse

# 감성 분석 함수 (임시)
def analyze_sentiment(text):
    # 실제 감성 분석 모델로 변경 가능
    return 1.0  # 긍정 점수 예시

# 네이버 뉴스 검색 API 호출 함수
def get_naver_news_api(query, display=30, start=1, sort="date"):
    client_id = st.secrets["naver"]["client_id"]
    client_secret = st.secrets["naver"]["client_secret"]

    enc_query = urllib.parse.quote(query)
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
        st.error(f"API 요청 실패: 상태 코드 {response.status_code}")
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
        # 최대 100건씩 나누어 여러번 호출 가능 (최대 1000건)
        for start_idx in range(1, max_results + 1, 100):
            count = min(100, max_results - start_idx + 1)
            df_part = get_naver_news_api(company_name, display=count, start=start_idx, sort="date")
            all_news = pd.concat([all_news, df_part], ignore_index=True)
            if len(df_part) < count:
                # 더 이상 뉴스 없음
                break

        if not all_news.empty:
            all_news = all_news.dropna(subset=['Date'])
            filtered_news = all_news[(all_news['Date'] >= start_date) & (all_news['Date'] <= end_date)]

            # 감성 분석 적용
            filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)

            st.write(f"총 {len(filtered_news)}건 뉴스 수집 및 감성 분석 완료")
            st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False))
        else:
            st.warning("검색된 뉴스가 없습니다.")
