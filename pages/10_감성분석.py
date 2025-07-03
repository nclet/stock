# ===============================
# 📄 뉴스 감성 분석 기반 주가 예측 앱 (최종본)
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
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    st.error("""
    **딥러닝 감성 분석 기능을 사용하려면 다음 라이브러리를 설치해야 합니다:**
    `pip install transformers torch sentencepiece`
    """)
    st.stop()

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
    st.info("AI 감성 분석 모델 로드 중입니다. 잠시만 기다려 주세요...")
    try:
        # Hugging Face 토큰을 st.secrets에서 불러옵니다.
        hf_token = st.secrets.get("HF_TOKEN") 
        if hf_token:
            st.info("Hugging Face 토큰 (secrets.toml에서 로드됨)을 사용하여 모델 로드를 시도합니다.")
        else:
            st.warning("""
            Hugging Face 토큰이 secrets.toml에 설정되지 않았거나 불러올 수 없습니다.
            '401 Unauthorized' 오류가 계속 발생한다면, 다음을 시도해주세요:
            1. secrets.toml 파일에 HF_TOKEN을 정확히 입력했는지 확인.
            2. 터미널에서 `pip install huggingface_hub` 후 `huggingface-cli login` 명령어로 로그인 시도.
            """)

        st.info(f"모델 'snunlp/KR-BERT-finetuned-sentiment' 로드를 시작합니다. (캐시 무시, 강제 다운로드 시도)")
        
        # KR-BERT 모델 및 토크나이저 로드
        # force_download=True 를 추가하여 캐시를 무시하고 강제로 다시 다운로드 시도
        tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-finetuned-sentiment", token=hf_token, force_download=True)
        model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-BERT-finetuned-sentiment", token=hf_token, force_download=True)
        st.success("✅ AI 감성 분석 모델 로드 완료!")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ AI 감성 분석 모델 로드 중 오류 발생: {e}")
        st.error("""
        '401 Unauthorized' 오류는 주로 토큰 문제 또는 네트워크 제한으로 인해 발생합니다.
        
        **다음 해결책들을 시도해 보세요:**
        
        1.  **Hugging Face 토큰 환경 변수 설정 (가장 강력한 방법):**
            * Hugging Face 웹사이트에서 유효한 Access Token을 복사합니다.
            * Windows 검색에서 '환경 변수'를 검색하여 '시스템 환경 변수 편집'을 엽니다.
            * '환경 변수' 버튼을 클릭하고, '시스템 변수' 섹션에서 '새로 만들기'를 클릭합니다.
            * 변수 이름: `HF_TOKEN`
            * 변수 값: 복사한 토큰 문자열을 붙여넣습니다.
            * 모든 창을 '확인'으로 닫은 후, **컴퓨터를 재부팅합니다.**
            * 재부팅 후 Streamlit 앱을 다시 실행해 보세요.
            
        2.  **Hugging Face 캐시 폴더 완전 삭제:**
            * `C:\Users\YOUR_USERNAME\.cache\huggingface\hub` 폴더를 **통째로 삭제**합니다. (YOUR_USERNAME은 본인의 사용자 이름입니다.)
            * 이후 Streamlit 앱을 다시 실행합니다.
            
        3.  **다른 공개 모델로 테스트 (문제 진단용):**
            * `snunlp/KR-BERT-finetuned-sentiment` 대신 `bert-base-uncased`와 같은 매우 일반적인 공개 모델을 로드해 보세요.
            * `load_sentiment_model` 함수 내의 모델 이름을 다음으로 변경하고 테스트합니다:
                ```python
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=hf_token, force_download=True)
                model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", token=hf_token, force_download=True)
                ```
            * 만약 `bert-base-uncased`도 실패한다면, 네트워크 환경 자체가 Hugging Face Hub로의 연결을 강력하게 차단하고 있을 가능성이 매우 높습니다.
            
        4.  **네트워크 환경 변경:**
            * 회사/학교 네트워크 환경이라면 방화벽이나 프록시 설정이 모델 다운로드를 차단할 수 있습니다. 개인 Wi-Fi나 모바일 핫스팟 등 다른 네트워크 환경에서 시도해 보세요.
            
        5.  **수동 모델 다운로드 및 로컬 로드 (최후의 수단):**
            * 이전 답변에서 안내해 드린 대로, Hugging Face 웹사이트에서 모델 파일을 직접 다운로드하여 로컬 경로에 저장한 후, 코드에서 해당 로컬 경로를 지정하여 모델을 로드하는 방법을 시도할 수 있습니다. 이 방법은 네트워크 문제를 완전히 우회합니다.
        """)
        st.stop()

tokenizer, sentiment_model = load_sentiment_model()

def analyze_sentiment(text):
    if not text:
        return 0.0
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    # snunlp/KR-BERT-finetuned-sentiment 모델은 0: 부정, 1: 긍정으로 학습됨
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return (score - 0.5) * 2  # -1 ~ 1 범위로 변환

# ------------------------
# ✨ 종목 선택 UI
# ------------------------
@st.cache_data(ttl=86400) # 종목 리스트는 하루에 한 번만 로드
def get_stock_listing(market):
    st.info(f"📊 {market} 종목 리스트를 로드 중입니다...")
    try:
        df = fdr.StockListing(market)
        st.success(f"✅ {market} 종목 리스트 로드 완료!")
        return df
    except Exception as e:
        st.error(f"❌ {market} 종목 리스트를 로드할 수 없습니다: {e}")
        st.stop()

market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
company_list_df = get_stock_listing(market_option)

if company_list_df.empty:
    st.warning("종목 리스트를 가져올 수 없습니다. 앱 실행이 어렵습니다.")
    st.stop()

company_names = company_list_df['Name'].tolist()
# "삼성전자"가 KOSPI에 있으므로, KOSPI 선택 시 기본값으로 설정
default_company_index = company_names.index("삼성전자") if "삼성전자" in company_names else 0
company_name = st.selectbox("✅ 분석할 기업 선택", company_names, index=default_company_index)
stock_code = company_list_df.loc[company_list_df['Name'] == company_name, 'Code'].values[0]

start_date = st.date_input("뉴스 검색 시작일", datetime.now().date() - timedelta(days=30))
end_date = st.date_input("뉴스 검색 종료일", datetime.now().date())

# ------------------------
# ✨ 네이버 뉴스 크롤링 함수
# ------------------------
@st.cache_data(ttl=3600) # 뉴스 크롤링 결과를 1시간 동안 캐싱
def get_naver_news_with_sentiment(company_name, start_date, end_date, max_pages=3):
    base_url = "https://search.naver.com/search.naver"
    news_data_list = []

    start_date_str = start_date.strftime('%Y.%m.%d')
    end_date_str = end_date.strftime('%Y.%m.%d')
    start_date_param = start_date.strftime('%Y%m%d')
    end_date_param = end_date.strftime('%Y%m%d')

    # User-Agent를 더 구체적인 형태로 변경 (이전 버전에서 사용했던 것)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    st.info(f"📰 '{company_name}' 관련 뉴스 크롤링 시작 (기간: {start_date_str} ~ {end_date_str})...")

    # 날짜 타입 맞추기 (datetime.date 객체로 변환)
    start_date_dt = start_date
    end_date_dt = end_date

    total_crawled_news_count = 0

    for i in range(max_pages):
        start_idx = i * 10 + 1 # 네이버 뉴스 페이지네이션은 1, 11, 21...
        params = {
            'where': 'news',
            'query': company_name,
            'sort': 0, # 0: 최신순
            'ds': start_date_str,
            'de': end_date_str,
            'nso': f'so:r,p:from{start_date_param}to{end_date_param},a:all',
            'start': start_idx
        }
        
        st.info(f"페이지 {i+1} (시작 인덱스: {start_idx}) 크롤링 중...")
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.select('div.news_area') # 뉴스 항목 컨테이너

            if not news_items:
                st.info(f"페이지 {i+1}에서 뉴스 항목을 찾을 수 없습니다. (HTML 구조 변경 또는 검색 결과 없음)")
                break # 더 이상 뉴스가 없으면 크롤링 중단

            page_news_count = 0
            for item in news_items:
                title_tag = item.select_one('a.news_tit') # 뉴스 제목
                date_tag = item.select_one('div.news_info > div.info_group > span.info') # 날짜 정보

                if title_tag and date_tag:
                    title = title_tag['title']
                    raw_date = date_tag.get_text().strip()

                    news_date = None
                    # 날짜 파싱 로직 강화
                    if re.match(r'\d{4}\.\d{2}\.\d{2}\.', raw_date): # "YYYY.MM.DD." 형식
                        news_date = datetime.strptime(raw_date, '%Y.%m.%d.').date()
                    elif "시간 전" in raw_date or "분 전" in raw_date or "일 전" in raw_date: # 상대 시간
                        # 상대 시간은 현재 날짜로 간주 (정확한 과거 날짜를 알기 어려움)
                        news_date = datetime.now().date()
                    else:
                        # 기타 알 수 없는 날짜 형식 처리 (예: "2023.01.01")
                        try:
                            news_date = pd.to_datetime(raw_date).date()
                        except ValueError:
                            st.warning(f"알 수 없는 날짜 형식 발견: '{raw_date}'. 이 뉴스는 건너뜁니다.")
                            continue # 다음 뉴스로 넘어감

                    # 날짜 범위 필터링
                    if news_date and start_date_dt <= news_date <= end_date_dt:
                        sentiment = analyze_sentiment(title)
                        news_data_list.append({
                            'Date': news_date,
                            'Title': title,
                            'Sentiment_Score': sentiment
                        })
                        page_news_count += 1
                        total_crawled_news_count += 1
                    # else:
                    #     st.info(f"날짜 범위 밖 뉴스: {news_date} - {title}") # 디버깅용
                else:
                    # 뉴스 항목 내에서 제목 또는 날짜 태그를 찾지 못한 경우
                    # st.info(f"뉴스 항목에서 제목 또는 날짜 태그를 찾을 수 없습니다: {item.prettify()}") # 디버깅용
                    pass # 이 메시지는 너무 많을 수 있으므로 기본적으로 비활성화

            st.info(f"페이지 {i+1}에서 {page_news_count}개의 유효한 뉴스를 처리했습니다.")
            
            if len(news_items) < 10: # 페이지당 10개 미만이면 마지막 페이지로 간주
                st.info(f"페이지 {i+1}의 뉴스 수가 10개 미만입니다. 마지막 페이지로 간주하고 크롤링을 종료합니다.")
                break

        except requests.exceptions.RequestException as e:
            st.error(f"❌ HTTP 요청 중 오류 발생 (페이지 {i+1}): {e}")
            st.error("네트워크 연결, 방화벽, 또는 네이버의 요청 차단 여부를 확인해주세요.")
            break
        except Exception as e:
            st.error(f"❌ 뉴스 데이터 파싱 중 예상치 못한 오류 발생 (페이지 {i+1}): {e}")
            break

    if not news_data_list:
        st.warning(f"'{company_name}' 관련 뉴스를 찾을 수 없거나 크롤링에 실패했습니다. 검색 조건(기간, 키워드)을 조정하거나 잠시 후 다시 시도해보세요.")
        return pd.DataFrame(columns=['Date', 'Sentiment_Score'])

    df_news = pd.DataFrame(news_data_list)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    df_daily = df_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
    st.success(f"✅ 총 {total_crawled_news_count}개의 뉴스 기사를 성공적으로 크롤링 및 분석했습니다.")
    return df_daily

# ------------------------
# ✨ 버튼 실행
# ------------------------
if st.button("🚀 뉴스 크롤링 및 분석 시작"):
    with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
        df_news = get_naver_news_with_sentiment(company_name, start_date, end_date)

    if df_news.empty:
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 위 메시지를 확인해주세요.")
    else:
        st.success("✅ 뉴스 감성 분석 완료!")
        st.subheader("📰 일별 평균 뉴스 감성 점수")
        st.dataframe(df_news.sort_values(by='Date', ascending=False).reset_index(drop=True))

        # ------------------------
        # ✨ 주가 데이터 로드
        # ------------------------
        st.info(f"📈 {company_name} ({stock_code}) 주가 데이터를 로드 중입니다...")
        try:
            df_stock = fdr.DataReader(stock_code, start_date, end_date)
            if df_stock.empty:
                st.error(f"❌ {company_name} ({stock_code}) 주가 데이터를 가져오지 못했습니다.")
                st.stop()
            df_stock = df_stock.reset_index()[['Date', 'Close']]
            st.success(f"✅ {company_name} 주가 데이터 로드 완료!")
        except Exception as e:
            st.error(f"❌ 주가 데이터 로드 중 오류 발생: {e}")
            st.stop()

        # ------------------------
        # ✨ 주가 & 감성 점수 병합
        # ------------------------
        st.info("📊 주가 데이터와 뉴스 감성 점수를 병합 중입니다...")
        df_merged = pd.merge(df_stock, df_news, on='Date', how='left').fillna(0)
        st.success("✅ 데이터 병합 완료!")

        # ------------------------
        # ✨ 단순 선형회귀 예측 (데모용)
        # ------------------------
        st.subheader("📉 감성 점수 기반 단순 선형 회귀 예측")
        from sklearn.linear_model import LinearRegression

        # 예측에 사용할 데이터 준비
        # 감성 점수 (X)와 종가 (y)
        X = df_merged[['Sentiment_Score']].values
        y = df_merged['Close'].values

        if len(X) > 1: # 최소 2개 이상의 데이터 포인트가 있어야 선형 회귀 가능
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            df_merged['Predicted_Close'] = y_pred

            # ------------------------
            # ✨ 결과 시각화
            # ------------------------
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_merged['Date'], df_merged['Close'], label='실제 종가', color='blue')
            ax.plot(df_merged['Date'], df_merged['Predicted_Close'], label='예측 종가 (감성 기반)', linestyle='--', color='red')
            ax.set_title(f"{company_name} 주가 및 감성 기반 예측")
            ax.set_xlabel("날짜")
            ax.set_ylabel("가격(₩/원)")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            st.metric(label="회귀계수 (감성 점수 → 주가 영향)", value=f"{model.coef_[0]:.2f}")
            st.info(f"회귀계수 {model.coef_[0]:.2f}는 감성 점수가 1점 증가할 때 주가가 약 {model.coef_[0]:.0f}원 변동함을 의미합니다.")
            st.info("이것은 매우 단순한 모델이며, 실제 주가 예측에는 다양한 요인과 복잡한 모델이 필요합니다.")
        else:
            st.warning("뉴스 또는 주가 데이터가 부족하여 예측을 수행할 수 없습니다. 검색 기간을 늘려보세요.")

        st.markdown("---")
        st.write("감성 점수는 -1 (강한 부정) ~ 1 (강한 긍정) 범위이며, 단순 예측 데모용입니다.")

