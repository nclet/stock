import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

# 딥러닝 감성 분석 관련 라이브러리 임포트
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    st.error("""
    **딥러닝 감성 분석 기능을 사용하려면 다음 라이브러리를 설치해야 합니다:**
    `pip install transformers torch sentencepiece`
    """)
    st.stop()

# --- ✨ Streamlit 페이지 설정 (가장 첫 번째 Streamlit 명령이어야 함) ✨ ---
st.set_page_config(layout="wide", page_title="뉴스 감성 분석 데모")

st.title("📰 네이버 뉴스 감성 분석 데모")
st.markdown("특정 기업의 네이버 뉴스 기사를 크롤링하고, 딥러닝 모델을 사용하여 기사 제목의 감성을 분석합니다.")

# --- 딥러닝 기반 감성 분석 모델 로드 및 함수 ---

@st.cache_resource
def load_sentiment_model():
    """
    사전 학습된 한국어 감성 분석 모델과 토크나이저를 로드합니다.
    모델: 'snunlp/KR-BERT-finetuned-sentiment' (네이버 영화 리뷰 데이터셋으로 학습됨)
    """
    st.info("AI 감성 분석 모델을 로드 중입니다. 잠시만 기다려 주세요...")
    try:
        # Hugging Face 토큰을 st.secrets에서 불러옵니다.
        # secrets.toml 파일에 HF_TOKEN = "YOUR_TOKEN_STRING" 형태로 저장되어 있어야 합니다.
        hf_token = st.secrets.get("HF_TOKEN") 
        if hf_token:
            st.info("Hugging Face 토큰을 사용하여 모델을 로드합니다.")
        else:
            st.warning("Hugging Face 토큰이 secrets.toml에 설정되지 않았거나 불러올 수 없습니다. 공개 모델은 토큰 없이 시도합니다.")

        # KR-BERT 모델 및 토크나이저 로드
        # force_download=True 를 추가하여 캐시를 무시하고 강제로 다시 다운로드 시도
        tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-finetuned-sentiment", token=hf_token, force_download=True)
        model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-BERT-finetuned-sentiment", token=hf_token, force_download=True)
        st.success("✅ AI 감성 분석 모델 로드 완료!")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ AI 감성 분석 모델 로드 중 오류 발생: {e}")
        st.stop()

tokenizer, sentiment_model = load_sentiment_model()

def analyze_sentiment_with_dl(text):
    """
    사전 학습된 딥러닝 모델을 사용하여 텍스트의 감성을 분석합니다.
    Args:
        text (str): 분석할 한국어 텍스트.
    Returns:
        float: 감성 점수 (긍정: 1에 가까움, 부정: 0에 가까움).
               여기서는 긍정/부정 확률을 기반으로 점수를 계산합니다.
               모델에 따라 클래스 순서가 다를 수 있으므로 확인 필요.
               snunlp/KR-BERT-finetuned-sentiment 모델은 0: 부정, 1: 긍정으로 학습됨.
    """
    if not text:
        return 0.0 # 빈 텍스트는 중립으로 처리

    try:
        # 텍스트 토큰화 및 모델 입력 준비
        inputs = tokenizer(
            text,
            return_tensors='pt', # PyTorch 텐서 반환
            truncation=True,     # 최대 길이 초과 시 자르기
            padding=True         # 패딩 추가
        )

        # 모델 예측 (로짓 반환)
        with torch.no_grad(): # 그래디언트 계산 비활성화 (추론 시)
            outputs = sentiment_model(**inputs)

        # 소프트맥스 함수를 적용하여 확률로 변환
        probabilities = torch.softmax(outputs.logits, dim=1)

        # '긍정' 클래스에 해당하는 확률을 감성 점수로 사용 (인덱스 1이 긍정, 0이 부정)
        sentiment_score = probabilities[0][1].item() # 긍정 확률
        
        # -1 (부정) ~ 1 (긍정) 범위로 변환
        # 0.5를 기준으로 긍정/부정으로 나뉘므로, (확률 - 0.5) * 2 로 변환
        return (sentiment_score - 0.5) * 2 
    except Exception as e:
        st.warning(f"감성 분석 중 오류 발생: {e}. 해당 뉴스는 중립으로 처리됩니다.")
        return 0.0 # 오류 발생 시 중립으로 처리

# --- 뉴스 크롤링 함수 ---

@st.cache_data(ttl=3600) # 뉴스 크롤링 결과를 1시간 동안 캐싱
def get_naver_news_with_sentiment(company_name, start_date, end_date, max_pages=5):
    """
    네이버 뉴스에서 특정 회사 관련 뉴스를 크롤링하고 딥러닝 감성 분석을 수행합니다.
    Args:
        company_name (str): 검색할 회사 이름.
        start_date (datetime): 검색 시작 날짜.
        end_date (datetime): 검색 종료 날짜.
        max_pages (int): 검색할 최대 페이지 수 (페이지당 10개 뉴스).
    Returns:
        pd.DataFrame: 'Date', 'Title', 'Sentiment_Score' 컬럼을 포함하는 데이터프레임.
    """
    base_url = "https://search.naver.com/search.naver"
    news_data_list = []

    st.info(f"📰 '{company_name}' 관련 뉴스 크롤링 및 딥러닝 감성 분석 중입니다...")
    
    start_date_str = start_date.strftime('%Y.%m.%d')
    end_date_str = end_date.strftime('%Y.%m.%d')
    start_date_param = start_date.strftime('%Y%m%d')
    end_date_param = end_date.strftime('%Y%m%d')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    total_crawled_news = 0
    for i in range(max_pages):
        start_idx = i * 10
        params = {
            'where': 'news',
            'query': company_name,
            'sort': 0, # 0: 최신순
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
                    elif re.match(r'\d{4}\.\d{2}\.\d{2}\.', raw_date):
                        news_date = datetime.strptime(raw_date, '%Y.%m.%d.').date()
                    
                    if news_date and start_date.date() <= news_date <= end_date.date():
                        sentiment = analyze_sentiment_with_dl(title)
                        news_data_list.append({
                            'Date': news_date,
                            'Title': title,
                            'Sentiment_Score': sentiment
                        })
                        total_crawled_news += 1
            
            if len(news_items) < 10: # 페이지당 10개 미만이면 마지막 페이지로 간주
                break
        
        except requests.exceptions.RequestException as e:
            st.warning(f"네이버 뉴스 크롤링 중 HTTP 요청 오류 발생 (페이지 {i+1}): {e}")
            break
        except Exception as e:
            st.warning(f"뉴스 데이터 파싱 중 예상치 못한 오류 발생 (페이지 {i+1}): {e}")
            break

    if not news_data_list:
        st.warning(f"'{company_name}' 관련 뉴스를 찾을 수 없거나 크롤링에 실패했습니다. 다른 검색어로 시도하거나 기간을 조정해보세요.")
        return pd.DataFrame(columns=['Date', 'Title', 'Sentiment_Score'])

    df_news = pd.DataFrame(news_data_list)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    
    st.success(f"✅ 총 {total_crawled_news}개 뉴스 크롤링 및 감성 분석 완료!")
    return df_news

# --- Streamlit UI ---

# 사용자 입력 섹션
st.sidebar.header("🔍 검색 설정")
company_name = st.sidebar.text_input("기업 이름 또는 키워드를 입력하세요 (예: 삼성전자)", "삼성전자")

today = datetime.now().date()
default_start_date = today - timedelta(days=7) # 기본적으로 7일 전부터
date_range = st.sidebar.date_input(
    "뉴스 검색 기간을 선택하세요:",
    value=(default_start_date, today),
    max_value=today # 오늘 날짜까지만 선택 가능
)

if len(date_range) == 2:
    start_date_input = datetime.combine(date_range[0], datetime.min.time())
    end_date_input = datetime.combine(date_range[1], datetime.max.time())
else:
    st.sidebar.warning("뉴스 검색 기간을 선택해주세요.")
    st.stop()

max_pages_input = st.sidebar.slider("크롤링할 최대 페이지 수 (페이지당 10개 뉴스)", 1, 10, 3)

if st.sidebar.button("🚀 뉴스 검색 및 감성 분석 시작"):
    if not company_name:
        st.sidebar.error("기업 이름을 입력해주세요.")
    else:
        with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
            df_sentiment_results = get_naver_news_with_sentiment(
                company_name,
                start_date_input,
                end_date_input,
                max_pages_input
            )

        if not df_sentiment_results.empty:
            st.subheader(f"'{company_name}' 관련 뉴스 감성 분석 결과")
            
            # 감성 점수에 따라 긍정/부정/중립 분류
            def classify_sentiment(score):
                if score > 0.3: # 0.5 기준에서 0.3 이상이면 긍정
                    return "긍정 😊"
                elif score < -0.3: # 0.5 기준에서 -0.3 이하면 부정
                    return "부정 😠"
                else:
                    return "중립 😐"
            
            df_sentiment_results['Sentiment_Class'] = df_sentiment_results['Sentiment_Score'].apply(classify_sentiment)

            # 결과 테이블 표시
            st.dataframe(df_sentiment_results[['Date', 'Title', 'Sentiment_Score', 'Sentiment_Class']].sort_values(by='Date', ascending=False).reset_index(drop=True))

            # 감성 점수 요약
            st.subheader("기간별 감성 요약")
            daily_avg_sentiment = df_sentiment_results.groupby('Date')['Sentiment_Score'].mean().reset_index()
            
            # 감성 점수 시각화
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(daily_avg_sentiment['Date'], daily_avg_sentiment['Sentiment_Score'], marker='o', linestyle='-')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8) # 중립선
            ax.set_title(f"'{company_name}' 일별 평균 뉴스 감성 점수")
            ax.set_xlabel("날짜")
            ax.set_ylabel("감성 점수 (-1:부정 ~ 1:긍정)")
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # 전체 기간 평균 감성 점수
            avg_overall_sentiment = df_sentiment_results['Sentiment_Score'].mean()
            st.metric(label=f"전체 기간 평균 감성 점수 ({start_date_input.strftime('%Y-%m-%d')} ~ {end_date_input.strftime('%Y-%m-%d')})",
                      value=f"{avg_overall_sentiment:.2f}",
                      delta=classify_sentiment(avg_overall_sentiment))
            
            st.info("감성 점수: -1 (강한 부정) ~ 1 (강한 긍정). 0에 가까울수록 중립입니다.")

st.markdown("---")
st.write("### 참고")
st.write("""
- **뉴스 크롤링:** 네이버 뉴스 검색 결과를 기반으로 합니다. 과도한 요청은 웹사이트 정책에 위배될 수 있으므로, 크롤링 페이지 수를 제한했습니다.
- **딥러닝 감성 분석:** `snunlp/KR-BERT-finetuned-sentiment` 모델을 사용하여 기사 제목의 감성을 분석합니다. 이 모델은 영화 리뷰 데이터셋으로 학습되었으므로, 주식 뉴스 도메인에 완벽하게 적용되지 않을 수 있습니다. 더 정확한 분석을 위해서는 주식 뉴스에 특화된 데이터로 파인튜닝된 모델이 필요합니다.
- **감성 점수 해석:** 감성 점수는 -1 (강한 부정)부터 1 (강한 긍정)까지의 범위입니다. 0에 가까울수록 중립적인 감성입니다.
""")
