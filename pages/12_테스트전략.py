import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import yfinance as yf # yfinance는 이제 사용하지 않지만, 기존 코드에 있었으므로 임포트 유지
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.linear_model import LinearRegression
import urllib.parse
import matplotlib as mpl # 한글 폰트 설정용

# 한글깨짐방지
mpl.rc('font', family='Malgun Gothic')
# 마이너 버그 방지: 마이너스 깨짐 현상 방지
mpl.rcParams['axes.unicode_minus'] = False

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
    st.info("AI 감성 분석 모델 로드 중입니다. 잠시만 기다려 주세요...")
    try:
        # Hugging Face 토큰을 st.secrets에서 불러옵니다.
        # secrets.toml 파일에 HF_TOKEN = "YOUR_TOKEN_STRING" 형태로 저장되어 있어야 합니다.
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
        
        # 감성 분석에 특화된 모델로 변경: 'snunlp/KR-BERT-finetuned-sentiment'
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
            
        3.  **네트워크 환경 변경:**
            * 회사/학교 네트워크 환경이라면 방화벽이나 프록시 설정이 모델 다운로드를 차단할 수 있습니다. 개인 Wi-Fi나 모바일 핫스팟 등 다른 네트워크 환경에서 시도해 보세요.
            
        4.  **수동 모델 다운로드 및 로컬 로드 (최후의 수단):**
            * Hugging Face 웹사이트에서 모델 파일을 직접 다운로드하여 로컬 경로에 저장한 후, 코드에서 해당 로컬 경로를 지정하여 모델을 로드하는 방법을 시도할 수 있습니다. 이 방법은 네트워크 문제를 완전히 우회합니다.
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
    # 따라서 인덱스 1의 확률을 긍정 점수로 사용하고 -1 ~ 1 범위로 변환합니다.
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return (score - 0.5) * 2  # -1 ~ 1

# ------------------------
# ✨ 종목 선택 UI
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

            # ------------------------
            # ✨ VIX 데이터 (FinanceDataReader 사용으로 변경)
            # ------------------------
            st.info("📉 VIX(변동성 지수) 데이터를 로드 중입니다 (FinanceDataReader 사용)...")
            try:
                # FinanceDataReader로 VIX 데이터 로드
                # 'VIX'는 CBOE Volatility Index의 심볼입니다.
                vix_raw = fdr.DataReader('VIX', start=start_date - timedelta(days=30), end=end_date + timedelta(days=1))
                
                if vix_raw.empty:
                    st.warning("⚠️ VIX 데이터를 가져오지 못했습니다. 예측에 포함되지 않습니다.")
                    vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close'])
                else:
                    # --- START MultiIndex/KeyError FIX ---
                    # 인덱스 이름이 'Date'가 아닌 경우 'Date'로 명시적으로 설정
                    if vix_raw.index.name != 'Date':
                        vix_raw.index.name = 'Date'
                    
                    vix_temp = vix_raw.reset_index() # 이제 'Date' 컬럼이 확실히 생성됨

                    # 'Close' 또는 'Adj Close' 컬럼을 찾아 사용
                    col_to_use = None
                    if 'Close' in vix_temp.columns:
                        col_to_use = 'Close'
                    elif 'Adj Close' in vix_temp.columns: # 혹시 모를 대체 컬럼
                        col_to_use = 'Adj Close'
                    
                    if 'Date' in vix_temp.columns and col_to_use: # 'Date'와 값 컬럼 모두 존재하는지 확인
                        vix_processed = vix_temp[['Date', col_to_use]].rename(columns={col_to_use: 'VIX_Close'})
                        vix_processed['Date'] = pd.to_datetime(vix_processed['Date']) # datetime.datetime으로 변환
                        st.success("✅ VIX 데이터 로드 완료 (FinanceDataReader)!")
                    else:
                        st.warning("⚠️ VIX 데이터에 필요한 'Date' 또는 'Close'/'Adj Close' 컬럼이 없습니다. 예측에 포함되지 않습니다.")
                        vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close'])
                    # --- END MultiIndex/KeyError FIX ---

            except Exception as e:
                st.warning(f"⚠️ VIX 데이터 로드 중 오류 발생 (FinanceDataReader): {e}. 예측에 포함되지 않습니다.")
                vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close']) # 오류 발생 시 빈 데이터프레임으로 초기화
            
            # ------------------------
            # ✨ 모멘텀
            # ------------------------
            df_stock['Momentum'] = df_stock['Close'].diff()

            # Date 컬럼 타입 통일
            df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            vix_processed['Date'] = pd.to_datetime(vix_processed['Date']) # vix_processed['Date']는 이미 위에서 pd.to_datetime 처리됨
            filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
            
            # 🟢 핵심 수정: 뉴스 그룹핑 후 reset_index() 추가
            filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
            
            # 병합
            df_merge = pd.merge(df_stock, vix_processed, on='Date', how='left') # vix_processed 사용
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
