import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.linear_model import LinearRegression
import urllib.parse
# from huggingface_hub import login # Streamlit Secrets를 사용하므로 명시적 login은 필요 없습니다.

# ------------------------
# ✨ 페이지 설정
# ------------------------
st.set_page_config(page_title="뉴스 감성분석 + 모멘텀 + VIX 전략", layout="wide")
st.title("뉴스 감성 분석 전략")

st.markdown("""
네이버 뉴스를 크롤링하여 VIX(변동성 지수), 모멘텀 데이터를 결합하여
기업의 주가를 더 정교하게 예측하는 전략입니다.
""")

# ------------------------
# ✨ 감성 분석 모델 로드
# ------------------------
@st.cache_resource
def load_sentiment_model():
    # Streamlit Secrets에서 Hugging Face 토큰 가져오기
    # secrets.toml 파일에 HF_TOKEN = "your_token_here" 로 설정되어 있어야 합니다.
    hf_token = st.secrets.get("HF_TOKEN")

    # 'snunlp/KR-FinBert-SC' 모델로 변경
    model_name = "snunlp/KR-FinBert-SC"
    
    try:
        # 모델을 로드할 때 device_map을 'cpu'로 명시
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='cpu')
        
        # GPU 사용 가능 여부 확인 및 모델을 GPU로 이동
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        st.success(f"✅ 감성 분석 모델 : '{model_name}' (장치: {device})")
        st.write(f"모델 라벨 맵핑: {model.config.id2label}") # 라벨 맵핑 확인 필수!
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
        st.stop() # 모델 로드 실패 시 앱 중단
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    if not text:
        return 0.0 # 빈 텍스트는 0점 (중립)
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    # 입력 데이터를 모델이 있는 장치로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)[0] # 첫 번째 샘플의 확률

    neg_idx = None
    neu_idx = None
    pos_idx = None
    for idx, label in sentiment_model.config.id2label.items():
        if 'negative' in label.lower() or '부정' in label:
            neg_idx = idx
        elif 'neutral' in label.lower() or '중립' in label:
            neu_idx = idx
        elif 'positive' in label.lower() or '긍정' in label:
            pos_idx = idx
    
    negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    neutral_score = probabilities[neu_idx].item() if neu_idx is not None else 0
    positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

    sentiment_score = positive_score - negative_score 
    
    return sentiment_score

# ------------------------
# ✨ 종목 선택 UI
# ------------------------
@st.cache_data
def get_company_list_from_csv():
    # 'all_listed_shares_naver_crawled.csv' 파일을 읽을 때 'Code' 컬럼을 문자열로 명시
    df = pd.read_csv('all_listed_shares_naver_crawled.csv', dtype={'종목코드': str})
    # 컬럼 이름을 표준화합니다.
    df.columns = ['Code', 'Name', 'Market', 'corp_code', 'Shares']
    # 'Code' 컬럼의 모든 값을 6자리 문자열로 패딩하여 확실하게 처리합니다.
    df['Code'] = df['Code'].str.zfill(6)
    return df

company_list = get_company_list_from_csv()

market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])

# 선택된 시장에 해당하는 회사 목록만 필터링합니다.
company_names = company_list[company_list['Market'] == market_option]['Name'].tolist()

if "selected_company" not in st.session_state:
    st.session_state.selected_company = "삼성전자" if "삼성전자" in company_names else company_names[0]

company_name = st.selectbox(
    "✅ 분석할 기업 선택",
    company_names,
    index=company_names.index(st.session_state.selected_company),
    key="selected_company"
)

# 선택된 회사 이름으로 종목 코드를 찾습니다.
stock_code = company_list.loc[company_list['Name'] == st.session_state.selected_company, 'Code'].values[0]

start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=30))
end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(company_name, display=30, start=1, sort="date"):
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError as e:
        st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
        st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
        return pd.DataFrame()

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
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 기업명을 확인해주세요.")
    else:
        filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)

        st.success("✅ 뉴스 감성 분석 완료!")
        st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False))

        # ------------------------
        # ✨ 주가 데이터
        # ------------------------
        st.info(f"📈 {company_name} 주가 데이터를 로드 중입니다...")
        df_stock = fdr.DataReader(stock_code, start_date, end_date)
        if df_stock.empty:
            st.error("❌ 주가 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
            st.stop()
        else:
            df_stock = df_stock.reset_index()[['Date', 'Close']]
            df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            st.success("✅ 주가 데이터 로드 완료!")

            # ------------------------
            # ✨ VIX 데이터 (FinanceDataReader 사용)
            # ------------------------
            st.info("📉 VIX(변동성 지수) 데이터를 로드 중입니다 (FinanceDataReader 사용)...")
            try:
                vix_raw = fdr.DataReader('VIX', start=start_date - timedelta(days=60), end=end_date + timedelta(days=1))
                
                if vix_raw.empty:
                    st.warning("⚠️ VIX 데이터를 가져오지 못했습니다. 예측에 포함되지 않습니다.")
                    vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close'])
                else:
                    if vix_raw.index.name != 'Date':
                        vix_raw.index.name = 'Date'
                    
                    vix_temp = vix_raw.reset_index()
                    
                    col_to_use = None
                    if 'Close' in vix_temp.columns:
                        col_to_use = 'Close'
                    elif 'Adj Close' in vix_temp.columns:
                        col_to_use = 'Adj Close'
                    
                    if 'Date' in vix_temp.columns and col_to_use:
                        vix_processed = vix_temp[['Date', col_to_use]].rename(columns={col_to_use: 'VIX_Close'})
                        vix_processed['Date'] = pd.to_datetime(vix_processed['Date'])
                        st.success("✅ VIX 데이터 로드 완료 (FinanceDataReader)!")
                    else:
                        st.warning("⚠️ VIX 데이터에 필요한 'Date' 또는 'Close'/'Adj Close' 컬럼이 없습니다. 예측에 포함되지 않습니다.")
                        vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close'])
            except Exception as e:
                st.warning(f"⚠️ VIX 데이터 로드 중 오류 발생 (FinanceDataReader): {e}. 예측에 포함되지 않습니다.")
                vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close'])
                
            # ------------------------
            # ✨ 모멘텀
            # ------------------------
            df_stock['Momentum'] = df_stock['Close'].diff()

            df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            vix_processed['Date'] = pd.to_datetime(vix_processed['Date'])
            filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
            
            filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
            
            df_merge = pd.merge(df_stock, vix_processed, on='Date', how='left')
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

                st.subheader("📊 주가 예측 결과")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_merge['Date'], df_merge['Close'], label='Actual Close', color='blue')
                ax.plot(df_merge['Date'], df_merge['Predicted_Close'], label='Predicted Close', linestyle='--', color='red')
                ax.set_title(f"{company_name} Stock Prediction (NEWS + MOMENTUM + VIX)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Close Price")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.subheader("📈 회귀 모델 계수")
                st.metric("감성 점수 회귀계수", f"{model.coef_[0]:.2f}")
                st.metric("모멘텀 회귀계수", f"{model.coef_[1]:.2f}")
                st.metric("VIX 회귀계수", f"{model.coef_[2]:.2f}")
            else:
                st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")

        st.markdown("---")
        st.write("👉 감성점수는 부정 뉴스에 -1, 긍정 뉴스에 1 점수를 대입합니다. 즉, -1(부정)~1(긍정)으로 점수가 계산됩니다.")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# from datetime import datetime, timedelta
# import FinanceDataReader as fdr
# import matplotlib.pyplot as plt
# # import yfinance as yf # yfinance는 이제 사용하지 않지만, 기존 코드에 있었으므로 임포트 유지
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from sklearn.linear_model import LinearRegression
# import urllib.parse
# # from huggingface_hub import login # Streamlit Secrets를 사용하므로 명시적 login은 필요 없습니다.

# # ------------------------
# # ✨ 페이지 설정
# # ------------------------
# st.set_page_config(page_title="뉴스 감성분석 + 모멘텀 + VIX 전략", layout="wide")
# st.title("뉴스 감성 분석 전략")

# st.markdown("""
# 네이버 뉴스를 크롤링하여 VIX(변동성 지수), 모멘텀 데이터를 결합하여
# 기업의 주가를 더 정교하게 예측하는 전략입니다.
# """)

# # ------------------------
# # ✨ 감성 분석 모델 로드
# # ------------------------
# @st.cache_resource
# def load_sentiment_model():
#     # Streamlit Secrets에서 Hugging Face 토큰 가져오기
#     # secrets.toml 파일에 HF_TOKEN = "your_token_here" 로 설정되어 있어야 합니다.
#     hf_token = st.secrets.get("HF_TOKEN")

#     # 'snunlp/KR-FinBert-SC' 모델로 변경
#     model_name = "snunlp/KR-FinBert-SC"
    
#     try:
#         # 토큰을 from_pretrained 함수에 전달하여 인증
#         tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
#         model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
        
#         # GPU 사용 가능 여부 확인 및 모델을 GPU로 이동
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
        
#         st.success(f"✅ 감성 분석 모델 : '{model_name}' (장치: {device})")
#         st.write(f"모델 라벨 맵핑: {model.config.id2label}") # 라벨 맵핑 확인 필수!
        
#         return tokenizer, model, device
#     except Exception as e:
#         st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
#         st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
#         st.stop() # 모델 로드 실패 시 앱 중단
#         return None, None, None # 이 부분은 실행되지 않지만, 명시적으로 None 반환

# tokenizer, sentiment_model, device = load_sentiment_model()

# def analyze_sentiment(text):
#     if not text:
#         return 0.0 # 빈 텍스트는 0점 (중립)
    
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     # 입력 데이터를 모델이 있는 장치로 이동
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = sentiment_model(**inputs)
    
#     probabilities = torch.softmax(outputs.logits, dim=1)[0] # 첫 번째 샘플의 확률

#     # snunlp/KR-FinBert-SC 모델의 라벨 맵핑은 model.config.id2label 출력을 통해 정확히 확인해야 합니다.
#     # 일반적으로 {0: 'neutral', 1: 'positive', 2: 'negative'} 또는 {0: 'negative', 1: 'neutral', 2: 'positive'}
#     # 여기서는 라벨 이름을 기반으로 인덱스를 동적으로 찾습니다.
    
#     neg_idx = None
#     neu_idx = None
#     pos_idx = None
#     for idx, label in sentiment_model.config.id2label.items(): # sentiment_model 사용
#         if 'negative' in label.lower() or '부정' in label:
#             neg_idx = idx
#         elif 'neutral' in label.lower() or '중립' in label:
#             neu_idx = idx
#         elif 'positive' in label.lower() or '긍정' in label:
#             pos_idx = idx
    
#     # 인덱스가 None이 아닌지 확인하여 안전하게 확률을 가져옵니다.
#     negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
#     neutral_score = probabilities[neu_idx].item() if neu_idx is not None else 0
#     positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

#     # (긍정 확률 - 부정 확률)을 사용하면 -1에서 1 사이의 값을 얻을 수 있습니다.
#     # 이 점수 변환 방식은 모델의 특성 및 원하는 예측 결과에 따라 조정될 수 있습니다.
#     sentiment_score = positive_score - negative_score 
    
#     return sentiment_score

# # ------------------------
# # ✨ 종목 선택 UI
# # ------------------------
# @st.cache_resource
# def get_company_list(market):
#     return fdr.StockListing(market)

# market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
# company_list = get_company_list(market_option)
# company_names = company_list['Name'].tolist()

# if "selected_company" not in st.session_state:
#     st.session_state.selected_company = "삼성전자" if "삼성전자" in company_names else company_names[0]

# company_name = st.selectbox(
#     "✅ 분석할 기업 선택",
#     company_names,
#     index=company_names.index(st.session_state.selected_company),
#     key="selected_company"
# )

# stock_code = company_list.loc[company_list['Name'] == st.session_state.selected_company, 'Code'].values[0]

# start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=30))
# end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# # ------------------------
# # ✨ 네이버 뉴스 API 함수
# # ------------------------
# def get_naver_news_api(company_name, display=30, start=1, sort="date"):
#     # Streamlit Secrets에서 네이버 API 키를 가져옵니다.
#     # secrets.toml 파일에 [naver] client_id = "..." client_secret = "..." 설정되어 있어야 합니다.
#     try:
#         client_id = st.secrets["naver"]["client_id"]
#         client_secret = st.secrets["naver"]["client_secret"]
#     except KeyError as e:
#         st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
#         st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
#         return pd.DataFrame()

#     enc_query = urllib.parse.quote(company_name)
#     url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

#     headers = {
#         "X-Naver-Client-Id": client_id,
#         "X-Naver-Client-Secret": client_secret
#     }

#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         data = response.json()
#         items = data.get('items', [])
#         news_data = []
#         for item in items:
#             title = item.get('title', '')
#             pub_date = item.get('pubDate', '')
#             try:
#                 pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
#             except Exception:
#                 pub_date_dt = None
#             news_data.append({
#                 'Date': pub_date_dt,
#                 'Title': title
#             })
#         df = pd.DataFrame(news_data)
#         return df
#     else:
#         st.error(f"API 요청 실패: 상태 코드 {response.status_code}")
#         return pd.DataFrame()

# # ------------------------
# # ✨ 실행 버튼
# # ------------------------
# max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=100, value=30, step=10)

# if st.button("🚀 크롤링 및 분석 시작"):
#     with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
#         all_news = pd.DataFrame()
#         # 네이버 API는 한 번에 최대 100개까지 가져올 수 있습니다.
#         # max_news가 100을 초과할 경우 여러 번 호출
#         for start_idx in range(1, max_news + 1, 100):
#             count = min(100, max_news - start_idx + 1) # 남은 뉴스 수와 100 중 작은 값
#             df_part = get_naver_news_api(company_name, display=count, start=start_idx)
#             all_news = pd.concat([all_news, df_part], ignore_index=True)
#             if len(df_part) < count: # 더 이상 가져올 뉴스가 없으면 중단
#                 break
#             # API 호출 간 지연 시간 추가 (선택 사항, Rate Limit 방지)
#             # time.sleep(0.1) 

#         all_news = all_news.dropna(subset=['Date'])
#         filtered_news = all_news[(all_news['Date'] >= start_date) & (all_news['Date'] <= end_date)]

#     if filtered_news.empty:
#         st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 기업명을 확인해주세요.")
#     else:
#         # 감성 분석 수행
#         filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)

#         st.success("✅ 뉴스 감성 분석 완료!")
#         st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False))

#         # ------------------------
#         # ✨ 주가 데이터
#         # ------------------------
#         st.info(f"📈 {company_name} 주가 데이터를 로드 중입니다...")
#         df_stock = fdr.DataReader(stock_code, start_date, end_date)
#         if df_stock.empty:
#             st.error("❌ 주가 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
#             st.stop()
#         else:
#             df_stock = df_stock.reset_index()[['Date', 'Close']]
#             df_stock['Date'] = pd.to_datetime(df_stock['Date'])
#             st.success("✅ 주가 데이터 로드 완료!")

#             # ------------------------
#             # ✨ VIX 데이터 (FinanceDataReader 사용)
#             # ------------------------
#             st.info("📉 VIX(변동성 지수) 데이터를 로드 중입니다 (FinanceDataReader 사용)...")
#             try:
#                 # VIX 데이터는 주가 데이터 시작일보다 넉넉하게 가져와야 병합 시 데이터 손실을 줄일 수 있습니다.
#                 vix_raw = fdr.DataReader('VIX', start=start_date - timedelta(days=60), end=end_date + timedelta(days=1))
                
#                 if vix_raw.empty:
#                     st.warning("⚠️ VIX 데이터를 가져오지 못했습니다. 예측에 포함되지 않습니다.")
#                     vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close'])
#                 else:
#                     if vix_raw.index.name != 'Date':
#                         vix_raw.index.name = 'Date'
                    
#                     vix_temp = vix_raw.reset_index()
                    
#                     col_to_use = None
#                     if 'Close' in vix_temp.columns:
#                         col_to_use = 'Close'
#                     elif 'Adj Close' in vix_temp.columns:
#                         col_to_use = 'Adj Close'
                    
#                     if 'Date' in vix_temp.columns and col_to_use:
#                         vix_processed = vix_temp[['Date', col_to_use]].rename(columns={col_to_use: 'VIX_Close'})
#                         vix_processed['Date'] = pd.to_datetime(vix_processed['Date'])
#                         st.success("✅ VIX 데이터 로드 완료 (FinanceDataReader)!")
#                     else:
#                         st.warning("⚠️ VIX 데이터에 필요한 'Date' 또는 'Close'/'Adj Close' 컬럼이 없습니다. 예측에 포함되지 않습니다.")
#                         vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close'])
                    
#             except Exception as e:
#                 st.warning(f"⚠️ VIX 데이터 로드 중 오류 발생 (FinanceDataReader): {e}. 예측에 포함되지 않습니다.")
#                 vix_processed = pd.DataFrame(columns=['Date', 'VIX_Close'])
                
#             # ------------------------
#             # ✨ 모멘텀
#             # ------------------------
#             df_stock['Momentum'] = df_stock['Close'].diff()

#             # Date 컬럼 타입 통일
#             df_stock['Date'] = pd.to_datetime(df_stock['Date'])
#             vix_processed['Date'] = pd.to_datetime(vix_processed['Date'])
#             filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
            
#             # 뉴스 감성 점수를 일별 평균으로 그룹핑
#             filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
            
#             # 모든 데이터 병합
#             df_merge = pd.merge(df_stock, vix_processed, on='Date', how='left')
#             df_merge = pd.merge(df_merge, filtered_news_grouped, on='Date', how='left').fillna(0)

#             # ------------------------
#             # ✨ 회귀 예측
#             # ------------------------
#             # 예측에 사용할 특징(Feature)과 타겟(Target) 정의
#             # VIX_Close와 Sentiment_Score는 NaN이 있을 수 있으므로 fillna(0)으로 처리
#             X = df_merge[['Sentiment_Score', 'Momentum', 'VIX_Close']].fillna(0).values
#             y = df_merge['Close'].values

#             if len(X) > 5: # 최소한의 데이터가 있어야 회귀 분석 가능
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 y_pred = model.predict(X)
#                 df_merge['Predicted_Close'] = y_pred

#                 st.subheader("📊 주가 예측 결과")
#                 fig, ax = plt.subplots(figsize=(12, 6))
#                 ax.plot(df_merge['Date'], df_merge['Close'], label='Actual Close', color='blue')
#                 ax.plot(df_merge['Date'], df_merge['Predicted_Close'], label='Predicted Close', linestyle='--', color='red')
#                 ax.set_title(f"{company_name} Stock Prediction (NEWS + MOMENTUM + VIX)")
#                 ax.set_xlabel("Date")
#                 ax.set_ylabel("Close Price")
#                 ax.legend()
#                 ax.grid(True)
#                 plt.xticks(rotation=45)
#                 st.pyplot(fig)

#                 st.subheader("📈 회귀 모델 계수")
#                 st.metric("감성 점수 회귀계수", f"{model.coef_[0]:.2f}")
#                 st.metric("모멘텀 회귀계수", f"{model.coef_[1]:.2f}")
#                 st.metric("VIX 회귀계수", f"{model.coef_[2]:.2f}")
#             else:
#                 st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")

#         st.markdown("---")
#         st.write("👉 감성점수는 부정 뉴스에 -1, 긍정 뉴스에 1 점수를 대입합니다. 즉, -1(부정)~1(긍정)으로 점수가 계산됩니다.")
