import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LinearRegression
import urllib.parse

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
    # Streamlit Secrets에서 Hugging Face 토큰 가져오기
    hf_token = st.secrets.get("HF_TOKEN")

    # 'snunlp/KR-FinBert-SC' 모델 사용
    model_name = "snunlp/KR-FinBert-SC"
    
    try:
        # 토큰을 from_pretrained 함수에 전달하여 인증
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
        
        # GPU 사용 가능 여부 확인 및 모델을 GPU로 이동
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        st.success(f"✅ 감성 분석 모델 '{model_name}' 로드 완료! (장치: {device})")
        st.write(f"모델 라벨 맵핑: {model.config.id2label}") # 라벨 맵핑 확인 필수!
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
        st.stop() # 모델 로드 실패 시 앱 중단
        return None, None, None # 이 부분은 실행되지 않지만, 명시적으로 None 반환

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

    # snunlp/KR-FinBert-SC 모델의 라벨 맵핑은 model.config.id2label 출력을 통해 정확히 확인해야 합니다.
    # 일반적으로 {0: 'neutral', 1: 'positive', 2: 'negative'} 또는 {0: 'negative', 1: 'neutral', 2: 'positive'}
    # 여기서는 라벨 이름을 기반으로 인덱스를 동적으로 찾습니다.
    
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
    
    # 인덱스가 None이 아닌지 확인하여 안전하게 확률을 가져옵니다.
    negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    neutral_score = probabilities[neu_idx].item() if neu_idx is not None else 0
    positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

    # (긍정 확률 - 부정 확률)을 사용하면 -1에서 1 사이의 값을 얻을 수 있습니다.
    sentiment_score = positive_score - negative_score 
    
    return sentiment_score

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
prediction_horizon = st.slider("주가 예측 기간 (미래 N일)", min_value=1, max_value=20, value=5, step=1)


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
        # 예측 기간을 고려하여 주가 데이터를 더 길게 가져옵니다.
        # 예를 들어, 오늘부터 N일 후를 예측하려면 N일 후까지의 실제 주가가 필요합니다.
        extended_end_date = end_date + timedelta(days=prediction_horizon + 7) # 여유분 추가
        df_stock = fdr.DataReader(stock_code, start_date, extended_end_date)
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
                # VIX 데이터도 예측 기간을 고려하여 더 길게 가져옵니다.
                vix_raw = fdr.DataReader('VIX', start=start_date - timedelta(days=60), end=extended_end_date + timedelta(days=1))
                
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

            # Date 컬럼 타입 통일
            df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            vix_processed['Date'] = pd.to_datetime(vix_processed['Date'])
            filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
            
            # 뉴스 감성 점수를 일별 평균으로 그룹핑
            filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
            
            # 모든 데이터 병합
            df_merge = pd.merge(df_stock, vix_processed, on='Date', how='left')
            df_merge = pd.merge(df_merge, filtered_news_grouped, on='Date', how='left')
            
            # 예측을 위한 NaN 처리: 예측에 사용될 특징들은 0으로 채우고, 예측 타겟은 NaN을 제거
            # 미래 주가 컬럼 생성: 'Close' 값을 prediction_horizon 만큼 위로 시프트 (미래 값을 가져옴)
            df_merge['Future_Close'] = df_merge['Close'].shift(-prediction_horizon)

            # 예측에 사용할 데이터프레임 (NaN 값 제거)
            # 예측에 필요한 모든 컬럼이 유효한 행만 선택
            df_pred_data = df_merge.dropna(subset=['Future_Close', 'Sentiment_Score', 'Momentum', 'VIX_Close']).copy()
            
            # ------------------------
            # ✨ 회귀 예측
            # ------------------------
            if len(df_pred_data) > 5: # 최소한의 데이터가 있어야 회귀 분석 가능
                # 특징(Feature)과 타겟(Target) 정의
                # X: 현재 날짜의 감성, 모멘텀, VIX
                # y: prediction_horizon 일 후의 종가
                X = df_pred_data[['Sentiment_Score', 'Momentum', 'VIX_Close']].values
                y = df_pred_data['Future_Close'].values

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                df_pred_data['Predicted_Future_Close'] = y_pred

                st.subheader(f"📊 {prediction_horizon}일 후 주가 예측 결과")
                fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
                ax_pred.plot(df_pred_data['Date'], df_pred_data['Future_Close'], label=f'Actual Close ({prediction_horizon} days later)', color='blue')
                ax_pred.plot(df_pred_data['Date'], df_pred_data['Predicted_Future_Close'], label=f'Predicted Close ({prediction_horizon} days later)', linestyle='--', color='red')
                ax_pred.set_title(f"{company_name} Stock Prediction ({prediction_horizon} days ahead)")
                ax_pred.set_xlabel("Date")
                ax_pred.set_ylabel("Close Price")
                ax_pred.legend()
                ax_pred.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig_pred)

                st.subheader("📈 회귀 모델 계수")
                st.metric("감성 점수 회귀계수", f"{model.coef_[0]:.2f}")
                st.metric("모멘텀 회귀계수", f"{model.coef_[1]:.2f}")
                st.metric("VIX 회귀계수", f"{model.coef_[2]:.2f}")
            else:
                st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")

        st.markdown("---")
        st.subheader("📰 일별 뉴스 감성 점수 변화")
        if not filtered_news_grouped.empty:
            fig_sentiment, ax_sentiment = plt.subplots(figsize=(12, 4))
            ax_sentiment.plot(filtered_news_grouped['Date'], filtered_news_grouped['Sentiment_Score'], label='Daily Avg Sentiment', color='green')
            ax_sentiment.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            ax_sentiment.set_title(f"{company_name} Daily Average News Sentiment Score")
            ax_sentiment.set_xlabel("Date")
            ax_sentiment.set_ylabel("Sentiment Score (-1 to 1)")
            ax_sentiment.legend()
            ax_sentiment.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig_sentiment)
        else:
            st.info("일별 감성 점수를 시각화할 뉴스 데이터가 없습니다.")

        st.markdown("---")
        st.write("👉 감성점수는 부정 뉴스에 -1, 긍정 뉴스에 1 점수를 대입합니다. 즉, -1(부정)~1(긍정)으로 점수가 계산됩니다.")
        st.write("""
        **참고:**
        - **미래 주가 예측:** 현재 날짜의 뉴스 감성, 모멘텀, VIX를 사용하여 설정된 '주가 예측 기간' 후의 종가를 예측합니다.
        - **회귀 모델의 한계:** 선형 회귀 모델은 복잡한 주가 변동을 완벽하게 예측하기 어렵습니다. 더 높은 정확도를 위해서는 LSTM, GRU, XGBoost 등 고급 예측 모델과 다양한 특징 공학(예: 감성 점수의 이동평균, VIX 변화율, 과거 주가 패턴 등)이 필요합니다.
        """)
