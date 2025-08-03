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
from json.decoder import JSONDecodeError

# ------------------------
# ✨ 페이지 설정
# ------------------------
st.set_page_config(page_title="뉴스 감성분석 + 모멘텀 + VIX 전략", layout="wide")
st.title("뉴스 감성 분석 전략")

st.markdown("""
뉴스, 모멘텀, VIX(변동성 지수) 데이터를 결합하여
주식 또는 암호화폐 가격을 분석하고 예측하는 전략입니다.
""")

# ------------------------
# ✨ 감성 분석 모델 로드
# ------------------------
@st.cache_resource
def load_sentiment_model():
    # Streamlit Secrets에서 Hugging Face 토큰 가져오기
    hf_token = st.secrets.get("HF_TOKEN")

    model_name = "snunlp/KR-FinBert-SC"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='cpu')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        st.success(f"✅ 감성 분석 모델 : '{model_name}' (장치: {device})")
        st.write(f"모델 라벨 맵핑: {model.config.id2label}")
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
        st.stop()
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    if not text:
        return 0.0
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

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
# '주식' 또는 '암호화폐' 선택 라디오 버튼 추가
data_type = st.radio("분석할 자산 종류 선택", ("주식", "암호화폐"))

if data_type == "주식":
    # 주식 목록은 FinanceDataReader를 사용하여 가져옵니다.
    @st.cache_data
    def get_stock_list(market):
        try:
            df = fdr.StockListing(market)
            df['Code'] = df['Code'].astype(str).str.zfill(6)
            return df
        except JSONDecodeError as e:
            st.error(f"❌ 주식 종목 목록을 가져오는 중 오류가 발생했습니다: {e}")
            st.info("FinanceDataReader가 데이터를 가져오는 서버의 응답 형식이 올바르지 않은 것 같습니다. 잠시 후 다시 시도하거나, '암호화폐' 분석을 시도해보세요.")
            st.stop()
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ 주식 종목 목록을 가져오는 중 오류가 발생했습니다: {e}")
            st.stop()
            return pd.DataFrame()

    market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
    company_list_df = get_stock_list(market_option)
    company_names = company_list_df['Name'].tolist()
    default_company = "삼성전자" if "삼성전자" in company_names else company_names[0]
    
    if "selected_company" not in st.session_state:
        st.session_state.selected_company = default_company

    company_name = st.selectbox(
        "✅ 분석할 기업 선택",
        company_names,
        index=company_names.index(st.session_state.selected_company),
        key="selected_company"
    )

    stock_code = company_list_df.loc[company_list_df['Name'] == st.session_state.selected_company, 'Code'].values[0]

else: # 암호화폐 선택 시
    # 암호화폐 목록을 직접 정의합니다.
    crypto_list = {
        '비트코인 (Bitcoin)': 'BTC/KRW',
        '이더리움 (Ethereum)': 'ETH/KRW',
        '리플 (Ripple)': 'XRP/KRW',
        '월드코인 (Worldcoin)': 'WLD/KRW',
        '솔라나 (Solana)': 'SOL/KRW'
    }
    company_names = list(crypto_list.keys())
    default_crypto = "비트코인 (Bitcoin)"
    
    if "selected_company" not in st.session_state:
        st.session_state.selected_company = default_crypto

    company_name = st.selectbox(
        "✅ 분석할 암호화폐 선택",
        company_names,
        index=company_names.index(st.session_state.selected_company),
        key="selected_company"
    )

    stock_code = crypto_list.get(st.session_state.selected_company)

start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=30))
end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(query, display=30, start=1, sort="date"):
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError as e:
        st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
        st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
        return pd.DataFrame()

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
        # 뉴스 검색어를 'company_name'으로 설정
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
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 기업/암호화폐명을 확인해주세요.")
    else:
        filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)

        st.success("✅ 뉴스 감성 분석 완료!")
        st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False))

        # ------------------------
        # ✨ 자산 데이터 로드
        # ------------------------
        st.info(f"📈 {company_name} 가격 데이터를 로드 중입니다...")
        
        try:
            df_asset = fdr.DataReader(stock_code, start_date, end_date)
        except Exception as e:
            st.error(f"❌ 자산 가격 데이터를 가져오지 못했습니다: {e}")
            st.info("종목 코드나 날짜 범위를 확인하거나, '주식'의 경우 FinanceDataReader 서버에 문제가 있을 수 있습니다.")
            st.stop()
            
        if df_asset.empty:
            st.error("❌ 자산 가격 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
            st.stop()
        else:
            # 주식과 암호화폐 데이터 모두 'Close' 컬럼을 사용하도록 통일
            df_asset = df_asset.reset_index()[['Date', 'Close']]
            df_asset['Date'] = pd.to_datetime(df_asset['Date'])
            st.success("✅ 자산 가격 데이터 로드 완료!")

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
            df_asset['Momentum'] = df_asset['Close'].diff()

            df_asset['Date'] = pd.to_datetime(df_asset['Date'])
            vix_processed['Date'] = pd.to_datetime(vix_processed['Date'])
            filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
            
            filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
            
            df_merge = pd.merge(df_asset, vix_processed, on='Date', how='left')
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

                st.subheader("📊 예측 결과")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_merge['Date'], df_merge['Close'], label='실제 가격 (Actual)', color='blue')
                ax.plot(df_merge['Date'], df_merge['Predicted_Close'], label='예측 가격 (Predicted)', linestyle='--', color='red')
                ax.set_title(f"{company_name} 가격 예측 (뉴스 감성 + 모멘텀 + VIX)")
                ax.set_xlabel("날짜")
                ax.set_ylabel("종가")
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
