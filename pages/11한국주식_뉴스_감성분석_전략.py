import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import mean_squared_error, r2_score
import urllib.parse
from json.decoder import JSONDecodeError
import FinanceDataReader as fdr
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

# ------------------------
# ✨ 페이지 설정
# ------------------------
st.set_page_config(page_title="한국 주식 뉴스 감성 분석 전략", layout="wide")
st.title("📰 한국 주식 뉴스 감성 분석 전략")

st.markdown("""
네이버 뉴스를 크롤링하여 기술적 데이터와 결합,
주요 한국 상장 기업의 주가를 분석하고 예측하는 전략입니다.
""")

# ------------------------
# ✨ 감성 분석 모델 로드
# ------------------------
@st.cache_resource
def load_sentiment_model():
    """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
    # Hugging Face 토큰을 Streamlit secrets에서 불러옴
    hf_token = st.secrets.get("HF_TOKEN")
    model_name = "snunlp/KR-FinBert-SC"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        # GPU 사용을 위해 device_map='cpu' 대신 'auto' 사용
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
        
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
    """주어진 텍스트의 감성 점수를 계산합니다."""
    if not text:
        return 0.0
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    neg_idx = None
    pos_idx = None
    for idx, label in sentiment_model.config.id2label.items():
        if 'negative' in label.lower() or '부정' in label:
            neg_idx = idx
        elif 'positive' in label.lower() or '긍정' in label:
            pos_idx = idx
    
    negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

    sentiment_score = positive_score - negative_score
    
    return sentiment_score

# ------------------------
# ✨ 종목 목록 로드 (FinanceDataReader)
# ------------------------
@st.cache_data
def get_stock_list():
    """FinanceDataReader를 사용하여 KRX 상장 종목 목록을 가져옵니다."""
    try:
        df_krx = fdr.StockListing('KRX')
        df_krx = df_krx[~df_krx['Name'].str.contains('리츠|스팩|ETN|ETF|인버스|곱버스|레버리지|선물|상장지수|지수', case=False, na=False)]
        
        if df_krx.empty:
            st.error("❌ FinanceDataReader에서 종목 리스트를 가져오지 못했습니다.")
            st.stop()
            
        return df_krx
    except Exception as e:
        st.error(f"❌ 종목 리스트 로드 중 오류 발생: {e}")
        st.info("인터넷 연결 상태를 확인하거나 잠시 후 다시 시도해주세요.")
        st.stop()
        return pd.DataFrame()

df_krx = get_stock_list()
company_names = df_krx['Name'].tolist()

# ------------------------
# ✨ 주식 종목 선택 UI
# ------------------------
default_company = "삼성전자"
if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
    st.session_state.selected_company = default_company if default_company in company_names else company_names[0]

company_name = st.selectbox(
    "✅ 분석할 기업 선택",
    company_names,
    index=company_names.index(st.session_state.selected_company),
    key="selected_company"
)

stock_code = df_krx[df_krx['Name'] == company_name]['Code'].iloc[0]

# 날짜 선택 위젯
start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=90))
end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(query, display=30, start=1, sort="date"):
    """네이버 뉴스 검색 API를 호출하여 데이터를 가져옵니다."""
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
# ✨ 주가 데이터 로드 (FinanceDataReader)
# ------------------------
@st.cache_data
def get_stock_data(code, start_date, end_date):
    """
    FinanceDataReader를 통해 일별 주가 데이터를 가져와 DataFrame으로 반환합니다.
    """
    try:
        df = fdr.DataReader(code, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ 주가 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# ------------------------
# ✨ 실행 버튼
# ------------------------
max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=200, value=100, step=10)

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

        st.info(f"📈 {company_name} 주가 데이터를 로드 중입니다...")
        df_stock = get_stock_data(stock_code, start_date, end_date)
            
        if df_stock.empty:
            st.error("❌ 주가 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
            st.stop()
        else:
            st.success("✅ 주가 데이터 로드 완료 (FinanceDataReader)!")
            
            # 기술적 지표 추가
            df_stock['SMA_20'] = df_stock['Close'].rolling(window=20).mean()
            df_stock['Volatility'] = df_stock['Close'].pct_change().rolling(window=20).std()
            
            # 데이터 병합 및 결측치 처리
            df_stock['Date'] = pd.to_datetime(df_stock.index).date
            filtered_news['Date'] = pd.to_datetime(filtered_news['Date']).dt.date
            
            filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
            df_merge = pd.merge(df_stock.reset_index(drop=True), filtered_news_grouped, on='Date', how='left')
            df_merge = df_merge.set_index('Date').fillna(method='ffill').fillna(0) # 결측치는 이전 값으로 채움

            # 예측을 위한 특징 및 타겟 설정
            features = ['Close', 'Volume', 'Open', 'High', 'Low', 'Sentiment_Score', 'SMA_20', 'Volatility']
            features = [f for f in features if f in df_merge.columns]
            
            # 다음 날 수익률을 예측 타겟으로 설정
            df_merge['Next_Day_Return'] = df_merge['Close'].pct_change().shift(-1) * 100
            
            # 모델 학습에 사용할 데이터 준비
            df_ml = df_merge[features + ['Next_Day_Return']].dropna()

            if len(df_ml) > 100:
                X = df_ml[features].values
                y = df_ml['Next_Day_Return'].values
                
                # 데이터 정규화
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 학습/테스트 데이터 분리
                test_size = max(1, int(0.2 * len(X_scaled)))
                X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
                y_train, y_test = y[:-test_size], y[-test_size:]
                
                # LightGBM 모델 학습
                lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=500,
                                               learning_rate=0.05, num_leaves=31, max_depth=-1,
                                               random_state=42, n_jobs=-1, verbose=-1)
                
                lgbm_model.fit(X_train, y_train,
                               eval_set=[(X_test, y_test)],
                               callbacks=[lgb.early_stopping(100, verbose=False)])

                y_pred = lgbm_model.predict(X_test)
                
                st.subheader("📊 모델 성능 평가")
                st.write(f"**평균 제곱 오차 (MSE)**: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"**결정 계수 (R² Score)**: {r2_score(y_test, y_pred):.2f}")
                
                st.subheader("📈 예측 결과 시각화")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(y_test, label='실제 수익률', color='blue', marker='o', linestyle='None', alpha=0.6)
                ax.plot(y_pred, label='예측 수익률', color='red', marker='x', linestyle='None', alpha=0.6)
                ax.set_title(f"{company_name} ({stock_code}) LightGBM 예측 vs. 실제 수익률")
                ax.set_xlabel("데이터 포인트 인덱스")
                ax.set_ylabel("수익률(%)")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("---")
                st.subheader("💡 다음 날 주가 수익률 예측")
                
                # 마지막 데이터 포인트를 사용하여 다음 날 수익률 예측
                last_data = df_ml[features].iloc[-1].values.reshape(1, -1)
                last_data_scaled = scaler.transform(last_data)
                next_day_return_pred = lgbm_model.predict(last_data_scaled)[0]
                
                st.write(f"다음 영업일의 주가 수익률은 **{next_day_return_pred:.2f}%**로 예측됩니다.")
                
                if next_day_return_pred > 0:
                    st.success("예측 수익률이 긍정적입니다. 매수 신호로 고려해볼 수 있습니다.")
                else:
                    st.warning("예측 수익률이 부정적입니다. 매도 또는 관망 신호로 고려해볼 수 있습니다.")
            else:
                st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 최소 100개 이상의 데이터가 필요합니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")

        st.markdown("---")
        st.write("👉 **감성점수 계산 방식**: Hugging Face 모델에서 추출한 '긍정' 점수에서 '부정' 점수를 뺀 값입니다.")
