import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="뉴스 감성 기반 주가 예측", layout="wide")
st.title("📰 뉴스 감성 기반 주가 예측 데모")

# 1️⃣ 기업 선택
company_list = fdr.StockListing('KOSPI')
company_names = company_list['Name'].tolist()

company_name = st.sidebar.selectbox("기업을 선택하세요", company_names, index=company_names.index("삼성전자") if "삼성전자" in company_names else 0)
start_date = st.sidebar.date_input("시작일", datetime.now() - timedelta(days=60))
end_date = st.sidebar.date_input("종료일", datetime.now())

# 코드 가져오기
code = company_list.loc[company_list['Name'] == company_name, 'Code'].values[0]

# 2️⃣ 주가 데이터 수집
df_price = fdr.DataReader(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
df_price.reset_index(inplace=True)

# 3️⃣ 뉴스 감성 분석 준비
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base")
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

def analyze_sentiment(text):
    if not text:
        return 0.0

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return (score - 0.5) * 2  # -1 ~ 1

# 4️⃣ 뉴스 크롤링 함수 (샘플: 감성 점수 랜덤 생성 예제)
# 실제로는 네이버 뉴스 크롤링 함수 사용 가능
def get_dummy_news_sentiment(start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq='B')
    scores = np.random.uniform(-0.5, 0.5, size=len(dates))
    df_news = pd.DataFrame({'Date': dates, 'Sentiment_Score': scores})
    return df_news

df_news = get_dummy_news_sentiment(start_date, end_date)

# 5️⃣ 데이터 병합
merged_df = pd.merge(df_price, df_news, on='Date', how='left')
merged_df['Sentiment_Score'].fillna(0, inplace=True)

# 6️⃣ 피처 엔지니어링
merged_df['MA5'] = merged_df['Close'].rolling(window=5).mean()
merged_df['Volatility'] = merged_df['Close'].rolling(window=5).std()
merged_df.dropna(inplace=True)

# 7️⃣ 학습 데이터 준비
X = merged_df[['Sentiment_Score', 'MA5', 'Volatility']]
y = merged_df['Close']

# 8️⃣ 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X[:-5], y[:-5])  # 마지막 5개는 테스트용

# 9️⃣ 예측 및 평가
y_pred = model.predict(X[-5:])
mse = mean_squared_error(y[-5:], y_pred)

# 10️⃣ 결과 출력
st.subheader(f"📈 {company_name} 주가 및 감성 점수")
fig, ax1 = plt.subplots(figsize=(12,5))

ax1.plot(merged_df['Date'], merged_df['Close'], color='blue', label='종가')
ax1.set_ylabel("종가")
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(merged_df['Date'], merged_df['Sentiment_Score'], color='orange', linestyle='--', label='감성 점수')
ax2.set_ylabel("감성 점수")
ax2.legend(loc='upper right')

st.pyplot(fig)

st.subheader("💡 예측 결과")
st.write("예측된 마지막 5일 종가:", y_pred.round(2).tolist())
st.write("실제 마지막 5일 종가:", y[-5:].values.round(2).tolist())
st.metric("테스트 MSE", f"{mse:.2f}")

st.info("""
- **감성 점수**는 샘플용 더미로 랜덤 생성했습니다.  
- 실제 뉴스 제목으로 분석하려면 `get_naver_news_with_sentiment()` 함수를 실제 네이버 뉴스 크롤링 로직과 연결하면 됩니다.
- 피처 엔지니어링 및 모델은 자유롭게 변경 가능 (LSTM, XGBoost 등).
""")
