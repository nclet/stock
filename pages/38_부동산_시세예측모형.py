import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# 올바른 라이브러리 임포트 (lightgbm)
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import urllib.parse
import re

# ------------------------------------------------------------------------------
# 1. 설정 및 상수 정의
# ------------------------------------------------------------------------------
st.set_page_config(page_title="🏠 국내 아파트 실거래가 예측", layout="wide")
st.title("🏠 국내 아파트 실거래가 예측 모델 (AI Sentiment + ML)")

st.markdown("""
**국토교통부 실거래가 데이터**와 **네이버 뉴스 감성 분석**을 결합하여 
특정 지역 아파트의 향후 가격 추세를 예측합니다.
""")

# 법정동 코드 매핑
DISTRICT_CODES = {
    "서울 강남구 개포동": "1168010300",
    "서울 강남구 대치동": "1168010600",
    "서울 서초구 반포동": "1165010700",
    "서울 송파구 잠실동": "1171010100",
    "서울 마포구 아현동": "1144010100",
    "성남 분당구 정자동": "4113510300",
    "대구 수성구 범어동": "2726010100",
    "부산 해운대구 우동": "2635010500"
}

# ------------------------------------------------------------------------------
# 2. 데이터 수집 함수 (국토교통부 & 네이버)
# ------------------------------------------------------------------------------

@st.cache_resource
def load_sentiment_model():
    """Hugging Face 감성 분석 모델 로드"""
    model_name = "snunlp/KR-FinBert-SC"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment_score(text):
    """텍스트 감성 점수 계산 (-1 ~ 1)"""
    if not text or not sentiment_model: return 0.0
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # KR-FinBert-SC: 0:Negative, 1:Neutral, 2:Positive
            neg_prob = probs[0][0].item()
            pos_prob = probs[0][2].item()
        return pos_prob - neg_prob
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_naver_news_sentiment(query, start_date, end_date):
    """네이버 뉴스 수집 및 감성 지수 집계"""
    # Secrets 접근 방식 수정 (안전한 접근을 위해 get 사용 권장)
    try:
        client_id = st.secrets.get("NAVER_CLIENT_ID")
        client_secret = st.secrets.get("NAVER_CLIENT_SECRET")
        if not client_id or not client_secret:
             # 하위 경로 구조인 경우 대응
             client_id = st.secrets["naver"]["client_id"]
             client_secret = st.secrets["naver"]["client_secret"]
    except Exception:
        st.error("Secrets에 네이버 API 키가 없습니다.")
        return pd.DataFrame()

    all_news = []
    enc_query = urllib.parse.quote(query)
    
    for start_idx in range(1, 401, 100): # 약 400개 샘플링
        url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display=100&start={start_idx}&sort=date"
        headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
        
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: break
            items = res.json().get('items', [])
            if not items: break
            
            for item in items:
                pub_date_str = item['pubDate']
                pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
                all_news.append({
                    'Date': pub_date.date(),
                    'Title': item['title'].replace('<b>', '').replace('</b>', '')
                })
        except:
            break
            
    if not all_news: return pd.DataFrame()
    
    df_news = pd.DataFrame(all_news)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    df_news = df_news[(df_news['Date'].dt.date >= start_date) & (df_news['Date'].dt.date <= end_date)]
    
    if df_news.empty: return pd.DataFrame()

    # 감성 분석 수행
    df_news['Score'] = df_news['Title'].apply(analyze_sentiment_score)
    monthly_sentiment = df_news.set_index('Date')['Score'].resample('M').mean()
    monthly_sentiment.index = monthly_sentiment.index.to_period('M').to_timestamp()
    
    return monthly_sentiment.to_frame(name='News_Sentiment')

@st.cache_data(ttl=86400)
def get_molit_apt_data(lawd_cd, start_date_str, end_date_str):
    """국토교통부 아파트 실거래가 API 호출"""
    # Secrets 접근 방식 수정
    try:
        service_key = st.secrets.get("MOLIT_KEY")
        if not service_key:
            # 하위 경로 구조인 경우 대응
            service_key = st.secrets["data"]["MOLIT_KEY"]
    except Exception:
        st.error("Secrets에 MOLIT_KEY가 없습니다.")
        return pd.DataFrame()

    url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev"
    start_dt = datetime.strptime(start_date_str, "%Y%m")
    end_dt = datetime.strptime(end_date_str, "%Y%m")
    
    current_dt = start_dt
    all_data = []
    
    while current_dt <= end_dt:
        params = {
            'serviceKey': service_key,
            'LAWD_CD': lawd_cd[:5],
            'DEAL_YMD': current_dt.strftime("%Y%m"),
            'numOfRows': '1000'
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('.//item'):
                    try:
                        all_data.append({
                            'Date': datetime(int(item.find('년').text), int(item.find('월').text), int(item.find('일').text)),
                            'Dong': item.find('법정동').text.strip(),
                            'Price': int(item.find('거래금액').text.replace(',', '').strip()),
                            'Area': float(item.find('전용면적').text)
                        })
                    except: continue
        except: pass
        current_dt += relativedelta(months=1)
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df['Price_Per_Area'] = df['Price'] / df['Area']
    return df

# ------------------------------------------------------------------------------
# 3. 데이터 전처리 및 모델 훈련
# ------------------------------------------------------------------------------
def process_and_train(apt_df, dong_name, sentiment_df):
    df = apt_df[apt_df['Dong'].str.contains(dong_name)].copy()
    if df.empty: return None
    
    # 월별 리샘플링
    df.set_index('Date', inplace=True)
    monthly = df.resample('M').agg({'Price_Per_Area': 'mean', 'Dong': 'count'}).rename(columns={'Dong': 'Volume'})
    monthly.index = monthly.index.to_period('M').to_timestamp()
    
    # 데이터 병합
    if not sentiment_df.empty:
        monthly = monthly.join(sentiment_df, how='left').fillna(0)
    else:
        monthly['News_Sentiment'] = 0.0
        
    # 기술적 지표
    monthly['Price_Lag_1'] = monthly['Price_Per_Area'].shift(1)
    monthly['Price_MA_3M'] = monthly['Price_Per_Area'].rolling(3).mean()
    monthly['Target'] = monthly['Price_Per_Area'].shift(-1)
    monthly.dropna(inplace=True)
    
    if len(monthly) < 10: return None
    
    # 모델 학습
    features = ['Price_Lag_1', 'Price_MA_3M', 'Volume', 'News_Sentiment']
    X = monthly[features]
    y = monthly['Target']
    
    # 시계열 분할 학습
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    
    return monthly, model, features, X_test, y_test, preds, score

# ------------------------------------------------------------------------------
# 4. 앱 UI
# ------------------------------------------------------------------------------
def app():
    st.write("### 🔍 분석 설정")
    region_name = st.selectbox("분석 대상 지역 선택", list(DISTRICT_CODES.keys()))
    lawd_cd = DISTRICT_CODES[region_name]
    dong_name = region_name.split()[-1]
    
    if st.button("분석 및 예측 시작", type="primary"):
        with st.spinner("데이터 수집 및 분석 중..."):
            end_date = datetime.now()
            start_date = end_date - relativedelta(years=3)
            
            # 데이터 수집
            apt_df = get_molit_apt_data(lawd_cd, start_date.strftime("%Y%m"), end_date.strftime("%Y%m"))
            sentiment_df = get_naver_news_sentiment(f"{dong_name} 부동산 전망", start_date.date(), end_date.date())
            
            # 분석 및 훈련
            result = process_and_train(apt_df, dong_name, sentiment_df)
            
            if result:
                monthly_df, model, features, X_test, y_test, preds, score = result
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"#### {dong_name} 평당가 추이")
                    fig = px.line(monthly_df, y='Price_Per_Area', title='월평균 평당 실거래가(만원)')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("#### 모델 성능 및 예측")
                    st.metric("모델 R2 Score", f"{score:.2f}")
                    
                    # 다음 달 예측
                    last_features = monthly_df[features].iloc[[-1]]
                    next_pred = model.predict(last_features)[0]
                    current = monthly_df['Price_Per_Area'].iloc[-1]
                    delta = (next_pred - current) / current * 100
                    
                    st.metric("다음 달 예상 평당가", f"{next_pred:,.0f} 만원", f"{delta+.2f}%")
                
                # 변수 중요도
                imp = pd.DataFrame({'Feature': features, 'Value': model.feature_importances_}).sort_values('Value', ascending=False)
                st.write("#### 주요 영향 요인")
                st.bar_chart(imp.set_index('Feature'))
                
            else:
                st.warning("분석을 위한 충분한 데이터를 확보하지 못했습니다.")

if __name__ == "__main__":
    app()
