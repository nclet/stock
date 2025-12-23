import streamlit as st
import pandas as pd
import numpy as np
import requests
import xmltodict
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import urllib.parse
import re
import time

# ------------------------
# ✨ 설정 및 상수
# ------------------------
st.set_page_config(page_title="🇰🇷 지역별 아파트 가격 추세 예측", layout="wide")
st.title("🏢 한국 아파트 실거래가 추세 예측 모델")

# 지역 코드 사전 (예시: 서울 주요 지역)
REGION_CODES = {
    "서울 강남구": "11680",
    "서울 서초구": "11650",
    "서울 송파구": "11710",
    "서울 마포구": "11440",
    "서울 용산구": "11170",
    "경기 성남 분당구": "41135"
}

# 뉴스 분석 키워드
REAL_ESTATE_KEYWORDS = ['부동산', '아파트', '매매', '금리', '공급', '전세', '하락', '상승', '재건축', '대출 규제']

# ------------------------
# 1. 국토교통부 실거래가 API 데이터 로드
# ------------------------
@st.cache_data(show_spinner="⏳ 국토교통부 실거래가 데이터를 가져오는 중...")
def get_molit_data(region_code, start_date, end_date, api_key):
    """국토부 API에서 실거래 데이터를 수집하여 DF로 반환"""
    base_url = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade"
    
    all_trades = []
    # API는 월 단위로만 요청 가능하므로 반복문 실행
    current_date = start_date
    while current_date <= end_date:
        ym = current_date.strftime("%Y%m")
        params = {
            "serviceKey": requests.utils.unquote(api_key),
            "LAWD_CD": region_code,
            "DEAL_YMD": ym
        }
        try:
            response = requests.get(base_url, params=params)
            data_dict = xmltodict.parse(response.text)
            items = data_dict['response']['body']['items']
            
            if items and 'item' in items:
                df_month = pd.DataFrame(items['item'])
                all_trades.append(df_month)
        except Exception as e:
            st.error(f"❌ {ym} 데이터 호출 오류: {e}")
        
        # 다음 달로 이동
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
        time.sleep(0.1)

    if not all_trades:
        return pd.DataFrame()

    df = pd.concat(all_trades, ignore_index=True)
    
    # 데이터 전처리
    df['거래금액'] = df['거래금액'].str.replace(',', '').astype(float)
    df['Date'] = pd.to_datetime(df['년'] + '-' + df['월'] + '-' + df['일'])
    df = df.sort_values('Date')
    
    # 일별 평균 가격 및 거래량 계산
    daily_df = df.groupby('Date').agg({'거래금액': 'mean', '아파트': 'count'}).rename(columns={'거래금액': 'Price', '아파트': 'Volume'})
    return daily_df

# ------------------------
# 2. 감성 분석 모델 (KR-FinBert 활용)
# ------------------------
@st.cache_resource
def load_sentiment_model():
    model_name = "snunlp/KR-FinBert-SC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    if not text: return 0.0
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    # KR-FinBert labels: 0: negative, 1: neutral, 2: positive
    return (probs[2].item() - probs[0].item())

# ------------------------
# 3. 네이버 뉴스 API (부동산 뉴스 수집)
# ------------------------
def get_naver_news_sentiment(query, start_date, end_date):
    client_id = st.secrets["naver"]["client_id"]
    client_secret = st.secrets["naver"]["client_secret"]
    
    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display=100&sort=date"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    
    try:
        response = requests.get(url, headers=headers)
        items = response.json().get('items', [])
        news_data = []
        for item in items:
            title = re.sub('<[^<]+?>', '', item.get('title', ''))
            pub_date = pd.to_datetime(item.get('pubDate')).date()
            sentiment = analyze_sentiment(title)
            news_data.append({'Date': pub_date, 'Sentiment': sentiment})
        
        news_df = pd.DataFrame(news_data)
        if news_df.empty: return pd.DataFrame()
        
        # 일별 평균 감성지수
        news_df['Date'] = pd.to_datetime(news_df['Date'])
        return news_df.groupby('Date')['Sentiment'].mean()
    except:
        return pd.DataFrame()

# ------------------------
# 4. 피처 엔지니어링
# ------------------------
def create_real_estate_features(df):
    df = df.copy()
    # 가격 변동성 및 이동평균
    for w in [7, 30]:
        df[f'Price_MA_{w}'] = df['Price'].rolling(window=w, min_periods=1).mean()
        df[f'Vol_MA_{w}'] = df['Volume'].rolling(window=w, min_periods=1).mean()
    
    # 과거 수익률 (Lag)
    df['Return_1D'] = df['Price'].pct_change()
    for l in [1, 3, 7]:
        df[f'Price_Lag_{l}'] = df['Price'].shift(l)
        df[f'Sent_Lag_{l}'] = df['Sentiment'].shift(l)
        
    # 타겟: 향후 30일 뒤의 가격 변동률 예측 (부동산은 흐름이 느림)
    df['Target_Next_30D'] = df['Price'].pct_change(periods=30).shift(-30) * 100
    return df.dropna()

# ------------------------
# 5. UI 및 실행 로직
# ------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    molit_api_key = st.text_input("국토부 API Key", type="password")
    selected_region = st.selectbox("예측 대상 지역", list(REGION_CODES.keys()))
    analysis_period = st.slider("분석 기간 (개월)", 6, 36, 12)
    
    st.markdown("---")
    st.info("실거래 데이터는 매매 신고일 기준이므로 실제 시장 상황과 1~2개월 시차가 발생할 수 있습니다.")

if st.button("🚀 부동산 시장 분석 및 예측 시작", use_container_width=True):
    if not molit_api_key:
        st.warning("API 키를 입력해주세요.")
        st.stop()
        
    # 날짜 설정
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=analysis_period * 30)
    
    # 1. 데이터 수집
    with st.spinner("🏢 실거래가 및 뉴스 데이터 수집 중..."):
        price_df = get_molit_data(REGION_CODES[selected_region], start_dt, end_dt, molit_api_key)
        news_sent = get_naver_news_sentiment(f"{selected_region} 아파트 전망", start_dt, end_dt)
    
    if price_df.empty:
        st.error("해당 기간에 실거래 데이터가 없습니다.")
        st.stop()

    # 2. 데이터 병합
    full_df = pd.merge(price_df, news_sent, left_index=True, right_index=True, how='left').fillna(0)
    
    # 3. 피처 생성
    ml_df = create_real_estate_features(full_df)
    
    # 4. 모델링 (LightGBM)
    features = [col for col in ml_df.columns if 'MA' in col or 'Lag' in col or 'Sentiment' == col]
    X = ml_df[features]
    y = ml_df['Target_Next_30D']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_scaled, y)
    
    # 5. 결과 시각화
    st.header(f"📊 {selected_region} 분석 보고서")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_price = px.line(ml_df, y='Price', title="실거래가 평균 추이")
        st.plotly_chart(fig_price, use_container_width=True)
    with col2:
        fig_sent = px.bar(ml_df, y='Sentiment', title="뉴스 감성지수 (긍부정)")
        st.plotly_chart(fig_sent, use_container_width=True)
        
    # 예측 결과
    latest_x = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(latest_x)[0]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="향후 30일 가격 예측 추세", value=f"{prediction:+.2f}%", 
                  delta="상승 전망" if prediction > 0 else "하락 전망")
    
    with res_col2:
        imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).head(5)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="예측 주요 요인")
        st.plotly_chart(fig_imp, use_container_width=True)

    st.success("✅ 분석이 완료되었습니다. 위 결과는 참고용이며 실제 투자 손실에 책임지지 않습니다.")

# --- 차트 추가: 실거래가 vs 뉴스 감성 상관관계 ---
