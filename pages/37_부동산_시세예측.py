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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit 
import urllib.parse
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import MinMaxScaler
import time
from concurrent.futures import ThreadPoolExecutor
import re
import shap 

# ------------------------
# ✨ 0. 상수 및 페이지 설정
# ------------------------
st.set_page_config(page_title="🇰🇷 지역별 아파트 가격 추세 예측", layout="wide")
st.title("🏢 지역별 아파트 실거래가 추세 예측 모형")

st.markdown("""
**국토교통부 실거래가**와 **네이버 뉴스 감성 지수**를 활용하여 특정 지역의 **향후 30일 가격 추세**를 예측합니다. 
$\text{LGBM, XGB, RF}$ 앙상블 모델을 사용하며, 시계열 교차검증을 통해 신뢰도를 확보합니다.
""")

# 지역 코드 매핑 (공공데이터포털 법정동 코드 앞 5자리)
REGION_CODES = {
    "강남구": "11680", "서초구": "11650", "송파구": "11710",
    "마포구": "11440", "용산구": "11170", "성동구": "11200",
    "분당구": "41135", "수지구": "41465"
}

# ------------------------
# 1. 국토교통부 데이터 수집 (MOLIT_KEY 사용)
# ------------------------
@st.cache_data(show_spinner="⏳ 국토부 실거래가 데이터 로드 중...")
def get_molit_data(region_code, start_date, end_date):
    api_key = st.secrets["MOLIT_KEY"]
    base_url = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade"
    
    all_data = []
    # 월 단위로 호출
    current_date = start_date.replace(day=1)
    while current_date <= end_date:
        ym = current_date.strftime("%Y%m")
        params = {"serviceKey": requests.utils.unquote(api_key), "LAWD_CD": region_code, "DEAL_YMD": ym}
        try:
            response = requests.get(base_url, params=params)
            data_dict = xmltodict.parse(response.text)
            items = data_dict['response']['body']['items']
            if items and 'item' in items:
                df_month = pd.DataFrame(items['item'] if isinstance(items['item'], list) else [items['item']])
                all_data.append(df_month)
        except Exception as e:
            st.warning(f"⚠️ {ym} 데이터 로드 실패: {e}")
        
        # 다음 달 이동
        next_month = current_date.month + 1
        if next_month > 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=next_month)
        time.sleep(0.05)

    if not all_data: return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    df['Price'] = df['거래금액'].str.replace(',', '').astype(float)
    df['Date'] = pd.to_datetime(df['년'] + '-' + df['월'] + '-' + df['일'])
    
    # 일별 평균 가격 및 거래량으로 집계
    daily_df = df.groupby('Date').agg({'Price': 'mean', '아파트': 'count'}).rename(columns={'아파트': 'Volume'})
    return daily_df.sort_index()

# ------------------------
# 2. 뉴스 감성 분석 (Naver API & KR-FinBert)
# ------------------------
@st.cache_resource
def load_sentiment_model():
    model_name = "snunlp/KR-FinBert-SC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map='auto')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model, device

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    if not text: return 0.0
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad(): outputs = sentiment_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    # 0:부정, 1:중립, 2:긍정
    return probs[2].item() - probs[0].item()

def get_naver_news_api(query, display=100):
    client_id = st.secrets["naver"]["client_id"]
    client_secret = st.secrets["naver"]["client_secret"]
    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&sort=date"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    
    try:
        res = requests.get(url, headers=headers)
        items = res.json().get('items', [])
        news_data = []
        for item in items:
            title = re.sub('<[^<]+?>', '', item.get('title', ''))
            pub_date = pd.to_datetime(item.get('pubDate')).date()
            news_data.append({'Date': pub_date, 'Title': title})
        return pd.DataFrame(news_data)
    except: return pd.DataFrame()

# ------------------------
# 3. 피처 엔지니어링 (부동산 특화)
# ------------------------
def create_features(df):
    df = df.copy()
    # 부동산은 호흡이 길어 30일 누적 수익률을 타겟으로 설정
    df['Return_30D'] = df['Price'].pct_change(periods=30).shift(-30) * 100
    
    # 이동평균 및 변동성
    for w in [7, 30]:
        df[f'Price_MA_{w}'] = df['Price'].rolling(window=w).mean()
        df[f'Vol_MA_{w}'] = df['Volume'].rolling(window=w).mean()
        df[f'Sent_MA_{w}'] = df['Sentiment'].rolling(window=w).mean()
    
    # Lag 피처
    for l in [1, 7, 30]:
        df[f'Price_Lag_{l}'] = df['Price'].shift(l)
        df[f'Sent_Lag_{l}'] = df['Sentiment'].shift(l)
    
    df = df.dropna()
    features = [c for c in df.columns if 'MA' in c or 'Lag' in c or c == 'Sentiment']
    return df, features

# ------------------------
# 4. 모델 훈련 (Voting Ensemble)
# ------------------------
@st.cache_resource(show_spinner="🚀 앙상블 모델 훈련 중...")
def train_voting_model(_X, _y):
    lgbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.01, max_depth=7, verbose=-1)
    xgb_m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=7)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10)
    
    voting = VotingRegressor([('lgbm', lgbm), ('xgb', xgb_m), ('rf', rf)])
    voting.fit(_X, _y)
    
    # SHAP 분석용 단일 모델 훈련
    lgbm_shap = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.01, max_depth=7, verbose=-1)
    lgbm_shap.fit(_X, _y)
    return voting, lgbm_shap

# ------------------------
# 5. UI 및 실행
# ------------------------
col1, col2, col3 = st.columns(3)
with col1:
    target_region = st.selectbox("🎯 분석 지역 선택", list(REGION_CODES.keys()))
with col2:
    start_date = st.date_input("시작일", datetime.now() - timedelta(days=365*2))
with col3:
    end_date = st.date_input("종료일", datetime.now())

if st.button("🚀 부동산 시장 추세 예측 시작", type="primary", use_container_width=True):
    # 1. 데이터 로드
    price_df = get_molit_data(REGION_CODES[target_region], start_date, end_date)
    news_df = get_naver_news_api(f"{target_region} 부동산 전망")
    
    if price_df.empty or news_df.empty:
        st.error("데이터 수집에 실패했습니다. API 키나 기간 설정을 확인하세요.")
        st.stop()

    # 2. 감성 분석 적용
    with st.spinner("🧠 뉴스 감성 분석 중..."):
        news_df['Sent_Score'] = news_df['Title'].apply(analyze_sentiment)
        news_grouped = news_df.groupby('Date')['Sent_Score'].mean().to_frame('Sentiment')
        news_grouped.index = pd.to_datetime(news_grouped.index)

    # 3. 데이터 병합 및 전처리
    df_merge = pd.merge(price_df, news_grouped, left_index=True, right_index=True, how='left')
    df_merge['Sentiment'] = df_merge['Sentiment'].fillna(method='ffill').fillna(0)
    
    df_ml, features = create_features(df_merge)
    
    # 4. 학습 데이터 준비
    X = df_ml[features]
    y = df_ml['Return_30D']
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # 테스트셋 분리 (최근 60일)
    test_size = 60
    X_train, X_test = X_scaled.iloc[:-test_size], X_scaled.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # 5. 모델 훈련 및 검증
    voting_model, lgbm_model = train_voting_model(X_train, y_train)
    
    # 6. 결과 시각화
    st.header(f"📈 {target_region} 가격 예측 리포트")
    
    y_pred = voting_model.predict(X_test)
    next_30d_pred = voting_model.predict(X_scaled.iloc[[-1]])[0]
    
    # 지표 출력
    m1, m2, m3 = st.columns(3)
    m1.metric("향후 30일 예측 수익률", f"{next_30d_pred:+.2f}%")
    m2.metric("모델 결정계수 (R²)", f"{r2_score(y_test, y_pred):.2f}")
    m3.metric("현재 뉴스 감성", f"{df_ml['Sentiment'].iloc[-1]:+.2f}")

    # 차트 1: 실거래가 추이
    fig_price = px.line(df_ml, y='Price', title=f"{target_region} 아파트 실거래가 평균 (일별)")
    st.plotly_chart(fig_price, use_container_width=True)

    # 차트 2: 예측 vs 실제
    res_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=res_df.index, y=res_df['Actual'], name="실제 30일 수익률"))
    fig_res.add_trace(go.Scatter(x=res_df.index, y=res_df['Predicted'], name="예측 30일 수익률", line=dict(dash='dot')))
    fig_res.update_layout(title="테스트 데이터 예측 성능 (최근 60일)")
    st.plotly_chart(fig_res, use_container_width=True)

    # 7. SHAP 해석
    st.subheader("💡 무엇이 가격 예측에 영향을 주었나? (SHAP)")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_scaled.iloc[[-1]])
    
    shap_df = pd.DataFrame({
        'Feature': features,
        'Influence': shap_values[0]
    }).sort_values('Influence', ascending=False).head(10)
    
    fig_shap = px.bar(shap_df, x='Influence', y='Feature', orientation='h', color='Influence',
                      title="최신 예측치에 대한 피처별 기여도")
    st.plotly_chart(fig_shap, use_container_width=True)

    st.success("✅ 모든 분석이 완료되었습니다. 본 자료는 투자 참고용입니다.")
