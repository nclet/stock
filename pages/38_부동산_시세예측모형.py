import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import lightgbm as lgb
from sklearn.metrics import r2_score
import urllib.parse

# ------------------------------------------------------------------------------
# 1. 설정 및 상수 정의
# ------------------------------------------------------------------------------
st.set_page_config(page_title="🏠 서울 아파트 실거래가 예측", layout="wide")
st.title("🏠 서울 아파트 실거래가 예측 모델 (Seoul Data API + ML)")

# 서울시 행정구역 목록 (서울시 API용)
SEOUL_DISTRICTS = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"
]

# ------------------------------------------------------------------------------
# 2. 데이터 수집 함수 (서울시 API용으로 전면 수정)
# ------------------------------------------------------------------------------

@st.cache_resource
def load_sentiment_model():
    model_name = "snunlp/KR-FinBert-SC"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except:
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment_score(text):
    if not text or not sentiment_model: return 0.0
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            neg_prob = probs[0][0].item()
            pos_prob = probs[0][2].item()
        return pos_prob - neg_prob
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_naver_news_sentiment(query):
    client_id = st.secrets.get("NAVER_CLIENT_ID")
    client_secret = st.secrets.get("NAVER_CLIENT_SECRET")
    
    if not client_id or not client_secret: return pd.DataFrame()

    all_news = []
    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display=100&sort=date"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            items = res.json().get('items', [])
            for item in items:
                pub_date = datetime.strptime(item['pubDate'], "%a, %d %b %Y %H:%M:%S %z")
                all_news.append({'Date': pub_date.date(), 'Title': item['title']})
    except: pass
            
    if not all_news: return pd.DataFrame()
    df_news = pd.DataFrame(all_news)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    df_news['Score'] = df_news['Title'].apply(analyze_sentiment_score)
    return df_news.set_index('Date')['Score'].resample('MS').mean().to_frame(name='News_Sentiment')

@st.cache_data(ttl=86400)
def get_seoul_apt_data(api_key, district, dong=""):
    """서울시 열린데이터 광장 API를 사용한 아파트 실거래가 수집"""
    if not api_key:
        return pd.DataFrame()

    # 서울시 아파트 매매 실거래가 정보 API
    # http://openapi.seoul.go.kr:8088/(인증키)/json/tbLnOpendataRtmsV/1/1000/
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/tbLnOpendataRtmsV/1/1000/"
    
    all_data = []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'tbLnOpendataRtmsV' in data:
                items = data['tbLnOpendataRtmsV']['row']
                for item in items:
                    # 아파트만 필터링 및 지역 필터링
                    if item['BJDONG_NM'] == dong or not dong:
                        if item['SGG_NM'] == district and "아파트" in item['BLDG_NM']:
                            try:
                                deal_date = datetime.strptime(item['DEAL_YMD'], "%Y%m%d")
                                all_data.append({
                                    'Date': deal_date,
                                    'Dong': item['BJDONG_NM'],
                                    'Price': float(item['OBJ_AMT']), # 만원 단위
                                    'Area': float(item['BLDG_AREA']), # 제곱미터
                                    'Name': item['BLDG_NM']
                                })
                            except: continue
    except Exception as e:
        st.error(f"API 호출 중 오류 발생: {e}")
        
    return pd.DataFrame(all_data)

# ------------------------------------------------------------------------------
# 3. 모델링 로직
# ------------------------------------------------------------------------------
def process_and_train(apt_df, sentiment_df):
    if apt_df.empty: return None
    
    df = apt_df.copy()
    df.set_index('Date', inplace=True)
    
    # 월별 데이터 집계
    monthly = df.resample('MS').agg({'Price': 'mean', 'Area': 'mean', 'Name': 'count'}).rename(columns={'Name': 'Volume'})
    monthly['Price_Per_Area'] = monthly['Price'] / monthly['Area']
    
    # 뉴스 심리 지수 통합
    if not sentiment_df.empty:
        monthly = monthly.join(sentiment_df, how='left').fillna(method='ffill').fillna(0)
    else:
        monthly['News_Sentiment'] = 0.0
        
    # 특성 공학
    monthly['Price_Lag_1'] = monthly['Price_Per_Area'].shift(1)
    monthly['Price_MA_3M'] = monthly['Price_Per_Area'].rolling(window=3).mean()
    monthly['Target'] = monthly['Price_Per_Area'].shift(-1) # 다음달 가격 예측
    
    monthly.dropna(inplace=True)
    
    if len(monthly) < 5: return None
    
    features = ['Price_Lag_1', 'Price_MA_3M', 'Volume', 'News_Sentiment']
    X = monthly[features]
    y = monthly['Target']
    
    # 학습/검증 분할
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    score = r2_score(y_test, preds) if len(y_test) > 1 else 0
    
    return monthly, model, features, X_test, y_test, preds, score

# ------------------------------------------------------------------------------
# 4. 메인 실행부
# ------------------------------------------------------------------------------
def app():
    st.sidebar.info(f"API Key Load Status: {'✅' if st.secrets.get('MOLIT_KEY') else '❌'}")
    
    st.write("### 🔍 서울시 지역 선택")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sgg_name = st.selectbox("구 선택", SEOUL_DISTRICTS)
    with col_sel2:
        dong_name = st.text_input("동 입력 (예: 개포동, 미입력 시 전체)", "")

    if st.button("분석 시작", type="primary"):
        api_key = st.secrets.get("MOLIT_KEY")
        
        with st.spinner("서울시 실거래 데이터를 수집 중..."):
            apt_df = get_seoul_apt_data(api_key, sgg_name, dong_name)
            
            if apt_df.empty:
                st.error("데이터를 가져오지 못했습니다. API 키가 서울시 전용인지, 혹은 지역명이 정확한지 확인하세요.")
                return

            sentiment_df = get_naver_news_sentiment(f"{sgg_name} {dong_name} 부동산 전망")
            result = process_and_train(apt_df, sentiment_df)
            
            if result:
                monthly_df, model, features, X_test, y_test, preds, score = result
                
                # 시각화
                st.write(f"#### 📊 {sgg_name} {dong_name} 분석 리포트")
                
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.line(monthly_df, y='Price_Per_Area', title="월별 평균 평당가 변화 (만원/㎡)")
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    last_val = monthly_df['Price_Per_Area'].iloc[-1]
                    last_feat = monthly_df[features].iloc[[-1]]
                    next_pred = model.predict(last_feat)[0]
                    diff = next_pred - last_val
                    
                    st.metric("현재 평당가", f"{last_val:,.0f} 만원")
                    st.metric("다음 달 예측가", f"{next_pred:,.0f} 만원", f"{diff:+.2f} 만원")
                    st.caption(f"모델 신뢰도 (R2): {score:.4f}")

                st.write("#### 💡 AI 변수 중요도")
                imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')
                st.plotly_chart(px.bar(imp_df, x='Importance', y='Feature', orientation='h'), use_container_width=True)
                
            else:
                st.warning("분석을 위한 충분한 시계열 데이터(최소 5개월 이상)가 확보되지 않았습니다.")

if __name__ == "__main__":
    app()
