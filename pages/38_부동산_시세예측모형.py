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
import lgbm as lgb
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

# 감성 분석 키워드 (부동산 시장 특화)
POS_KEYWORDS = ['상승', '급등', '호재', '개발', 'GTX', '재건축', '완화', '회복', '신고가', '매수', '공급부족']
NEG_KEYWORDS = ['하락', '급락', '폭락', '규제', '미분양', '금리인상', '역전세', '깡통', '매도', '관망', '침체']

# 법정동 코드 매핑 (주요 지역 예시)
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
    except Exception:
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
            neg_prob = probs[0][0].item()
            pos_prob = probs[0][2].item()
        return pos_prob - neg_prob
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_naver_news_sentiment(query, start_date, end_date):
    """네이버 뉴스 수집 및 감성 지수 집계 (월별)"""
    try:
        # secrets 구조에 맞춰 수정
        client_id = st.secrets.get("NAVER_CLIENT_ID") or st.secrets.get("naver", {}).get("client_id")
        client_secret = st.secrets.get("NAVER_CLIENT_SECRET") or st.secrets.get("naver", {}).get("client_secret")
        
        if not client_id or not client_secret:
            st.warning("네이버 API 키가 설정되지 않아 뉴스 분석을 건너뜁니다.")
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    all_news = []
    enc_query = urllib.parse.quote(query)
    
    for start_idx in range(1, 500, 100): 
        url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display=100&start={start_idx}&sort=date"
        headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
        
        try:
            res = requests.get(url, headers=headers)
            if res.status_code != 200: break
            items = res.json().get('items', [])
            if not items: break
            
            for item in items:
                pub_date_str = item['pubDate']
                pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
                all_news.append({
                    'Date': pub_date.date(),
                    'Title': item['title'],
                    'Description': item['description']
                })
        except:
            break
            
    if not all_news:
        return pd.DataFrame()
    
    df_news = pd.DataFrame(all_news)
    df_news = df_news[(df_news['Date'] >= start_date) & (df_news['Date'] <= end_date)]
    
    if df_news.empty:
        return pd.DataFrame()

    st.info(f"뉴스 {len(df_news)}건 감성 분석 중...")
    df_news['Score'] = df_news['Title'].apply(lambda x: analyze_sentiment_score(x))
    
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    df_news.set_index('Date', inplace=True)
    monthly_sentiment = df_news['Score'].resample('M').mean()
    monthly_sentiment.index = monthly_sentiment.index.to_period('M').to_timestamp()
    
    return monthly_sentiment.to_frame(name='News_Sentiment')

@st.cache_data(ttl=3600 * 24)
def get_molit_apt_data(lawd_cd, start_date_str, end_date_str):
    """국토교통부 아파트 실거래가 API 호출"""
    try:
        # 사용자 설정에 맞춰 st.secrets["MOLIT_KEY"]로 직접 접근
        service_key = st.secrets.get("MOLIT_KEY")
        if not service_key:
            st.error("Secrets에 'MOLIT_KEY'가 없습니다. .streamlit/secrets.toml 파일을 확인하세요.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Secrets 로드 오류: {e}")
        return pd.DataFrame()

    url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev"
    
    start_dt = datetime.strptime(start_date_str, "%Y%m")
    end_dt = datetime.strptime(end_date_str, "%Y%m")
    
    current_dt = start_dt
    all_data = []
    
    # 진행 상황 표시용
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
    progress_bar = st.progress(0)
    idx = 0
    
    while current_dt <= end_dt:
        ym = current_dt.strftime("%Y%m")
        params = {
            'serviceKey': service_key,
            'LAWD_CD': lawd_cd[:5],
            'DEAL_YMD': ym,
            'numOfRows': '1000'
        }
        
        try:
            # API 호출 시 디코딩된 키가 필요할 수 있으므로 주의
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                # API 응답 결과 코드 확인
                result_code = root.find('.//resultCode')
                if result_code is not None and result_code.text != '00':
                    msg = root.find('.//resultMsg').text
                    st.warning(f"{ym} API 응답 오류: {msg}")
                
                items = root.findall('.//item')
                for item in items:
                    dong = item.find('법정동').text.strip()
                    try:
                        deal_amount = int(item.find('거래금액').text.replace(',', '').strip())
                        area = float(item.find('전용면적').text)
                        year = int(item.find('년').text)
                        month = int(item.find('월').text)
                        day = int(item.find('일').text)
                        date = datetime(year, month, day)
                        
                        all_data.append({
                            'Date': date,
                            'Dong': dong,
                            'Price': deal_amount,
                            'Area': area,
                            'Price_Per_Area': deal_amount / area
                        })
                    except:
                        continue
        except Exception as e:
            pass
        
        current_dt += relativedelta(months=1)
        idx += 1
        progress_bar.progress(min(idx / total_months, 1.0))
        
    progress_bar.empty()
    return pd.DataFrame(all_data)

# ------------------------------------------------------------------------------
# 3. 데이터 전처리 및 모델링 (생략 없이 이전 로직 유지)
# ------------------------------------------------------------------------------
def preprocess_real_estate_data(df, target_dong_name, sentiment_df):
    if df.empty: return pd.DataFrame()
    df = df[df['Dong'].str.contains(target_dong_name)].copy()
    if df.empty: return pd.DataFrame()
    
    q_low = df['Price_Per_Area'].quantile(0.01)
    q_high = df['Price_Per_Area'].quantile(0.99)
    df = df[(df['Price_Per_Area'] >= q_low) & (df['Price_Per_Area'] <= q_high)]

    df.set_index('Date', inplace=True)
    monthly_df = df.resample('M').agg({
        'Price_Per_Area': 'mean',
        'Price': 'mean',
        'Dong': 'count'
    }).rename(columns={'Dong': 'Volume'})
    
    monthly_df.index = monthly_df.index.to_period('M').to_timestamp()
    
    if not sentiment_df.empty:
        monthly_df = pd.merge(monthly_df, sentiment_df, left_index=True, right_index=True, how='left')
        monthly_df['News_Sentiment'] = monthly_df['News_Sentiment'].fillna(0)
    else:
        monthly_df['News_Sentiment'] = 0.0

    monthly_df['Price_Per_Area'] = monthly_df['Price_Per_Area'].replace(0, np.nan).ffill()
    monthly_df['Price_MA_3M'] = monthly_df['Price_Per_Area'].rolling(window=3).mean()
    monthly_df['Price_Change_1M'] = monthly_df['Price_Per_Area'].pct_change()
    
    lags = [1, 2, 3]
    for lag in lags:
        monthly_df[f'Price_Lag_{lag}'] = monthly_df['Price_Per_Area'].shift(lag)
        monthly_df[f'Volume_Lag_{lag}'] = monthly_df['Volume'].shift(lag)

    monthly_df.dropna(inplace=True)
    return monthly_df

def train_model(df):
    df['Target'] = df['Price_Per_Area'].shift(-1)
    df_train = df.dropna()
    
    features = [
        'News_Sentiment', 'Volume', 
        'Price_MA_3M', 'Price_Change_1M', 
        'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3',
        'Volume_Lag_1'
    ]
    
    X = df_train[features]
    y = df_train['Target']
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    pred_test = model.predict(X_test)
    score = r2_score(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    
    return model, X_test, y_test, pred_test, score, rmse, features

# ------------------------------------------------------------------------------
# 5. 메인 앱 UI
# ------------------------------------------------------------------------------
def app():
    st.write("### 🔍 분석 설정")
    
    col_input1, col_input2 = st.columns([3, 1])
    
    with col_input1:
        region_name = st.selectbox("분석 대상 지역 선택", list(DISTRICT_CODES.keys()))
    
    region_code_full = DISTRICT_CODES[region_name]
    lawd_cd = region_code_full[:5]
    dong_name = region_name.split()[-1]
    
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=3)
    
    start_date_str = start_date.strftime("%Y%m")
    end_date_str = end_date.strftime("%Y%m")
    
    with col_input2:
        st.write("") 
        st.write("")
        analyze_button = st.button("분석 및 예측 시작", type="primary")
    
    st.markdown("---")

    if analyze_button:
        st.write(f"### 🏙️ {region_name} 아파트 실거래가 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. 국토교통부 데이터 수집")
            with st.spinner("데이터 로딩 중..."):
                apt_df = get_molit_apt_data(lawd_cd, start_date_str, end_date_str)
            
            if apt_df.empty:
                st.error("데이터 수집 실패. API 키나 서버 상태를 확인하세요.")
                return
            st.success(f"거래 데이터 {len(apt_df)}건 수집 완료")
                
        with col2:
            st.markdown("#### 2. 뉴스 감성 분석")
            news_query = f"{dong_name} 아파트 가격 전망"
            with st.spinner("뉴스 분석 중..."):
                sentiment_df = get_naver_news_sentiment(news_query, start_date.date(), end_date.date())
            st.success("뉴스 데이터 처리 완료")

        final_df = preprocess_real_estate_data(apt_df, dong_name, sentiment_df)
        
        if final_df.empty:
            st.error(f"'{dong_name}' 지역의 충분한 거래 데이터를 찾을 수 없습니다.")
            return
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Price_Per_Area'], name='평당 가격(만원)'))
        fig.update_layout(title=f"{dong_name} 시세 추이", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        model, X_test, y_test, pred_test, r2, rmse, features = train_model(final_df)
        
        st.write("#### 4. AI 예측 결과 (다음 달)")
        last_row = final_df.iloc[[-1]][features]
        next_month_pred = model.predict(last_row)[0]
        current_price = final_df['Price_Per_Area'].iloc[-1]
        change_rate = (next_month_pred - current_price) / current_price * 100
        
        c1, c2 = st.columns(2)
        c1.metric("다음 달 예상 평당가", f"{next_month_pred:,.0f} 만원", f"{change_rate:.2f}%")
        c2.metric("모델 신뢰도 (R2)", f"{r2:.2f}")

if __name__ == "__main__":
    app()
