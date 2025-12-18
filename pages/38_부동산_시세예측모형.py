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
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
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
# 2. 데이터 수집 함수
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
    except Exception as e:
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
def get_naver_news_sentiment(query, start_date, end_date):
    try:
        # Secrets에서 키 가져오기 (다양한 경로 대응)
        client_id = st.secrets.get("NAVER_CLIENT_ID") or st.secrets.get("naver", {}).get("client_id")
        client_secret = st.secrets.get("NAVER_CLIENT_SECRET") or st.secrets.get("naver", {}).get("client_secret")
        
        if not client_id:
            st.error("네이버 API 키가 설정되지 않았습니다.")
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    all_news = []
    enc_query = urllib.parse.quote(query)
    
    for start_idx in range(1, 301, 100):
        url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display=100&start={start_idx}&sort=date"
        headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                items = res.json().get('items', [])
                for item in items:
                    pub_date = datetime.strptime(item['pubDate'], "%a, %d %b %Y %H:%M:%S %z")
                    all_news.append({'Date': pub_date.date(), 'Title': item['title']})
        except: break
            
    if not all_news: return pd.DataFrame()
    
    df_news = pd.DataFrame(all_news)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    df_news = df_news[(df_news['Date'].dt.date >= start_date) & (df_news['Date'].dt.date <= end_date)]
    
    if df_news.empty: return pd.DataFrame()

    df_news['Score'] = df_news['Title'].apply(analyze_sentiment_score)
    monthly_sentiment = df_news.set_index('Date')['Score'].resample('MS').mean().fillna(0)
    return monthly_sentiment.to_frame(name='News_Sentiment')

@st.cache_data(ttl=86400)
def get_molit_apt_data(lawd_cd, start_date_str, end_date_str):
    try:
        service_key = st.secrets.get("MOLIT_KEY") or st.secrets.get("data", {}).get("MOLIT_KEY")
        if not service_key:
            st.error("국토교통부 API 키가 설정되지 않았습니다.")
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev"
    start_dt = datetime.strptime(start_date_str, "%Y%m")
    end_dt = datetime.strptime(end_date_str, "%Y%m")
    
    current_dt = start_dt
    all_data = []
    
    while current_dt <= end_dt:
        params = {'serviceKey': service_key, 'LAWD_CD': lawd_cd[:5], 'DEAL_YMD': current_dt.strftime("%Y%m"), 'numOfRows': '1000'}
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
# 3. 모델링 로직
# ------------------------------------------------------------------------------
def process_and_train(apt_df, dong_name, sentiment_df):
    df = apt_df[apt_df['Dong'].str.contains(dong_name)].copy()
    if df.empty: return None
    
    df.set_index('Date', inplace=True)
    monthly = df.resample('MS').agg({'Price_Per_Area': 'mean', 'Dong': 'count'}).rename(columns={'Dong': 'Volume'})
    
    if not sentiment_df.empty:
        monthly = monthly.join(sentiment_df, how='left').fillna(0)
    else:
        monthly['News_Sentiment'] = 0.0
        
    monthly['Price_Lag_1'] = monthly['Price_Per_Area'].shift(1)
    monthly['Price_MA_3M'] = monthly['Price_Per_Area'].rolling(3).mean()
    monthly['Target'] = monthly['Price_Per_Area'].shift(-1)
    monthly.dropna(inplace=True)
    
    if len(monthly) < 5: return None
    
    features = ['Price_Lag_1', 'Price_MA_3M', 'Volume', 'News_Sentiment']
    X = monthly[features]
    y = monthly['Target']
    
    split = int(len(X) * 0.8)
    if split < 1: split = len(X) - 1
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, importance_type='split')
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    score = r2_score(y_test, preds) if len(y_test) > 1 else 0
    
    return monthly, model, features, X_test, y_test, preds, score

# ------------------------------------------------------------------------------
# 4. 메인 실행부
# ------------------------------------------------------------------------------
def app():
    st.write("### 🔍 분석 설정")
    region_name = st.selectbox("분석 대상 지역 선택", list(DISTRICT_CODES.keys()))
    lawd_cd = DISTRICT_CODES[region_name]
    dong_name = region_name.split()[-1]
    
    if st.button("분석 및 예측 시작", type="primary"):
        with st.spinner("데이터를 불러오고 분석하는 중입니다..."):
            end_date = datetime.now()
            start_date = end_date - relativedelta(years=3)
            
            apt_df = get_molit_apt_data(lawd_cd, start_date.strftime("%Y%m"), end_date.strftime("%Y%m"))
            sentiment_df = get_naver_news_sentiment(f"{dong_name} 부동산 전망", start_date.date(), end_date.date())
            
            result = process_and_train(apt_df, dong_name, sentiment_df)
            
            if result:
                monthly_df, model, features, X_test, y_test, preds, score = result
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"#### {dong_name} 평당가 추이")
                    fig = px.line(monthly_df, y='Price_Per_Area', title='월평균 평당 가격(만원)')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("#### AI 분석 결과")
                    st.metric("모델 신뢰도 (R2)", f"{score:.2f}")
                    
                    last_features = monthly_df[features].iloc[[-1]]
                    next_pred = model.predict(last_features)[0]
                    current = monthly_df['Price_Per_Area'].iloc[-1]
                    change_pct = ((next_pred - current) / current) * 100
                    
                    # 오류가 났던 부분 수정: f-string 내의 표현식 정리
                    st.metric(
                        label="다음 달 예상 평당가", 
                        value=f"{next_pred:,.0f} 만원", 
                        delta=f"{change_pct:+.2f}%"
                    )
                
                st.write("#### 주요 가격 결정 요인 (Feature Importance)")
                imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                fig_imp = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("분석에 필요한 데이터가 충분하지 않습니다 (최근 거래 데이터 부족).")

if __name__ == "__main__":
    app()









# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# import xml.etree.ElementTree as ET
# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta
# import plotly.graph_objects as go
# import plotly.express as px
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import lightgbm as lgb
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import MinMaxScaler
# import urllib.parse
# import re

# # ------------------------------------------------------------------------------
# # 1. 설정 및 상수 정의
# # ------------------------------------------------------------------------------
# st.set_page_config(page_title="🏠 국내 아파트 실거래가 예측", layout="wide")
# st.title("🏠 국내 아파트 실거래가 예측 모델 (AI Sentiment + ML)")

# st.markdown("""
# **국토교통부 실거래가 데이터**와 **네이버 뉴스 감성 분석**을 결합하여 
# 특정 지역 아파트의 향후 가격 추세를 예측합니다.
# """)

# # 감성 분석 키워드 (부동산 시장 특화)
# POS_KEYWORDS = ['상승', '급등', '호재', '개발', 'GTX', '재건축', '완화', '회복', '신고가', '매수', '공급부족']
# NEG_KEYWORDS = ['하락', '급락', '폭락', '규제', '미분양', '금리인상', '역전세', '깡통', '매도', '관망', '침체']

# # 법정동 코드 매핑 (예시: 일부 주요 지역만 포함, 실제로는 전체 법정동 코드 DB 필요)
# # 사용자가 직접 입력하거나 선택할 수 있게 확장 가능
# DISTRICT_CODES = {
#     "서울 강남구 개포동": "1168010300",
#     "서울 강남구 대치동": "1168010600",
#     "서울 서초구 반포동": "1165010700",
#     "서울 송파구 잠실동": "1171010100",
#     "서울 마포구 아현동": "1144010100",
#     "성남 분당구 정자동": "4113510300",
#     "대구 수성구 범어동": "2726010100",
#     "부산 해운대구 우동": "2635010500"
# }

# # ------------------------------------------------------------------------------
# # 2. 데이터 수집 함수 (국토교통부 & 네이버)
# # ------------------------------------------------------------------------------

# @st.cache_resource
# def load_sentiment_model():
#     """Hugging Face 감성 분석 모델 로드"""
#     # hf_token = st.secrets.get("HF_TOKEN") # 필요한 경우 사용
#     model_name = "snunlp/KR-FinBert-SC"
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSequenceClassification.from_pretrained(model_name)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         return tokenizer, model, device
#     except Exception as e:
#         return None, None, None

# tokenizer, sentiment_model, device = load_sentiment_model()

# def analyze_sentiment_score(text):
#     """텍스트 감성 점수 계산 (-1 ~ 1)"""
#     if not text or not sentiment_model: return 0.0
#     try:
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = sentiment_model(**inputs)
#             probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             # Label: 0(Negative), 1(Neutral), 2(Positive) -> 모델마다 다를 수 있으니 확인 필요
#             # KR-FinBert-SC: 0:Negative, 1:Neutral, 2:Positive
#             neg_prob = probs[0][0].item()
#             pos_prob = probs[0][2].item()
#         return pos_prob - neg_prob
#     except:
#         return 0.0

# @st.cache_data(ttl=3600)
# def get_naver_news_sentiment(query, start_date, end_date):
#     """네이버 뉴스 수집 및 감성 지수 집계 (월별)"""
#     try:
#         client_id = st.secrets["naver"]["client_id"]
#         client_secret = st.secrets["naver"]["client_secret"]
#     except KeyError:
#         st.error("Secrets에 네이버 API 키가 없습니다.")
#         return pd.DataFrame()

#     # 기간 내 데이터를 월별로 수집하기엔 API 한계가 있으므로, 최근 100~200개 기사로 추세 파악
#     # 정확한 시계열 매칭을 위해선 날짜별 쿼리가 필요하나, 여기서는 '최근 뉴스'의 경향성을 
#     # 과거 데이터에 Smoothing하여 적용하는 약식 방법을 사용하거나, 
#     # 실제 운영 시엔 매일 수집하여 DB에 쌓아야 함.
#     # **본 코드에서는 최근 1000개 뉴스를 수집하여 날짜별로 그룹화합니다.**
    
#     all_news = []
    
#     # 검색어 확장
#     enc_query = urllib.parse.quote(query)
    
#     for start_idx in range(1, 1000, 100): # 최대 1000개
#         url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display=100&start={start_idx}&sort=date"
#         headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
        
#         try:
#             res = requests.get(url, headers=headers)
#             if res.status_code != 200: break
#             items = res.json().get('items', [])
#             if not items: break
            
#             for item in items:
#                 pub_date_str = item['pubDate']
#                 # "Tue, 15 Nov 2022 10:00:00 +0900" 형식
#                 pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
#                 all_news.append({
#                     'Date': pub_date.date(),
#                     'Title': item['title'],
#                     'Description': item['description']
#                 })
#         except:
#             break
            
#     if not all_news:
#         return pd.DataFrame()
    
#     df_news = pd.DataFrame(all_news)
    
#     # 날짜 필터링
#     df_news = df_news[(df_news['Date'] >= start_date) & (df_news['Date'] <= end_date)]
    
#     if df_news.empty:
#         return pd.DataFrame()

#     # 감성 분석 수행 (오래 걸릴 수 있으므로 progress bar)
#     st.info(f"뉴스 {len(df_news)}건 감성 분석 중...")
    
#     # 단순화: 제목만 분석
#     df_news['Score'] = df_news['Title'].apply(lambda x: analyze_sentiment_score(x))
    
#     # 일별 평균 -> 월별 평균으로 변환 (부동산은 월간 데이터가 메인)
#     df_news['Date'] = pd.to_datetime(df_news['Date'])
#     df_news.set_index('Date', inplace=True)
#     monthly_sentiment = df_news['Score'].resample('M').mean()
    
#     # 인덱스를 해당 월의 마지막 날짜에서 첫 날짜로 변경 (실거래가 데이터와 매칭 편의)
#     monthly_sentiment.index = monthly_sentiment.index.to_period('M').to_timestamp()
    
#     return monthly_sentiment.to_frame(name='News_Sentiment')

# @st.cache_data(ttl=3600 * 24)
# def get_molit_apt_data(lawd_cd, start_date_str, end_date_str):
#     """
#     국토교통부 아파트 실거래가 API 호출 (월별 반복)
#     lawd_cd: 법정동코드 앞 5자리 (시군구) -> API 스펙상 지역코드 5자리
#     """
#     try:
#         service_key = st.secrets["data"]["MOLIT_KEY"] # Decoding key 권장
#     except KeyError:
#         st.error("Secrets에 MOLIT_KEY가 없습니다.")
#         return pd.DataFrame()

#     url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev"
    
#     start_dt = datetime.strptime(start_date_str, "%Y%m")
#     end_dt = datetime.strptime(end_date_str, "%Y%m")
    
#     current_dt = start_dt
#     all_data = []
    
#     progress_bar = st.progress(0)
#     total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
#     idx = 0
    
#     while current_dt <= end_dt:
#         ym = current_dt.strftime("%Y%m")
#         params = {
#             'serviceKey': service_key,
#             'LAWD_CD': lawd_cd[:5], # 시군구 코드 5자리
#             'DEAL_YMD': ym,
#             'numOfRows': '1000' # 한번에 가져올 개수
#         }
        
#         try:
#             response = requests.get(url, params=params)
#             if response.status_code == 200:
#                 root = ET.fromstring(response.content)
#                 items = root.findall('.//item')
                
#                 for item in items:
#                     # 법정동 필터링 (API는 시군구 전체를 주므로 동으로 필터링 필요)
#                     dong = item.find('법정동').text.strip()
#                     # 법정동 코드가 아닌 법정동 명으로 필터링 해야 함 (데이터에 법정동코드가 없을 수 있음)
#                     # 여기서는 일단 다 가져오고 나중에 필터링
                    
#                     try:
#                         deal_amount = int(item.find('거래금액').text.replace(',', '').strip())
#                         area = float(item.find('전용면적').text)
#                         year = int(item.find('년').text)
#                         month = int(item.find('월').text)
#                         day = int(item.find('일').text)
#                         date = datetime(year, month, day)
                        
#                         all_data.append({
#                             'Date': date,
#                             'Dong': dong,
#                             'Price': deal_amount,
#                             'Area': area,
#                             'Price_Per_Area': deal_amount / area # 전용면적당 가격
#                         })
#                     except:
#                         continue
                        
#         except Exception as e:
#             # st.warning(f"{ym} 데이터 로드 중 오류: {e}")
#             pass
        
#         current_dt += relativedelta(months=1)
#         idx += 1
#         progress_bar.progress(min(idx / total_months, 1.0))
        
#     progress_bar.empty()
    
#     return pd.DataFrame(all_data)

# # ------------------------------------------------------------------------------
# # 3. 데이터 전처리 및 피처 엔지니어링
# # ------------------------------------------------------------------------------
# def preprocess_real_estate_data(df, target_dong_name, sentiment_df):
#     """데이터 정제 및 피처 생성"""
#     if df.empty:
#         return pd.DataFrame()
        
#     # 1. 특정 법정동 필터링
#     # target_dong_name 예: "개포동"
#     df = df[df['Dong'] == target_dong_name].copy()
    
#     if df.empty:
#         st.warning(f"'{target_dong_name}'에 대한 거래 데이터가 없습니다.")
#         return pd.DataFrame()

#     # 2. 이상치 제거 (평당 가격 기준 상하위 1% 제거)
#     q_low = df['Price_Per_Area'].quantile(0.01)
#     q_high = df['Price_Per_Area'].quantile(0.99)
#     df = df[(df['Price_Per_Area'] >= q_low) & (df['Price_Per_Area'] <= q_high)]

#     # 3. 월 단위로 리샘플링 (평균 가격, 거래량)
#     df.set_index('Date', inplace=True)
#     monthly_df = df.resample('M').agg({
#         'Price_Per_Area': 'mean', # 월 평균 평당가
#         'Price': 'mean',          # 월 평균 거래가
#         'Dong': 'count'           # 거래량
#     }).rename(columns={'Dong': 'Volume'})
    
#     # 인덱스 조정 (월초 기준)
#     monthly_df.index = monthly_df.index.to_period('M').to_timestamp()
    
#     # 4. 뉴스 감성 데이터 병합
#     if not sentiment_df.empty:
#         monthly_df = pd.merge(monthly_df, sentiment_df, left_index=True, right_index=True, how='left')
#         # 결측치는 0 (중립) 또는 전월 값으로 채움
#         monthly_df['News_Sentiment'] = monthly_df['News_Sentiment'].fillna(0)
#     else:
#         monthly_df['News_Sentiment'] = 0.0

#     # 5. 기술적 피처 생성 (Lag, MA, Change)
#     # 결측치 채우기 (거래 없는 달은 이전 달 가격 유지)
#     monthly_df['Price_Per_Area'] = monthly_df['Price_Per_Area'].replace(0, np.nan).ffill()
    
#     monthly_df['Price_MA_3M'] = monthly_df['Price_Per_Area'].rolling(window=3).mean()
#     monthly_df['Price_MA_6M'] = monthly_df['Price_Per_Area'].rolling(window=6).mean()
#     monthly_df['Price_Change_1M'] = monthly_df['Price_Per_Area'].pct_change()
#     monthly_df['Volume_MA_3M'] = monthly_df['Volume'].rolling(window=3).mean()
    
#     # Lag Features (과거 데이터로 미래 예측)
#     lags = [1, 2, 3, 6]
#     for lag in lags:
#         monthly_df[f'Price_Lag_{lag}'] = monthly_df['Price_Per_Area'].shift(lag)
#         monthly_df[f'Volume_Lag_{lag}'] = monthly_df['Volume'].shift(lag)
#         monthly_df[f'Sentiment_Lag_{lag}'] = monthly_df['News_Sentiment'].shift(lag)

#     monthly_df.dropna(inplace=True)
#     return monthly_df

# # ------------------------------------------------------------------------------
# # 4. 모델 훈련 및 예측
# # ------------------------------------------------------------------------------
# def train_model(df):
#     # 타겟: 다음 달의 평당 가격 (Price_Per_Area)
#     # 현재 시점의 Feature들로 T+1 시점의 가격을 예측하도록 데이터셋 구성
    
#     # Shift target -1 (Next Month Price)
#     df['Target'] = df['Price_Per_Area'].shift(-1)
#     df_train = df.dropna()
    
#     features = [c for c in df_train.columns if c not in ['Target', 'Price', 'Price_Per_Area']] 
#     # Price_Per_Area 자체는 Lag 피처로 들어가므로 제외, 현재가 기준으로 예측하려면 Lag 0도 포함 가능하나 
#     # 여기선 Lag 변수들과 MA 등을 주로 사용
    
#     # 주요 피처 명시적 선택
#     features = [
#         'News_Sentiment', 'Volume', 
#         'Price_MA_3M', 'Price_Change_1M', 
#         'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3',
#         'Volume_Lag_1', 'Sentiment_Lag_1'
#     ]
    
#     X = df_train[features]
#     y = df_train['Target']
    
#     # Train/Test Split (시계열)
#     train_size = int(len(X) * 0.8)
#     X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
#     y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
#     # LightGBM Model
#     model = lgb.LGBMRegressor(
#         n_estimators=500,
#         learning_rate=0.05,
#         num_leaves=31,
#         random_state=42,
#         n_jobs=-1
#     )
    
#     model.fit(X_train, y_train)
    
#     # 예측
#     pred_train = model.predict(X_train)
#     pred_test = model.predict(X_test)
    
#     # 평가
#     score = r2_score(y_test, pred_test)
#     rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    
#     return model, X_test, y_test, pred_test, score, rmse, features

# # ------------------------------------------------------------------------------
# # 5. 메인 앱 UI
# # ------------------------------------------------------------------------------
# def app():
#     # 검색 설정 (메인 본문 상단으로 이동)
#     st.write("### 🔍 분석 설정")
    
#     col_input1, col_input2 = st.columns([3, 1])
    
#     with col_input1:
#         region_name = st.selectbox(
#             "분석 대상 지역 선택",
#             list(DISTRICT_CODES.keys())
#         )
    
#     # 지역 코드 및 법정동 명 추출
#     region_code_full = DISTRICT_CODES[region_name]
#     lawd_cd = region_code_full[:5] # API 요청용 시군구 코드
#     # 법정동 이름 추출 (예: "서울 강남구 개포동" -> "개포동")
#     dong_name = region_name.split()[-1]
    
#     # 날짜 설정
#     end_date = datetime.now()
#     start_date = end_date - relativedelta(years=3) # 3년치 데이터
    
#     start_date_str = start_date.strftime("%Y%m")
#     end_date_str = end_date.strftime("%Y%m")
    
#     with col_input2:
#         # 버튼을 아래쪽으로 정렬하기 위해 빈 공간 추가할 수도 있음
#         st.write("") 
#         st.write("")
#         analyze_button = st.button("분석 및 예측 시작", type="primary")
    
#     st.markdown("---")

#     if analyze_button:
#         st.write(f"### 🏙️ {region_name} 아파트 실거래가 분석")
        
#         col1, col2 = st.columns(2)
        
#         # 1. 데이터 수집
#         with col1:
#             st.markdown("#### 1. 데이터 수집 중...")
#             with st.spinner("국토교통부 실거래가 데이터 로딩..."):
#                 apt_df = get_molit_apt_data(lawd_cd, start_date_str, end_date_str)
            
#             if apt_df.empty:
#                 st.error("국토교통부 데이터 수집 실패. API 키나 조회 기간을 확인하세요.")
#                 return
#             else:
#                 st.success(f"실거래가 데이터 {len(apt_df)}건 수집 완료")
                
#         with col2:
#             st.markdown("#### 2. 뉴스 감성 분석 중...")
#             news_query = f"{dong_name} 아파트 시장 전망"
#             with st.spinner(f"네이버 뉴스 검색: '{news_query}'..."):
#                 # 뉴스 수집 기간은 데이터 분석 기간과 맞춤
#                 sentiment_df = get_naver_news_sentiment(news_query, start_date.date(), end_date.date())
            
#             if sentiment_df.empty:
#                 st.warning("관련 뉴스 데이터가 부족하여 감성 지수를 0으로 설정합니다.")
#             else:
#                 st.success(f"월별 감성 지수 데이터 생성 완료")

#         # 2. 데이터 전처리
#         st.markdown("#### 3. 데이터 전처리 및 피처 엔지니어링")
#         final_df = preprocess_real_estate_data(apt_df, dong_name, sentiment_df)
        
#         if final_df.empty:
#             st.error(f"선택한 동({dong_name})의 데이터가 부족하여 분석할 수 없습니다.")
#             return
            
#         # 데이터 미리보기
#         with st.expander("가공된 데이터셋 보기"):
#             st.dataframe(final_df.tail(10))
            
#         # 시각화 1: 가격 및 감성 추이
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Price_Per_Area'], name='평당 가격(만원)', yaxis='y1'))
#         fig.add_trace(go.Bar(x=final_df.index, y=final_df['News_Sentiment'], name='뉴스 감성지수', yaxis='y2', marker_color='orange', opacity=0.3))
        
#         fig.update_layout(
#             title=f"{dong_name} 평당 실거래가 및 시장 심리 추이",
#             yaxis=dict(title='평당 가격 (만원)', side='left'),
#             yaxis2=dict(title='감성 지수', side='right', overlaying='y', range=[-1, 1]),
#             hovermode="x unified"
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         # 3. 모델 훈련
#         st.markdown("#### 4. AI 예측 모델링 (LightGBM)")
#         model, X_test, y_test, pred_test, r2, rmse, features = train_model(final_df)
        
#         col_m1, col_m2 = st.columns(2)
#         col_m1.metric("모델 설명력 (R2 Score)", f"{r2:.2f}")
#         col_m2.metric("평균 오차 (RMSE)", f"{rmse:.0f} 만원")
        
#         # 4. 예측 결과 시각화
#         fig_pred = go.Figure()
#         fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines+markers', name='실제 가격'))
#         fig_pred.add_trace(go.Scatter(x=y_test.index, y=pred_test, mode='lines+markers', name='예측 가격', line=dict(dash='dot', color='red')))
#         fig_pred.update_layout(title="테스트 기간 예측 결과 비교", xaxis_title="날짜", yaxis_title="평당 가격")
#         st.plotly_chart(fig_pred, use_container_width=True)
        
#         # 5. 향후 예측 (다음 달)
#         last_row = final_df.iloc[[-1]][features] # 가장 최근 데이터
#         next_month_pred = model.predict(last_row)[0]
#         current_price = final_df['Price_Per_Area'].iloc[-1]
        
#         change_rate = (next_month_pred - current_price) / current_price * 100
        
#         st.info(f"📊 **[{dong_name}]** 아파트의 다음 달 예상 평당 가격은 **약 {next_month_pred:,.0f}만원** 입니다.")
#         st.metric(
#             label="다음 달 예상 변동률", 
#             value=f"{next_month_pred:,.0f}만원", 
#             delta=f"{change_rate:.2f}%"
#         )
        
#         # 피처 중요도
#         st.markdown("---")
#         st.subheader("🔍 가격 결정 요인 분석 (Feature Importance)")
#         importance_df = pd.DataFrame({
#             'Feature': features,
#             'Importance': model.feature_importances_
#         }).sort_values(by='Importance', ascending=False)
        
#         fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='변수별 중요도')
#         st.plotly_chart(fig_imp, use_container_width=True)

# if __name__ == "__main__":
#     app()
