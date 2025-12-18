import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit 
import urllib.parse
import FinanceDataReader as fdr
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import MinMaxScaler
import re
import shap 

# ------------------------
# ✨ 1. 부동산 전용 상수 및 키워드 설정
# ------------------------
st.set_page_config(page_title="🇰🇷 국내 부동산 추세 예측", layout="wide")
st.title("🏙️ 국내 부동산 시장 중단기 추세 예측 모델")

# 부동산 특화 뉴스 키워드
RE_POS_KEYWORDS = ['상승', '반등', '급등', '호재', '완화', '최고가', '분양호조', '금리인하', '공급부족']
RE_NEG_KEYWORDS = ['하락', '급락', '침체', '악재', '규제', '미분양', '금리인상', '매물적체', '거래절벽']
RE_MACRO_KEYWORDS = ['한국은행', '기준금리', 'LTV', 'DSR', '재건축', '공공택지', '신도시', '전세사기']

# ------------------------
# 📂 2. 국토교통부(MOLIT) API 데이터 수집 함수
# ------------------------
@st.cache_data(show_spinner="⏳ 국토교통부 실거래가 데이터 로드 중...")
def get_molit_real_estate(lawd_cd, deal_ym):
    # 수정된 엔드포인트: 포트 8081 제거 및 최신 주소 사용 가능성 확인
    # 1안 (기존 주소에서 포트만 제거):
    # url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade"
    
    # 2안 (공공데이터포털 통합 엔드포인트 - 권장):
    url = "http://apis.data.go.kr/1613000/RTMSOBJSvc/getRTMSDataSvcAptTrade"
    
    api_key = st.secrets["MOLIT_KEY"]
    
    params = {
        'serviceKey': api_key, # st.secrets에 이미 인코딩된 키가 있다면 그대로 사용
        'LAWD_CD': lawd_cd,
        'DEAL_YMD': deal_ym
    }
    
    try:
        # timeout 설정을 추가하여 무한 대기를 방지합니다.
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # HTTP 에러 발생 시 예외 발생
        
        root = ET.fromstring(response.content)
        
        # 결과 코드 확인 (정상: 00)
        header = root.find('.//header')
        result_code = header.find('resultCode').text
        if result_code != '00':
            st.error(f"API 응답 오류: {header.find('resultMsg').text}")
            return pd.DataFrame()

        items = root.findall('.//item')
        data = []
        for item in items:
            # 안전하게 데이터를 가져오기 위해 find().text 사용 시 None 체크
            try:
                price = item.find('거래금액').text.replace(',', '').strip()
                area = item.find('전용면적').text
                day = item.find('일').text.zfill(2)
                month = item.find('월').text.zfill(2)
                year = item.find('년').text
                
                data.append({
                    'Date': f"{year}-{month}-{day}",
                    'Price': int(price),
                    'Area': float(area),
                    'Name': item.find('아파트').text
                })
            except AttributeError:
                continue
                
        df = pd.DataFrame(data)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Price_per_Area'] = df['Price'] / df['Area']
            return df
        return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        st.error(f"🌐 네트워크 연결 오류: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ 데이터 파싱 오류: {e}")
        return pd.DataFrame()# ------------------------
# 📰 3. 네이버 뉴스 수집 및 분석 (부동산 특화)
# ------------------------
# (기존 analyze_sentiment, get_naver_news_api 함수는 유지하되 쿼리만 부동산으로 변경)
def analyze_re_text(title, description):
    text = title + " " + description
    pos_count = sum(text.count(word) for word in RE_POS_KEYWORDS)
    neg_count = sum(text.count(word) for word in RE_NEG_KEYWORDS)
    macro_count = sum(text.count(word) for word in RE_MACRO_KEYWORDS)
    
    pos_neg_ratio = (pos_count + 1) / (neg_count + 1)
    total_words = len(text.split())
    macro_ratio = macro_count / total_words if total_words > 0 else 0
    
    return pos_count, neg_count, pos_neg_ratio, macro_ratio

# ------------------------
# 📊 4. 매크로 데이터 수집 (FRED - 한국 금리 및 통화량)
# ------------------------
@st.cache_data(show_spinner="⏳ 매크로 지표(금리/통화량) 로드 중...")
def get_kr_macro_data():
    # 한국 관련 FRED 티커 (INTDSRKRM193N: 한국 기준금리, MYAGM2KRM193N: 한국 M2 통화량)
    tickers = {
        "INTDSRKRM193N": "KR_Interest_Rate", 
        "MYAGM2KRM193N": "KR_M2_Money",
        "USRECOVERY": "US_Recession" # 참고용 미국 경기지표
    }
    # (기존 get_fred_data 로직 활용하여 데이터 수집)
    # ... (생략: 기존 코드와 동일하게 구현하되 티커만 위 항목으로 교체)
    return results

# ------------------------
# ⚙️ 5. 피처 엔지니어링 및 모델 학습
# ------------------------
def create_re_features(df):
    """부동산 데이터 특성에 맞춘 피처 생성"""
    df = df.copy()
    
    # 1. 가격 지수화 (단위 면적당 가격의 이동 평균)
    df['Price_MA_30'] = df['Price_per_Area'].rolling(window=30).mean()
    
    # 2. 거래량 피처 (일별 거래 건수)
    df['Trade_Count'] = df.groupby(df.index)['Price'].transform('count')
    
    # 3. 타겟 설정: 향후 30일(약 1개월) 뒤 가격 변동률
    df['Target_Next_Month'] = df['Price_per_Area'].pct_change(periods=30).shift(-30) * 100
    
    # 4. 뉴스/매크로 결합 및 Lag 생성
    # ... (기존 create_features 로직과 동일하게 시차 변수 생성)
    
    return df.dropna()

# ------------------------
# 🚀 6. Streamlit 메인 UI 및 실행 로직
# ------------------------
st.sidebar.header("🔍 분석 설정")
region_code = st.sidebar.selectbox("지역 선택", ["11680 (강남구)", "11110 (종로구)", "41135 (분당구)"])
lawd_cd = region_code.split(" ")[0]

if st.button("🚀 부동산 시장 분석 시작"):
    # 데이터 수집 파이프라인
    # 1. 실거래가 데이터 수집 (최근 12개월치 순회하며 수집)
    curr_year = datetime.now().year
    curr_month = datetime.now().month
    
    all_re_data = []
    for i in range(12): # 최근 1년치
        target_date = datetime(curr_year, curr_month, 1) - timedelta(days=i*30)
        ym = target_date.strftime("%Y%m")
        month_df = get_molit_real_estate(lawd_cd, ym)
        all_re_data.append(month_df)
    
    df_re = pd.concat(all_re_data).sort_values('Date').set_index('Date')
    
    # 2. 뉴스 데이터 분석
    news_query = f"{region_code.split(' ')[1]} 부동산|아파트 전망|미분양"
    # ... (기존 뉴스 수집 및 analyze_re_text 적용)
    
    # 3. 매크로 데이터 결합 (KR_Interest_Rate, KR_M2)
    # ... (기존 merge 로직 적용)

    # 4. 결과 시각화
    st.subheader(f"📈 {region_code} 가격 추이 및 예측")
    fig = px.line(df_re, y='Price_per_Area', title="단위면적당 실거래가 추이")
    st.plotly_chart(fig, use_container_width=True)
    
    # (이하 모델 학습, SHAP 해석, 투자 시그널 출력 로직은 기존 코드 구조를 그대로 유지)
    st.success("분석이 완료되었습니다. 하단의 예측 결과를 확인하세요.")
