import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import urllib.parse

# ------------------------------------------------------------------------------
# 1. 설정 및 상수 정의
# ------------------------------------------------------------------------------
st.set_page_config(page_title="🏠 전국 아파트 실거래가 예측", layout="wide")

# 주요 지역 법정동 코드
DISTRICT_CODES = {
    "서울 강남구": "11680",
    "서울 서초구": "11650",
    "서울 송파구": "11710",
    "서울 마포구": "11440",
    "경기 성남 분당구": "41135",
    "경기 수원 영통구": "41117",
    "인천 연수구": "28185",
    "부산 해운대구": "26350",
    "대구 수성구": "27260",
    "세종특별자치시": "36110"
}

# ------------------------------------------------------------------------------
# 2. 데이터 수집 함수 (안전성 강화)
# ------------------------------------------------------------------------------

@st.cache_data(ttl=3600) # 오류 시 재시도를 위해 캐시 시간을 줄임
def get_molit_apt_data(service_key, lawd_cd, months_back=12):
    if not service_key:
        return pd.DataFrame()

    url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev"
    all_data = []
    end_dt = datetime.now()
    
    progress_bar = st.progress(0)
    
    for i in range(months_back):
        target_dt = end_dt - relativedelta(months=i)
        deal_ymd = target_dt.strftime("%Y%m")
        
        # 국토부 API 특이사항: 인증키가 이미 인코딩된 경우 unquote가 필요함
        params = {
            'serviceKey': urllib.parse.unquote(service_key),
            'LAWD_CD': lawd_cd,
            'DEAL_YMD': deal_ymd,
            'numOfRows': '1000'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                # 에러 메시지 확인
                header = root.find('header')
                res_code = header.find('resultCode').text if header is not None else "Unknown"
                res_msg = header.find('resultMsg').text if header is not None else "No Message"
                
                if res_code != '00':
                    st.error(f"⚠️ API 에러 ({deal_ymd}): {res_msg}")
                    continue

                items = root.findall('.//item')
                for item in items:
                    try:
                        price = int(item.find('거래금액').text.replace(',', '').strip())
                        area = float(item.find('전용면적').text)
                        
                        all_data.append({
                            'Date': datetime(int(item.find('년').text), 
                                             int(item.find('월').text), 
                                             int(item.find('일').text)),
                            'Dong': item.find('법정동').text.strip(),
                            'Name': item.find('아파트').text.strip(),
                            'Price': price,
                            'Area': area,
                            'Price_Per_Area': price / area
                        })
                    except Exception: continue
        except Exception as e:
            st.warning(f"연결 오류 ({deal_ymd}): {e}")
        
        progress_bar.progress((i + 1) / months_back)
    
    progress_bar.empty()
    return pd.DataFrame(all_data)

# ------------------------------------------------------------------------------
# 3. 모델링 함수 (에러 방지 로직 추가)
# ------------------------------------------------------------------------------
def run_ai_analysis(df):
    # 데이터 부족 시 조기 종료 (최소 10개 이상의 월별 데이터 권장)
    if df.empty:
        return None, "데이터가 존재하지 않습니다."
        
    df = df.sort_values('Date')
    # 월별 평균 평당가 계산
    monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
        'Price_Per_Area': 'mean',
        'Name': 'count'
    }).rename(columns={'Name': 'Volume'})
    monthly.index = monthly.index.to_timestamp()
    
    if len(monthly) < 5:
        return None, f"분석 가능한 월별 데이터가 너무 적습니다. (현재 {len(monthly)}개월)"

    # 특성 생성
    monthly['Lag_1'] = monthly['Price_Per_Area'].shift(1)
    monthly['MA_3'] = monthly['Price_Per_Area'].rolling(window=3).mean()
    monthly['Volume_Lag'] = monthly['Volume'].shift(1)
    monthly['Target'] = monthly['Price_Per_Area'].shift(-1)
    
    # 학습 데이터 구성 (결측치 제거)
    train_df = monthly.dropna()
    if train_df.empty:
        return None, "학습용 특성을 생성하기에 데이터가 부족합니다."

    features = ['Lag_1', 'MA_3', 'Volume_Lag']
    X = train_df[features]
    y = train_df['Target']
    
    # 스케일링 및 모델링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # 문제의 구간: X가 비어있지 않음을 위에서 확인
    
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # 다음 달 예측 데이터 준비
    last_row = monthly.iloc[[-1]]
    # 만약 마지막 행의 특성에 NaN이 있다면 (데이터가 너무 적을 때)
    if last_row[features].isnull().values.any():
        return None, "예측을 위한 최근 데이터가 불완전합니다."
        
    current_features_scaled = scaler.transform(last_row[features])
    prediction = model.predict(current_features_scaled)[0]
    
    return (monthly, prediction), None

# ------------------------------------------------------------------------------
# 4. 메인 UI
# ------------------------------------------------------------------------------
def main():
    st.title("🏠 전국 아파트 AI 시세 예측")
    
    molit_key = st.secrets.get("MOLIT_KEY", "")
    
    with st.sidebar:
        st.header("설정")
        region = st.selectbox("지역 선택", list(DISTRICT_CODES.keys()))
        period = st.slider("조회 기간 (개월)", 6, 36, 12)
        if not molit_key:
            st.error("🔑 Secrets에 'MOLIT_KEY'를 등록해주세요.")

    if st.button("분석 시작", type="primary"):
        if not molit_key:
            st.error("인증키가 설정되지 않았습니다.")
            return

        with st.spinner("국토교통부 실거래 데이터를 수집 중입니다..."):
            raw_df = get_molit_apt_data(molit_key, DISTRICT_CODES[region], period)
            
            if not raw_df.empty:
                st.success(f"✅ {len(raw_df):,}건의 실거래 데이터를 성공적으로 가져왔습니다.")
                
                # 분석 실행
                result, error_msg = run_ai_analysis(raw_df)
                
                if error_msg:
                    st.warning(f"⚠️ 분석 불가: {error_msg}")
                    # 수집된 데이터라도 보여줌
                    st.dataframe(raw_df.head())
                else:
                    monthly_df, pred = result
                    
                    # 시각화
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.line(monthly_df, y='Price_Per_Area', markers=True, 
                                     title=f"{region} 월별 평당가 추이 (만원/㎡)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        curr = monthly_df['Price_Per_Area'].iloc[-1]
                        diff = pred - curr
                        st.metric("현재 시세 (평당)", f"{curr:,.0f} 만원")
                        st.metric("AI 예측 (다음 달)", f"{pred:,.0f} 만원", f"{diff:+.2f} 만원")
                        
                        st.info("실거래 기반 통계 모델이므로 실제 시장 상황과 다를 수 있습니다.")
            else:
                st.error("❌ 수집된 데이터가 없습니다. 인증키가 유효한지, 혹은 해당 기간에 거래가 있는지 확인하세요.")

if __name__ == "__main__":
    main()
