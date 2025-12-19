import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
import lightgbm as lgb
from sklearn.metrics import r2_score
import urllib.parse

# ------------------------------------------------------------------------------
# 1. 설정 및 상수 정의
# ------------------------------------------------------------------------------
st.set_page_config(page_title="🏠 전국 아파트 실거래가 예측", layout="wide")

# 주요 지역 법정동 코드 (국토부 API는 앞 5자리가 구 코드입니다)
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
# 2. 데이터 수집 함수 (공공데이터포털/국토교통부 API 전용)
# ------------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_molit_apt_data(service_key, lawd_cd, months_back=24):
    """
    국토교통부 아파트매매 실거래 상세 자료 API 호출
    """
    if not service_key:
        return pd.DataFrame()

    url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev"
    
    all_data = []
    end_dt = datetime.now()
    
    # 최근 n개월치 데이터 수집
    for i in range(months_back):
        target_dt = end_dt - relativedelta(months=i)
        deal_ymd = target_dt.strftime("%Y%m")
        
        # 국토부 API는 인증키를 unquote 해서 보내야 하는 경우가 많습니다.
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
                # 결과 코드 확인
                result_code = root.find('.//resultCode')
                if result_code is not None and result_code.text != '00':
                    continue
                    
                for item in root.findall('.//item'):
                    try:
                        price = int(item.find('거래금액').text.replace(',', '').strip())
                        area = float(item.find('전용면적').text)
                        year = item.find('년').text
                        month = item.find('월').text
                        day = item.find('일').text
                        
                        all_data.append({
                            'Date': datetime(int(year), int(month), int(day)),
                            'Dong': item.find('법정동').text.strip(),
                            'Name': item.find('아파트').text.strip(),
                            'Price': price,
                            'Area': area,
                            'Price_Per_Area': price / area
                        })
                    except: continue
        except: continue
    
    return pd.DataFrame(all_data)

# ------------------------------------------------------------------------------
# 3. 모델링 및 예측
# ------------------------------------------------------------------------------
def run_analysis(df):
    if df.empty or len(df) < 20:
        return None
    
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # 월별 평당가 추이 집계
    monthly = df.resample('MS').agg({
        'Price_Per_Area': 'mean',
        'Name': 'count'
    }).rename(columns={'Name': 'Volume'})
    
    monthly = monthly.dropna()
    if len(monthly) < 6:
        return None
        
    # 특성 생성 (Feature Engineering)
    monthly['Lag_1'] = monthly['Price_Per_Area'].shift(1)
    monthly['MA_3'] = monthly['Price_Per_Area'].rolling(window=3).mean()
    monthly['Target'] = monthly['Price_Per_Area'].shift(-1)
    
    train_df = monthly.dropna()
    features = ['Lag_1', 'MA_3', 'Volume']
    
    X = train_df[features]
    y = train_df['Target']
    
    # LightGBM 모델링
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    # 다음 달 예측
    last_features = monthly.iloc[[-1]][features]
    prediction = model.predict(last_features)[0]
    
    return monthly, prediction, features

# ------------------------------------------------------------------------------
# 4. 메인 화면
# ------------------------------------------------------------------------------
def main():
    st.title("🏠 전국 아파트 시세 AI 분석기 (국토부 API)")
    st.markdown("공공데이터포털에서 발급받은 인증키를 사용하여 전국의 실거래가를 분석합니다.")
    
    # API 키 로드 상태 확인
    molit_key = st.secrets.get("MOLIT_KEY", "")
    
    with st.sidebar:
        st.header("분석 설정")
        if not molit_key:
            st.error("🔑 MOLIT_KEY를 Secrets에 등록해주세요.")
        else:
            st.success("✅ API 키 로드 완료")
            
        region = st.selectbox("분석 지역 선택", list(DISTRICT_CODES.keys()))
        period = st.slider("조회 기간 (개월)", 6, 36, 24)

    if st.button("분석 실행", type="primary"):
        if not molit_key:
            st.warning("인증키 없이는 분석을 시작할 수 없습니다.")
            return

        with st.spinner(f"{region} 데이터를 가져오는 중입니다..."):
            lawd_cd = DISTRICT_CODES[region]
            df = get_molit_apt_data(molit_key, lawd_cd, period)
            
            if not df.empty:
                st.write(f"### 📊 {region} 실거래 분석 리포트")
                
                # 시각화 1: 개별 거래 분포
                fig_scatter = px.scatter(df, x='Date', y='Price_Per_Area', 
                                         hover_name='Name', color='Dong',
                                         title="최근 실거래 분포 (만원/㎡)",
                                         labels={'Price_Per_Area': '평당가(만원)'})
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 분석 결과 실행
                analysis_results = run_analysis(df)
                
                if analysis_results:
                    monthly_df, pred, features = analysis_results
                    
                    st.divider()
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("#### 월별 평균가 및 예측 지표")
                        fig_line = px.line(monthly_df, y='Price_Per_Area', markers=True)
                        st.plotly_chart(fig_line, use_container_width=True)
                        
                    with col2:
                        current_avg = monthly_df['Price_Per_Area'].iloc[-1]
                        change = pred - current_avg
                        st.metric("현재 평균 평당가", f"{current_avg:,.0f} 만원")
                        st.metric("다음 달 예상 평당가", f"{pred:,.0f} 만원", f"{change:+.2f} 만원")
                        
                        st.info("💡 **AI 분석 의견**: 최근 거래 추세를 기반으로 다음 달 시세는 소폭 " + 
                                ("상승" if change > 0 else "하락") + "할 것으로 예측됩니다.")
                else:
                    st.warning("시계열 분석을 위한 데이터가 부족합니다.")
            else:
                st.error("데이터를 수집하지 못했습니다. 인증키를 확인하거나 공공데이터포털 서버 상태를 확인하세요.")

if __name__ == "__main__":
    main()
