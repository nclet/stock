import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.express as px
import lightgbm as lgb
from sklearn.metrics import r2_score
import urllib.parse

# ------------------------------------------------------------------------------
# 1. 설정 및 상수 정의
# ------------------------------------------------------------------------------
st.set_page_config(page_title="🏠 서울 실거래가 예측 (Seoul API)", layout="wide")

# 서울시 행정구역 목록
SEOUL_DISTRICTS = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"
]

# ------------------------------------------------------------------------------
# 2. 데이터 수집 함수 (안정성 강화)
# ------------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_seoul_apt_data(api_key, district, dong=""):
    """
    서울시 열린데이터 광장 API: 서울시 부동산 전월세/매매가 통합 정보
    인증키 형식: 346b6f7316ef40b9ad26f977378de07e
    """
    if not api_key:
        st.error("API 키가 없습니다. Secrets 설정을 확인하세요.")
        return pd.DataFrame()

    # 서울시 부동산 실거래가 정보 (tbLnOpendataRtmsV)
    # 호출 경로 형식: http://openapi.seoul.go.kr:8088/(인증키)/(데이터형식)/(서비스명)/(시작지점)/(종료지점)/(자치구명)/(법정동명)
    
    # URL 인코딩 (한글 파라미터 처리)
    encoded_district = urllib.parse.quote(district)
    
    # 1~1000번까지의 최신 데이터를 가져옴
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/tbLnOpendataRtmsV/1/1000/{encoded_district}"
    
    if dong:
        encoded_dong = urllib.parse.quote(dong)
        url += f"/{encoded_dong}"

    all_data = []
    try:
        response = requests.get(url, timeout=15)
        
        # HTTP 상태 코드 확인
        if response.status_code != 200:
            st.error(f"서버 응답 오류 (HTTP {response.status_code})")
            return pd.DataFrame()

        # JSON 파싱 전 텍스트 확인 (디버깅용)
        raw_text = response.text
        if not raw_text.strip().startswith('{'):
            st.error("API 서버에서 JSON이 아닌 응답을 보냈습니다. 키 권한이나 URL을 확인하세요.")
            with st.expander("서버 응답 내용 보기"):
                st.code(raw_text)
            return pd.DataFrame()

        data = response.json()
        
        # 서울시 API 특유의 에러 메시지 처리 (INFO-200 등)
        if 'RESULT' in data and data['RESULT'].get('CODE') != 'INFO-000':
            st.warning(f"API 메시지: {data['RESULT'].get('MESSAGE')}")
            return pd.DataFrame()

        if 'tbLnOpendataRtmsV' in data:
            items = data['tbLnOpendataRtmsV']['row']
            for item in items:
                # 아파트 거래만 추출
                if "아파트" in str(item.get('BLDG_NM', '')):
                    try:
                        # 금액 단위: 만원
                        price = float(item.get('OBJ_AMT', 0))
                        area = float(item.get('BLDG_AREA', 0))
                        deal_date = datetime.strptime(str(item.get('DEAL_YMD')), "%Y%m%d")
                        
                        all_data.append({
                            'Date': deal_date,
                            'SGG': item.get('SGG_NM'),
                            'Dong': item.get('BJDONG_NM'),
                            'Price': price,
                            'Area': area,
                            'Name': item.get('BLDG_NM'),
                            'Price_Per_Area': price / area if area > 0 else 0
                        })
                    except (ValueError, TypeError):
                        continue
    except Exception as e:
        st.error(f"알 수 없는 오류 발생: {e}")
        
    return pd.DataFrame(all_data)

# ------------------------------------------------------------------------------
# 3. 모델링 및 예측
# ------------------------------------------------------------------------------
def train_and_predict(df):
    if len(df) < 10:
        return None
    
    # 시계열 순 정렬
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # 월별 평균가 집계
    monthly = df.resample('MS').agg({
        'Price_Per_Area': 'mean',
        'Name': 'count'
    }).rename(columns={'Name': 'Volume'})
    
    monthly.dropna(inplace=True)
    
    if len(monthly) < 4:
        return None
        
    # 특성 생성
    monthly['Lag_1'] = monthly['Price_Per_Area'].shift(1)
    monthly['MA_3'] = monthly['Price_Per_Area'].rolling(window=3).mean()
    monthly['Target'] = monthly['Price_Per_Area'].shift(-1)
    
    train_data = monthly.dropna()
    
    if train_data.empty:
        return None
        
    features = ['Lag_1', 'MA_3', 'Volume']
    X = train_data[features]
    y = train_data['Target']
    
    # 모델 학습 (데이터가 적으므로 간단한 파라미터 사용)
    model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # 마지막 달 데이터를 바탕으로 다음 달 예측
    last_row = monthly.iloc[[-1]]
    prediction = model.predict(last_row[features])[0]
    
    return monthly, prediction, features, model

# ------------------------------------------------------------------------------
# 4. UI 구성
# ------------------------------------------------------------------------------
def main():
    st.title("🏠 서울 아파트 실거래가 AI 예측")
    st.markdown("서울시 열린데이터 광장의 실거래 데이터를 활용하여 다음 달 시세를 예측합니다.")
    
    # Sidebar - API 상태 확인
    with st.sidebar:
        st.header("설정")
        molit_key = st.secrets.get("MOLIT_KEY", "")
        if molit_key:
            st.success("API 키가 로드되었습니다.")
        else:
            st.error("MOLIT_KEY가 Secrets에 없습니다.")
            
    # 메인 검색창
    col1, col2 = st.columns(2)
    with col1:
        selected_sgg = st.selectbox("구 선택", SEOUL_DISTRICTS)
    with col2:
        target_dong = st.text_input("동 이름 (선택사항, 예: 개포동)", "")

    if st.button("실거래 데이터 분석 실행", type="primary"):
        with st.spinner(f"{selected_sgg} 데이터를 불러오는 중..."):
            df = get_seoul_apt_data(molit_key, selected_sgg, target_dong)
            
            if not df.empty:
                st.write(f"### 📈 {selected_sgg} {target_dong} 분석 결과")
                
                # 데이터 통계
                c1, c2, c3 = st.columns(3)
                c1.metric("최근 거래수", f"{len(df)} 건")
                c2.metric("평균 평당가", f"{df['Price_Per_Area'].mean():,.0f} 만원")
                c3.metric("최고가 단지", df.loc[df['Price'].idxmax(), 'Name'])

                # 시각화
                fig = px.scatter(df, x='Date', y='Price_Per_Area', color='Name', 
                                 title="최근 실거래 분포 (㎡당 가격)",
                                 labels={'Price_Per_Area': '만원/㎡'})
                st.plotly_chart(fig, use_container_width=True)
                
                # 예측 모델 가동
                results = train_and_predict(df)
                if results:
                    monthly_df, pred, features, model = results
                    
                    st.divider()
                    col_res1, col_res2 = st.columns([2, 1])
                    
                    with col_res1:
                        st.write("#### 월별 추세 및 예측값")
                        line_fig = px.line(monthly_df, y='Price_Per_Area', markers=True)
                        st.plotly_chart(line_fig, use_container_width=True)
                        
                    with col_res2:
                        current_val = monthly_df['Price_Per_Area'].iloc[-1]
                        change = pred - current_val
                        st.metric("다음 달 예상 평당가", f"{pred:,.0f} 만원", f"{change:+.2f} 만원")
                        st.info("※ 위 예측은 최근 거래 경향성을 바탕으로 한 AI의 추정치이며 투자 참고용으로만 사용하세요.")
                else:
                    st.warning("예측 모델을 생성하기에 월별 데이터가 부족합니다 (최소 4개월 이상의 데이터 필요).")
            else:
                st.error("데이터 수집에 실패했습니다. 상단의 에러 메시지를 확인하세요.")

if __name__ == "__main__":
    main()
