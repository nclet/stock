import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import sys

# ------------------------------------------------------------------------------
# 1. Configuration & Secret Management
# ------------------------------------------------------------------------------
st.set_page_config(page_title="부동산 통계 정보 분석", layout="wide")

# Get API Key from Streamlit Secrets
# 로컬 실행 시: .streamlit/secrets.toml 파일에 MOLIT_KEY = "your_key_here" 작성 필요
try:
    API_KEY = st.secrets["MOLIT_KEY"]
except KeyError:
    st.error("❌ 'MOLIT_KEY'를 찾을 수 없습니다. Streamlit Secrets 설정을 확인해주세요.")
    st.info("로컬에서 실행 중이라면 `.streamlit/secrets.toml` 파일에 키를 설정해야 합니다.")
    st.stop()

# Region mapping for R-ONE API
REGION_MAP = {
    "전국": "00", "서울특별시": "11", "부산광역시": "26", "대구광역시": "27",
    "인천광역시": "28", "광주광역시": "29", "대전광역시": "30", "울산광역시": "31",
    "세종특별자치시": "36", "경기도": "41", "강원도": "42", "충청북도": "43",
    "충청남도": "44", "전라북도": "45", "전라남도": "46", "경상북도": "47", "경상남도": "48"
}

# ------------------------------------------------------------------------------
# 2. Data Fetching Logic
# ------------------------------------------------------------------------------
def fetch_reb_data(api_key, region_code, start_month, end_month):
    """
    Fetches real estate statistics from R-ONE API.
    Endpoint: Apartment Sales Price Index
    """
    url = "https://www.reb.or.kr/r-one/openapi/statistics/propertyPriceIndex"
    
    params = {
        "key": api_key,
        "format": "json",
        "startmonth": start_month,
        "endmonth": end_month,
        "region": region_code,
        "p_type": "01" # 01: 아파트
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            # The structure of R-ONE API response often contains 'item' or 'body'
            if "item" in res_json:
                return pd.DataFrame(res_json["item"])
            else:
                return res_json # For debugging if structure differs
        else:
            st.error(f"API 호출 실패: 상태 코드 {response.status_code}")
            return None
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")
        return None

# ------------------------------------------------------------------------------
# 3. App UI & Execution
# ------------------------------------------------------------------------------
def main():
    st.title("📊 부동산 통계 정보 시스템 데이터 분석")
    st.markdown("Streamlit Secrets를 통해 인증키를 안전하게 불러와 사용합니다.")
    
    # Sidebar Filters
    st.sidebar.header("🔍 조회 설정")
    selected_region = st.sidebar.selectbox("지역 선택", list(REGION_MAP.keys()))
    
    col_s, col_e = st.sidebar.columns(2)
    start_year = col_s.number_input("시작 연도", 2018, 2025, 2022)
    end_year = col_e.number_input("종료 연도", 2018, 2025, 2024)
    
    start_month_str = f"{start_year}01"
    end_month_str = f"{end_year}12"
    
    if st.sidebar.button("데이터 불러오기", type="primary"):
        with st.spinner(f"{selected_region} 데이터 수집 중..."):
            df = fetch_reb_data(API_KEY, REGION_MAP[selected_region], start_month_str, end_month_str)
            
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                st.success(f"{selected_region} 데이터를 성공적으로 가져왔습니다.")
                
                # Data Processing for Chart
                # Attempt to identify date and value columns automatically
                date_col = next((c for c in df.columns if 'date' in c.lower() or 'research' in c.lower()), None)
                val_col = next((c for c in df.columns if 'idx' in c.lower() or 'price' in c.lower()), None)
                
                if date_col and val_col:
                    df = df.sort_values(by=date_col)
                    
                    # Layout with Metric & Chart
                    m1, m2 = st.columns(2)
                    latest_val = float(df[val_col].iloc[-1])
                    prev_val = float(df[val_col].iloc[0])
                    delta = round(latest_val - prev_val, 2)
                    
                    m1.metric("최근 지수", latest_val, delta=f"{delta}")
                    m2.metric("조회 기간", f"{start_year} ~ {end_year}")
                    
                    # Plotting
                    fig = px.line(df, x=date_col, y=val_col, 
                                 title=f"[{selected_region}] 아파트 매매가격지수 추이",
                                 labels={date_col: "조사년월", val_col: "지수"},
                                 markers=True,
                                 template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Raw Data 보기"):
                        st.dataframe(df, use_container_width=True)
                else:
                    st.warning("데이터는 수집되었으나 차트 생성용 컬럼을 찾을 수 없습니다.")
                    st.write("수집된 데이터 컬럼:", list(df.columns))
                    st.write(df)
            else:
                st.error("데이터를 가져오지 못했습니다. API 키의 권한 또는 호출 파라미터를 확인하세요.")
                if df is not None:
                    st.json(df) # Show error response from API

if __name__ == "__main__":
    main()
