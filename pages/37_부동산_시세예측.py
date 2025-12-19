import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

# ------------------------------------------------------------------------------
# 1. Configuration & Constants
# ------------------------------------------------------------------------------
st.set_page_config(page_title="부동산 통계 정보 시스템(R-ONE) 분석", layout="wide")

# User's API Key
API_KEY = "9f7c27f87c204528b4eb9945627038ce"

# Reference: R-ONE Common Region Codes (Representative samples)
REGION_MAP = {
    "전국": "00",
    "서울특별시": "11",
    "부산광역시": "26",
    "대구광역시": "27",
    "인천광역시": "28",
    "광주광역시": "29",
    "대전광역시": "30",
    "울산광역시": "31",
    "세종특별자치시": "36",
    "경기도": "41"
}

# ------------------------------------------------------------------------------
# 2. API Data Fetching Function
# ------------------------------------------------------------------------------
def fetch_reb_stats(api_key, region_code, start_date, end_date):
    """
    Fetches Apartment Price Index data from R-ONE Open API.
    Endpoint: http://www.reb.or.kr/r-one/openapi/statistics/propertyPriceIndex
    (Note: The actual endpoint may vary based on the specific statistics applied for)
    """
    # This is a representative endpoint for 'Apartment Purchase Price Index'
    # Base URL for R-ONE Statistics API
    url = "https://www.reb.or.kr/r-one/openapi/statistics/propertyPriceIndex"
    
    params = {
        "key": api_key,
        "format": "json",
        "startmonth": start_date,  # Format: YYYYMM
        "endmonth": end_date,      # Format: YYYYMM
        "region": region_code,     # Region code
        "p_type": "01"             # 01: Apartment
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if data exists in the response
            if "item" in data:
                # Assuming 'item' contains a list of monthly data points
                df = pd.DataFrame(data["item"])
                
                # Standardize Column Names (Adjust based on actual API response keys)
                # Typical R-ONE keys: 'rsdl_prc_idx' (index), 'research_date' (date)
                if not df.empty:
                    # Rename for clarity if needed (hypothetical mapping)
                    # df.rename(columns={'research_date': 'date', 'rsdl_prc_idx': 'value'}, inplace=True)
                    return df
            else:
                st.warning("API 응답에 데이터 항목(item)이 없습니다. 인증키의 서비스 승인 여부를 확인하세요.")
                return pd.DataFrame()
        else:
            st.error(f"API Error: Status Code {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------------------------------------
def main():
    st.title("🏠 부동산 통계(R-ONE) 데이터 대시보드")
    st.markdown(f"**인증키:** `{API_KEY}` 를 사용하여 데이터를 수집합니다.")
    
    # Sidebar for filters
    st.sidebar.header("조회 필터")
    
    selected_region = st.sidebar.selectbox("지역 선택", list(REGION_MAP.keys()))
    region_code = REGION_MAP[selected_region]
    
    col1, col2 = st.sidebar.columns(2)
    start_year = col1.selectbox("시작 연도", range(2020, 2026), index=0)
    end_year = col2.selectbox("종료 연도", range(2020, 2026), index=5)
    
    start_month = f"{start_year}01"
    end_month = f"{end_year}12"

    if st.sidebar.button("데이터 조회하기", type="primary"):
        with st.spinner("한국부동산원에서 통계 데이터를 가져오는 중..."):
            df = fetch_reb_stats(API_KEY, region_code, start_month, end_month)
            
            if not df.empty:
                st.subheader(f"📊 {selected_region} 아파트 가격지수 변동 현황")
                
                # Sample Data Visualization logic
                # Note: We must verify actual column names from the R-ONE JSON response.
                # Usually, it provides 'research_date' and 'indices'.
                
                # For demonstration, we show the raw data
                with st.expander("원천 데이터 보기"):
                    st.write(df)
                
                # Logic to handle Plotly Chart
                # Attempt to find Date and Value columns
                date_col = next((c for c in df.columns if 'date' in c.lower() or 'month' in c.lower()), None)
                val_col = next((c for c in df.columns if 'idx' in c.lower() or 'val' in c.lower() or 'price' in c.lower()), None)
                
                if date_col and val_col:
                    df = df.sort_values(by=date_col)
                    fig = px.line(
                        df, 
                        x=date_col, 
                        y=val_col, 
                        title=f"{selected_region} 시계열 통계 차트",
                        markers=True,
                        labels={date_col: "조회년월", val_col: "지수/가격"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("차트를 그리기 위한 컬럼을 식별하는 중입니다. 상단 '원천 데이터'를 확인해 주세요.")
            else:
                st.error("데이터를 불러오지 못했습니다. 인증키가 해당 통계 서비스(아파트가격지수)에 권한이 있는지 확인이 필요합니다.")

    # Educational Footer
    st.divider()
    st.caption("본 대시보드는 한국부동산원(REB) 부동산통계정보시스템 오픈API를 활용합니다.")
    with st.expander("💡 API 사용 가이드"):
        st.markdown("""
        1. **R-ONE 키의 특징:** 공공데이터포털(data.go.kr) 키와 호환되지 않으며, [부동산통계정보시스템](https://www.reb.or.kr/r-one)에서 별도 승인받아야 합니다.
        2. **엔드포인트:** 신청하신 통계 종류(지수, 평균가격, 실거래가격지수 등)에 따라 URL 끝부분이 달라집니다.
        3. **데이터 제한:** 월간 통계 데이터는 보통 매월 중순에 업데이트됩니다.
        """)

if __name__ == "__main__":
    main()
