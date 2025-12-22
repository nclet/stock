import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ------------------------------------------------------------------------------
# 1. 설정 및 보안 (Streamlit Secrets)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="부동산 통계 분석", layout="wide")

# API 키 불러오기
try:
    API_KEY = st.secrets["MOLIT_KEY"]
except KeyError:
    st.error("❌ Streamlit Secrets에 'MOLIT_KEY'가 설정되지 않았습니다.")
    st.stop()

# 지역 코드 매핑
REGION_MAP = {
    "전국": "00", "서울특별시": "11", "경기도": "41", "인천광역시": "28",
    "부산광역시": "26", "대구광역시": "27", "광주광역시": "29", "대전광역시": "30",
    "울산광역시": "31", "세종특별자치시": "36"
}

# ------------------------------------------------------------------------------
# 2. 데이터 수집 함수 (에러 핸들링 강화)
# ------------------------------------------------------------------------------
def fetch_property_data(region_code, start_month, end_month):
    # R-ONE API 공식 엔드포인트 확인 필요 (아파트매매가격지수)
    url = "https://www.reb.or.kr/r-one/openapi/statistics/propertyPriceIndex"
    
    params = {
        "key": API_KEY,
        "format": "json",
        "startmonth": start_month,
        "endmonth": end_month,
        "region": region_code,
        "p_type": "01"  # 01: 아파트
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        # 1. HTTP 상태 코드 확인
        if response.status_code != 200:
            return {"error": f"HTTP 오류: {response.status_code}"}

        # 2. 응답 내용이 비어있는지 확인
        if not response.text.strip():
            return {"error": "API가 빈 응답을 반환했습니다."}

        # 3. JSON 파싱 시도
        try:
            data = response.json()
        except Exception:
            # JSON이 아니면 에러 메시지(HTML 등)일 가능성이 높음
            return {"error": f"JSON 파싱 실패. 응답 내용: {response.text[:100]}..."}

        return data

    except Exception as e:
        return {"error": f"네트워크 오류: {str(e)}"}

# ------------------------------------------------------------------------------
# 3. 메인 UI
# ------------------------------------------------------------------------------
def main():
    st.title("🏠 아파트 매매가격지수 분석")
    st.info("국토교통부(한국부동산원) Open API 데이터를 사용합니다.")

    with st.sidebar:
        st.header("설정")
        region_name = st.selectbox("지역", list(REGION_MAP.keys()))
        s_year = st.number_input("시작 연도", 2020, 2025, 2023)
        e_year = st.number_input("종료 연도", 2020, 2025, 2024)
        
        btn = st.button("데이터 조회", type="primary")

    if btn:
        start_m = f"{s_year}01"
        end_m = f"{e_year}12"
        
        with st.spinner("데이터를 불러오는 중..."):
            result = fetch_property_data(REGION_MAP[region_name], start_m, end_m)
            
            # 에러 발생 시 출력
            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])
                st.info("💡 팁: API 키가 '활용승인' 상태인지, 혹은 일일 트래픽 초과인지 확인하세요.")
                return

            # 데이터프레임 변환
            try:
                # R-ONE API는 보통 'item' 리스트에 데이터가 담겨 옵니다.
                items = result.get("item", [])
                if not items:
                    st.warning("조회된 데이터가 없습니다. 기간을 조절해보세요.")
                    return
                
                df = pd.DataFrame(items)
                
                # 컬럼명 정리 (API 응답에 따라 수정될 수 있음)
                # 보통 'research_date'가 날짜, 'indices'가 지수 값입니다.
                if 'research_date' in df.columns:
                    df = df.rename(columns={'research_date': '날짜', 'indices': '지수'})
                    df['지수'] = pd.to_numeric(df['지수'])
                    
                    # 차트 출력
                    fig = px.line(df, x='날짜', y='지수', title=f"{region_name} 아파트 매매가격지수 추이")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("상세 데이터")
                    st.dataframe(df)
                else:
                    st.write("응답 데이터 구조:", result)
            
            except Exception as e:
                st.error(f"데이터 처리 중 오류 발생: {e}")
                st.write("전체 응답:", result)

if __name__ == "__main__":
    main()
