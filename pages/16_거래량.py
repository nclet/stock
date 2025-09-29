import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
from pykrx.stock import get_market_trading_value_by_date

# --------------------------
# 페이지 설정
# --------------------------
st.set_page_config(layout="wide", page_title="pykrx 기반 코스피/코스닥 매매 주체별 자금 흐름 분석")

# --------------------------
# 앱 제목 및 설명
# --------------------------
st.title("💰 KOSPI/KOSDAQ 매매 주체별 자금 흐름 분석 (pykrx)")
st.markdown("""
이 대시보드는 **pykrx** 라이브러리를 사용하여  
코스피(KOSPI)와 코스닥(KOSDAQ) 시장의 **개인, 기관, 외국인**의  
일별 매매 금액(억 원 단위)을 시각화합니다.
""")

# --------------------------
# 시장 매핑
# --------------------------
MARKET_MAPPING = {
    "KOSPI (코스피)": "KOSPI",
    "KOSDAQ (코스닥)": "KOSDAQ"
}

# --------------------------
# 데이터 로드 함수
# --------------------------
@st.cache_data(ttl=3600)
def load_investor_data(market, start_date, end_date):
    """
    pykrx에서 투자자별 매매 금액 데이터를 가져오고,
    Long Format으로 변환하여 반환
    """
    try:
        start_str, end_str = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        df_raw = get_market_trading_value_by_date(start_str, end_str, market)

        if df_raw.empty:
            return pd.DataFrame()

        # 개인, 외국인, 기관합계만 사용
        df_market = df_raw[["개인", "외국인", "기관합계"]].copy()
        df_market.rename(columns={"기관합계": "기관"}, inplace=True)

        # Long Format 변환
        df_long = df_market.reset_index().melt(
            id_vars="날짜",
            value_vars=["개인", "외국인", "기관"],
            var_name="Investor",
            value_name="Net_Flow"
        )
        df_long.rename(columns={"날짜": "Date"}, inplace=True)

        return df_long

    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# --------------------------
# 사용자 입력
# --------------------------
st.header("⚙️ 분석 옵션 선택")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_market_name = st.selectbox(
            "📊 분석할 시장 선택",
            list(MARKET_MAPPING.keys())
        )

    today = date.today()
    default_start_date = today - timedelta(days=180)

    with col2:
        start_date = st.date_input("🗓️ 시작 날짜", default_start_date)

    with col3:
        # pykrx는 장 마감일까지만 데이터 제공
        end_date = st.date_input("🗓️ 종료 날짜", today - timedelta(days=1))

    st.markdown("---")
    col_btn_center = st.columns([3, 1, 3])[1]
    with col_btn_center:
        run_analysis = st.button("자금 흐름 분석 시작", type="primary", use_container_width=True)

# --------------------------
# 실행 로직
# --------------------------
if run_analysis:
    if start_date > end_date:
        st.error("❌ 시작 날짜는 종료 날짜보다 빠를 수 없습니다.")
    else:
        with st.spinner(f"{selected_market_name} 매매 주체별 자금 흐름 데이터를 불러오는 중..."):
            df_long = load_investor_data(
                MARKET_MAPPING[selected_market_name],
                start_date,
                end_date
            )

        if df_long.empty:
            st.warning("선택한 시장과 기간에 대한 데이터를 찾을 수 없습니다.")
        else:
            st.subheader(f"📈 {selected_market_name} 일별 매매 주체별 거래 금액 추이")

            # Plotly 시각화
            fig = px.bar(
                df_long,
                x="Date",
                y="Net_Flow",
                color="Investor",
                title=f"{selected_market_name} 일별 매매 주체별 거래 금액 (단위: 억원)",
                labels={"Net_Flow": "거래 금액 (억원)", "Date": "날짜", "Investor": "투자 주체"},
                template="plotly_white",
                barmode="relative"
            )

            fig.update_layout(
                xaxis_title="날짜",
                yaxis_title="거래 금액 (억원)",
                hovermode="x unified",
                shapes=[
                    dict(
                        type="line",
                        xref="paper", yref="y",
                        x0=0, y0=0, x1=1, y1=0,
                        line=dict(color="gray", width=1, dash="dot")
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Wide Format 테이블
            st.subheader("📋 데이터 미리보기")
            df_wide = df_long.pivot(index="Date", columns="Investor", values="Net_Flow")
            st.dataframe(df_wide.sort_index(ascending=False), use_container_width=True)
