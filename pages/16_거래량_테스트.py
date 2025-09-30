import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
from pykrx import stock # FinanceDataReader 대신 pykrx 사용

# --------------------------
# 페이지 설정
# --------------------------
st.set_page_config(layout="wide", page_title="pykrx 기반 KOSPI/KOSDAQ 매매 주체별 자금 흐름 분석")

st.title("💰 KOSPI/KOSDAQ 매매 주체별 자금 흐름 분석 (pykrx)")
st.markdown("""
**FinanceDataReader**의 해당 데이터 소스(**KRX/INVESTOR**)가 현재 불안정하여 **pykrx**로 데이터 로드 방식을 변경했습니다.
이 대시보드는 **pykrx** 라이브러리를 사용하여 **코스피(KOSPI)와 코스닥(KOSDAQ)** 시장의 
**개인, 기관, 외국인**의 일별 순매수/순매도(자금 유입/이탈) 추이를 시각화합니다.
""")

# --------------------------
# 시장 선택 옵션
# --------------------------
MARKET_MAPPING = {
    "KOSPI (코스피)": "KOSPI",
    "KOSDAQ (코스닥)": "KOSDAQ"
}

# --------------------------
# 상수 정의
# --------------------------
INVESTOR_COLUMNS = ['개인', '외국인', '기관합계']

# --------------------------
# 데이터 로드 함수
# --------------------------
@st.cache_data(ttl=3600)
def load_investor_data(market, start, end):
    """
    pykrx를 사용하여 KOSPI 또는 KOSDAQ 시장의 투자 주체별 순매수 금액을 로드합니다.
    """
    try:
        # pykrx의 get_market_net_purchases_by_investor 함수를 사용하여 데이터 로드
        df = stock.get_market_net_purchases_by_investor(
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            market
        )
        
        if df.empty:
            return pd.DataFrame()

        # 필요한 컬럼만 선택
        df = df[INVESTOR_COLUMNS].copy()
        
        # 인덱스(날짜)를 리셋하고 long format으로 변환
        df = df.reset_index().rename(columns={'날짜': 'Date'})
        
        df_long = df.melt(
            id_vars='Date',
            value_vars=INVESTOR_COLUMNS,
            var_name='Investor',
            value_name='Net_Flow'
        )
        
        return df_long
    except Exception as e:
        # 데이터 로드 실패 오류를 표시
        st.error(f"데이터 로드 중 오류 발생: {e}. pykrx 라이브러리 문제일 수 있습니다.")
        return pd.DataFrame()

# --------------------------
# 사용자 입력 (시장 선택 재도입)
# --------------------------
st.header("⚙️ 분석 옵션 선택")

col1, col2, col3 = st.columns(3)

with col1:
    # 시장 선택 UI 재도입
    selected_market_name = st.selectbox(
        "📊 분석할 시장 선택",
        list(MARKET_MAPPING.keys()),
        key="market_select"
    )
    selected_market = MARKET_MAPPING[selected_market_name] # 실제 티커명 (KOSPI/KOSDAQ)

today = date.today()
default_end_date = today - timedelta(days=1)  # 어제까지
default_start_date = default_end_date - timedelta(days=365)

with col2:
    start_date = st.date_input("🗓️ 시작 날짜", default_start_date, key="start_date")

with col3:
    end_date = st.date_input("🗓️ 종료 날짜", default_end_date, key="end_date")

st.markdown("---")
col_btn_left, col_btn_center, col_btn_right = st.columns([1, 1, 1])

with col_btn_center:
    run_analysis = st.button("자금 흐름 분석 시작", type="primary", use_container_width=True)

# --------------------------
# 실행 및 시각화
# --------------------------
if run_analysis:
    if start_date > end_date:
        st.error("❌ 시작 날짜는 종료 날짜보다 빠를 수 없습니다.")
    else:
        with st.spinner(f"{selected_market_name}의 매매 주체별 자금 흐름 데이터를 로드 중..."):
            
            # load_investor_data에 market 인수를 전달
            df_long = load_investor_data(selected_market, start_date, end_date)

        if df_long.empty:
            st.warning("선택한 시장과 기간에 대한 매매 주체별 데이터를 찾을 수 없습니다. (날짜 범위를 확인해 주세요.)")
        else:
            st.subheader(f"📈 {selected_market_name} 일별 매매 주체별 순매수/순매도 추이")

            # Plotly 차트
            fig = px.bar(
                df_long,
                x='Date',
                y='Net_Flow',
                color='Investor',
                title=f'{selected_market_name} 일별 매매 주체별 순매수/순매도 (단위: 백만원)',
                labels={'Net_Flow': '순매수 금액 (백만원)', 'Date': '날짜', 'Investor': '투자 주체'},
                template='plotly_white',
                barmode='relative'
            )

            fig.update_layout(
                xaxis_title="날짜",
                yaxis_title="순매수 금액 (백만원)",
                legend_title="투자 주체",
                hovermode="x unified",
                shapes=[
                    dict(
                        type='line',
                        xref='paper', yref='y',
                        x0=0, y0=0, x1=1, y1=0,
                        line=dict(color='gray', width=1, dash='dot')
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # 데이터 테이블
            st.subheader("📋 데이터 미리보기 (일별 순매수/순매도 금액)")
            # Pivot table을 만들기 전에 'Investor'의 '기관합계'를 '기관'으로 변경하여 표시 편의성을 높입니다.
            df_display = df_long.copy()
            df_display['Investor'] = df_display['Investor'].replace({'기관합계': '기관'})
            
            df_wide = df_display.pivot(index='Date', columns='Investor', values='Net_Flow')
            st.dataframe(df_wide.sort_index(ascending=False), use_container_width=True)
