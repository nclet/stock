import streamlit as st
# FinanceDataReader를 사용하여 안정적으로 데이터 로드
import FinanceDataReader as fdr 
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
import time 

# 페이지 설정
st.set_page_config(layout="wide", page_title="FDR 기반 코스피/코스닥 매매 주체별 자금 흐름 분석")

# --- 앱 제목 및 설명 ---
st.title("💰 KOSPI/KOSDAQ 매매 주체별 자금 흐름 분석 (FDR)")
st.markdown("""
이 대시보드는 **FinanceDataReader**를 사용하여 코스피(KOSPI)와 코스닥(KOSDAQ) 시장의 **개인, 기관, 외국인**의 일별 순매수/순매도(자금 유입/이탈) 추이를 시각화합니다.
FDR 데이터셋은 누적 순매수 금액을 제공하므로, **일별 순매수 금액**을 계산하여 표시합니다.
""")

# FDR에서 사용하는 시장 코드 및 컬럼명 매핑
FDR_MAPPING = {
    "KOSPI (코스피)": {
        "code": "KOSPI",
        "investors": ['Individual(KOSPI)', 'Foreign(KOSPI)', 'Institution(KOSPI)']
    },
    "KOSDAQ (코스닥)": {
        "code": "KOSDAQ",
        "investors": ['Individual(KOSDAQ)', 'Foreign(KOSDAQ)', 'Institution(KOSDAQ)']
    }
}

# --- 1. 데이터 로드 함수 (FinanceDataReader 사용 및 캐싱) ---

@st.cache_data(ttl=3600) # 1시간마다 데이터 갱신
def load_investor_data(market_name, start_date, end_date):
    """
    FinanceDataReader를 사용하여 투자 주체별 순매수 누적 데이터를 가져오고,
    일별 순매수 금액으로 변환하여 반환합니다.
    """
    market_config = FDR_MAPPING[market_name]
    investor_cols = market_config["investors"]
    
    try:
        # FDR 코드: KRX/INVESTOR는 모든 시장의 투자 주체별 누적 순매수 데이터를 제공합니다.
        df_raw = fdr.DataReader('KRX/INVESTOR', start_date, end_date)
        
        if df_raw.empty:
            st.warning("선택한 기간에 대한 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 1. 선택한 시장의 투자 주체 컬럼만 추출
        df_market = df_raw[investor_cols].copy()
        
        # 2. 누적 금액을 일별 순매수 금액 (Net Flow)으로 변환 (.diff())
        # 첫 날의 NaN 값은 0으로 채웁니다.
        df_net_flow = df_market.diff().fillna(0)
        
        # 3. 컬럼 이름 간소화 (시각화용)
        new_cols = ['개인', '외국인', '기관']
        df_net_flow.columns = new_cols
        
        # 4. Long Format으로 변환하여 Plotly에 적합하게 만듦
        df_long = df_net_flow.reset_index().melt(
            id_vars='Date', 
            value_vars=new_cols,
            var_name='Investor', 
            value_name='Net_Flow'
        )
        
        return df_long
    
    except Exception as e:
        st.error(f"FinanceDataReader 데이터 로드 중 오류가 발생했습니다: {e}")
        st.warning("네트워크 연결 또는 FinanceDataReader 라이브러리에 문제가 있을 수 있습니다.")
        return pd.DataFrame()


# --- 2. 메인 본문 사용자 입력 ---
st.header("⚙️ 분석 옵션 선택")

with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_market_name = st.selectbox(
            "📊 분석할 시장 선택",
            list(FDR_MAPPING.keys()),
            key="market_select"
        )
    
    today = date.today()
    default_start_date = today - timedelta(days=365)
    
    with col2:
        start_date = st.date_input("🗓️ 시작 날짜", default_start_date, key="start_date")
    
    with col3:
        end_date = st.date_input("🗓️ 종료 날짜", today, key="end_date")

    st.markdown("---")
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 1, 1])
    
    with col_btn_center:
        run_analysis = st.button("자금 흐름 분석 시작", type="primary", use_container_width=True)


# --- 3. 실행 로직 및 시각화 ---
if run_analysis:
    if start_date > end_date:
        st.error("❌ 시작 날짜는 종료 날짜보다 빠를 수 없습니다.")
    else:
        
        # 데이터 로드
        with st.spinner(f"{selected_market_name}의 매매 주체별 자금 흐름 데이터를 로드 중..."):
            # FDR은 datetime.date 객체를 직접 처리할 수 있습니다.
            df_long = load_investor_data(
                selected_market_name, 
                start_date, 
                end_date
            )

        if df_long.empty:
            st.warning("선택한 시장과 기간에 대한 매매 주체별 데이터를 찾을 수 없습니다. (날짜 범위를 확인해 주세요.)")
        else:
            st.subheader(f"📈 {selected_market_name} 일별 매매 주체별 순매수/순매도 추이")
            
            # ----------------------------------------------------
            # Plotly 시각화: 누적 막대 차트로 순매수/순매도 금액 표시
            # ----------------------------------------------------
            
            fig = px.bar(
                df_long,
                x='Date',
                y='Net_Flow',
                color='Investor', # '개인', '외국인', '기관'
                title=f'{selected_market_name} 일별 매매 주체별 순매수/순매도 (단위: 원)',
                labels={
                    'Net_Flow': '순매수 금액 (원)',
                    'Date': '날짜',
                    'Investor': '투자 주체'
                },
                template='plotly_white',
                barmode='relative' 
            )
            
            # 차트 레이아웃 설정 및 0 기준선 추가
            fig.update_layout(
                xaxis_title="날짜",
                yaxis_title="순매수 금액 (원)",
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
            
            # --- 4. 데이터 테이블 ---
            st.subheader("📋 데이터 미리보기 (일별 순매수/순매도 금액)")
            # Long Format을 다시 Wide Format으로 변환하여 테이블에 표시
            df_wide = df_long.pivot(index='Date', columns='Investor', values='Net_Flow')
            st.dataframe(df_wide.sort_index(ascending=False), use_container_width=True)
