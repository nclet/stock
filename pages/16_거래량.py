import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# 페이지 설정
st.set_page_config(layout="wide", page_title="코스피/코스닥 매매 주체별 자금 흐름 분석")

# --- 앱 제목 및 설명 ---
st.title("💰 KOSPI/KOSDAQ 매매 주체별 자금 흐름 분석")
st.markdown("""
이 대시보드는 **코스피(KOSPI)**와 **코스닥(KOSDAQ)** 시장에 대한 **개인, 기관, 외국인**의 일별 순매수/순매도(자금 유입/이탈) 추이를 시각화합니다.
양수 값은 순매수(자금 유입)를, 음수 값은 순매도(자금 이탈)를 나타냅니다.
""")

# KOSPI/KOSDAQ 매매 주체별 데이터의 'fdr' 인덱스 매핑
INDEX_MAPPING = {
    "KOSPI (코스피)": "KOSPI",
    "KOSDAQ (코스닥)": "KOSDAQ"
}

# --- 1. 데이터 로드 함수 (캐싱 사용) ---
@st.cache_data
def load_investor_data(market_fdr_code, start_date, end_date):
    """
    FinanceDataReader를 사용하여 KOSPI 또는 KOSDAQ의 투자 주체별
    순매수/순매도 데이터를 가져옵니다.
    """
    try:
        # FinanceDataReader의 매매 주체별 데이터를 가져오는 함수
        data = fdr.DataReader(market_fdr_code, start_date, end_date)
        
        if data.empty:
            return pd.DataFrame()
            
        # 데이터프레임 클리닝 및 컬럼 이름 표준화
        data.columns = [col.replace('외국인', 'Foreigner').replace('기관', 'Institution').replace('개인', 'Individual') for col in data.columns]
        
        target_cols = ['Individual', 'Institution', 'Foreigner']
        present_cols = [col for col in target_cols if col in data.columns]

        if len(present_cols) < 3:
             st.warning(f"경고: '개인', '기관', '외국인' 순매수 데이터가 DataFrame에 포함되어 있지 않습니다. 현재 컬럼: {data.columns.tolist()}")
             return pd.DataFrame()
        
        data = data[present_cols]
        data = data.rename_axis('Date')
        
        # 시각화를 위해 데이터의 형식을 Long Format으로 변환
        data_long = data.reset_index().melt(
            id_vars='Date', 
            value_vars=present_cols,
            var_name='Investor', 
            value_name='Net_Flow'
        )
        
        # 투자자 이름을 한글로 다시 매핑 (시각화용)
        investor_mapping = {
            'Individual': '개인',
            'Institution': '기관',
            'Foreigner': '외국인'
        }
        data_long['Investor (한글)'] = data_long['Investor'].map(investor_mapping)
        
        return data_long
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# --- 2. 메인 본문 사용자 입력 (사이드바에서 이동) ---
st.header("⚙️ 분석 옵션 선택")

# 컨테이너를 사용하여 옵션 영역을 깔끔하게 구분
with st.container(border=True):
    
    # 3개의 컬럼을 생성하여 시장 선택과 날짜 입력을 나란히 배치
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_market_name = st.selectbox(
            "📊 분석할 시장 선택",
            list(INDEX_MAPPING.keys()),
            key="market_select"
        )
        market_fdr_code = INDEX_MAPPING[selected_market_name]
    
    # 기간 설정
    today = date.today()
    default_start_date = today - timedelta(days=365)
    
    with col2:
        start_date = st.date_input("🗓️ 시작 날짜", default_start_date, key="start_date")
    
    with col3:
        end_date = st.date_input("🗓️ 종료 날짜", today, key="end_date")

    # 실행 버튼을 중앙에 배치
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
            df_long = load_investor_data(market_fdr_code, start_date, end_date)

        if df_long.empty:
            st.warning("선택한 시장과 기간에 대한 매매 주체별 데이터를 찾을 수 없습니다. 날짜를 확인하거나 데이터 소스에 문제가 없는지 확인해주세요.")
        else:
            st.subheader(f"📈 {selected_market_name} 일별 매매 주체별 순매수/순매도 추이")
            
            # ----------------------------------------------------
            # Plotly 시각화: 누적 막대 차트로 순매수/순매도 금액 표시
            # ----------------------------------------------------
            
            # 누적 막대 차트 생성 
            fig = px.bar(
                df_long,
                x='Date',
                y='Net_Flow',
                color='Investor (한글)',
                title=f'{selected_market_name} 일별 매매 주체별 순매수/순매도 (단위: 천원)',
                labels={
                    'Net_Flow': '순매수 금액 (천원)',
                    'Date': '날짜',
                    'Investor (한글)': '투자 주체'
                },
                template='plotly_white',
                barmode='relative' 
            )
            
            # 차트 레이아웃 설정
            fig.update_layout(
                xaxis_title="날짜",
                yaxis_title="순매수 금액 (천원)",
                legend_title="투자 주체",
                hovermode="x unified",
                # 0 기준선 추가
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
            # Long format 데이터를 다시 Wide format으로 변환하여 테이블에 표시
            df_wide = df_long.pivot(index='Date', columns='Investor (한글)', values='Net_Flow')
            st.dataframe(df_wide.sort_index(ascending=False))
