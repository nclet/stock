import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import datetime

# --- 설정 및 데이터 로드 함수 ---
def load_investor_data(market, start_date):
    """
    FinanceDataReader를 사용해 시장별 투자자 매매 동향(수급) 데이터를 로드합니다.
    시장 코드: KOSPI ('KOSPI'), KOSDAQ ('KOSDAQ')
    """
    try:
        # 투자자 매매 동향 데이터 로드
        df = fdr.read_investor_trade(market, start=start_date)
        
        # '합계' 컬럼 제거 (필요 없는 경우)
        if '합계' in df.columns:
            df = df.drop(columns=['합계'])

        # 개인, 기관, 외국인 순매수 금액으로 변환 (매수 - 매도)
        # FinanceDataReader는 이미 순매수 금액을 제공하므로 별도 계산 불필요
        return df

    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생 ({market}): {e}")
        return pd.DataFrame()

def load_index_data(symbol, start_date):
    """
    FinanceDataReader를 사용해 시장 지수 데이터를 로드합니다.
    KOSPI: 'KS11', KOSDAQ: 'KQ11'
    """
    try:
        index_df = fdr.DataReader(symbol, start=start_date)
        return index_df['Close']
    except Exception as e:
        st.error(f"지수 데이터 로드 중 오류 발생 ({symbol}): {e}")
        return pd.Series()

# --- Streamlit 앱 메인 함수 ---
def main():
    st.title("🇰🇷 코스피/코스닥 일별 투자자 수급 시각화")
    st.markdown("---")

    # 1. 사이드바 설정 (필터링 위젯)
    st.sidebar.header("📊 데이터 설정")

    # 시장 선택
    market_options = {'코스피': 'KOSPI', '코스닥': 'KOSDAQ'}
    selected_market_name = st.sidebar.selectbox(
        "시장 선택",
        list(market_options.keys())
    )
    selected_market_code = market_options[selected_market_name]

    # 기간 설정 (최근 1년 기본)
    end_date = datetime.date.today()
    default_start_date = end_date - datetime.timedelta(days=365)
    start_date = st.sidebar.date_input(
        "시작 날짜",
        value=default_start_date
    )

    # 2. 데이터 로드
    investor_df = load_investor_data(selected_market_code, start_date)
    
    # 지수 데이터 로드 (시각화에 활용)
    index_symbol = 'KS11' if selected_market_code == 'KOSPI' else 'KQ11'
    index_series = load_index_data(index_symbol, start_date)

    if investor_df.empty:
        st.warning("선택하신 기간 동안의 데이터가 없습니다. 날짜를 다시 설정해주세요.")
        return

    # 3. 데이터 시각화
    st.subheader(f"📈 {selected_market_name} 일별 투자자 순매수 추이 (단위: 백만원)")

    # 3-1. 수급 데이터 차트
    # '개인', '기관', '외국인' 컬럼만 선택하여 시각화
    flow_columns = ['개인', '기관', '외국인']
    flow_df_to_plot = investor_df[flow_columns]
    
    # 누적 순매수 데이터 추가 (추세 파악 용이)
    cumulative_df = flow_df_to_plot.cumsum()
    cumulative_df.columns = [f'{col} (누적)' for col in flow_columns]

    tab1, tab2 = st.tabs(["일별 순매수", "누적 순매수"])

    with tab1:
        st.bar_chart(flow_df_to_plot) # 일별 순매수는 막대 차트가 직관적
        st.caption("막대 차트: 일별 순매수/순매도 금액. 0 이상: 순매수, 0 미만: 순매도")
        st.dataframe(investor_df.tail(10)) # 최근 데이터 테이블로 표시

    with tab2:
        st.line_chart(cumulative_df) # 누적 순매수는 꺾은선 차트가 추세 파악에 용이
        st.caption("꺾은선 차트: 누적 순매수/순매도 금액 추이")
        
    st.markdown("---")
    
    # 3-2. 지수와 수급 비교 시각화 (선택 사항)
    if not index_series.empty:
        st.subheader("📊 지수와 누적 수급 비교")
        
        # 지수와 누적 수급 데이터를 합치기 위해 정규화 (스케일 조정) 필요
        # 간단하게 최대값으로 나눠 스케일을 맞춥니다.
        combined_df = pd.DataFrame({
            '지수 (종가)': index_series,
        }).join(cumulative_df)
        
        # Streamlit의 Line Chart는 여러 시리즈를 한 번에 보여줄 때 유용합니다.
        st.line_chart(combined_df.dropna())
        st.caption("주의: 지수와 수급은 서로 다른 스케일이므로, 추세 비교용으로만 활용하세요.")

if __name__ == "__main__":
    main()
