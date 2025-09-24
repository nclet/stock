import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# @st.cache_data 데코레이터는 데이터를 캐싱하여 앱의 속도를 향상시킵니다.
@st.cache_data
def get_stock_data(ticker):
    """
    지정된 티커의 주식 데이터를 가져옵니다.
    :param ticker: 주식 티커 심볼 (예: 'AAPL')
    :return: pandas DataFrame 형태의 주식 데이터
    """
    try:
        # yfinance를 사용하여 데이터 가져오기
        stock = yf.Ticker(ticker)
        history = stock.history(period="1y")
        info = stock.info
        
        # 주식 데이터가 비어 있는지 확인
        if history.empty:
            return None, None, "데이터를 찾을 수 없습니다. 올바른 티커 심볼을 입력했는지 확인하세요."

        # Market Cap (시가총액) 계산
        # 시가총액 = 종가 * 발행 주식수
        if 'sharesOutstanding' in info:
            history['MarketCap'] = history['Close'] * info['sharesOutstanding']
        else:
            history['MarketCap'] = 0  # 발행 주식수 정보가 없을 경우 0으로 설정
        
        return history, info, None
    except Exception as e:
        return None, None, f"데이터를 가져오는 중 오류가 발생했습니다: {e}"

# Streamlit 앱의 제목과 설명
st.set_page_config(
    page_title="미국 주식 시장 시각화",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 미국 주식 시장 일일 거래량 및 시가총액 시각화")
st.markdown("관심 있는 미국 주식의 일일 거래량과 시가총액 변화를 확인해보세요.")

# --- 사이드바 ---
st.sidebar.header("주식 선택 및 설정")
# 사용자에게 티커를 입력받는 셀렉트 박스
ticker_list = ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM')
ticker = st.sidebar.selectbox("티커 심볼을 선택하거나 입력하세요:", ticker_list)

if st.sidebar.button("데이터 분석"):
    with st.spinner('데이터를 불러오는 중입니다... 잠시만 기다려주세요.'):
        data, info, error_message = get_stock_data(ticker)
        
    if error_message:
        st.error(error_message)
    elif data is not None:
        st.subheader(f"📊 {info.get('longName', ticker)} ({ticker}) 분석")
        
        # --- 시각화 ---
        
        # 1. 일일 거래량 차트
        st.markdown("#### 일일 거래량 (Daily Volume)")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color='rgb(34,139,34)',
            name='일일 거래량'
        ))
        fig_volume.update_layout(
            title=f'{ticker} 일일 거래량',
            xaxis_title='날짜',
            yaxis_title='거래량',
            bargap=0.2
        )
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # 2. 시가총액 차트
        st.markdown("#### 시가총액 (Market Capitalization)")
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(
            x=data.index,
            y=data['MarketCap'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='시가총액'
        ))
        fig_cap.update_layout(
            title=f'{ticker} 시가총액 변화',
            xaxis_title='날짜',
            yaxis_title='시가총액',
            yaxis_tickformat='.2s' # 축 값 포맷 설정
        )
        st.plotly_chart(fig_cap, use_container_width=True)

    else:
        st.info("데이터를 분석하려면 왼쪽 사이드바에서 티커를 선택하고 '데이터 분석' 버튼을 눌러주세요.")
else:
    st.info("데이터를 분석하려면 왼쪽 사이드바에서 티커를 선택하고 '데이터 분석' 버튼을 눌러주세요.")
