import streamlit as st
import requests
from json.decoder import JSONDecodeError
import FinanceDataReader as fdr
import pyupbit
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import datetime
import numpy as np

# ---------------------------------------------------------------------------------
# 1. 데이터 처리 및 지표 계산 함수
# ---------------------------------------------------------------------------------

@st.cache_data
def get_stock_listing(market):
    try:
        df = fdr.StockListing(market)
        if 'Code' not in df.columns and 'Symbol' in df.columns:
            df.rename(columns={'Symbol': 'Code'}, inplace=True)
        df['Code'] = df['Code'].astype(str)
        df['label'] = df['Name'] + ' (' + df['Code'] + ')'
        return df
    except Exception as e:
        st.error(f"종목 리스트 오류: {e}")
        return pd.DataFrame()

@st.cache_data
def get_coin_listing():
    try:
        url = "https://api.upbit.com/v1/market/all"
        response = requests.get(url, params={'isDetails': 'false'})
        all_markets = response.json()
        krw_markets = [m for m in all_markets if m['market'].startswith('KRW-')]
        df_coin = pd.DataFrame(krw_markets)
        df_coin.rename(columns={'market': 'Code', 'korean_name': 'korean_name'}, inplace=True)
        df_coin['label'] = df_coin['korean_name'] + ' (' + df_coin['Code'].str.replace('KRW-', '') + ')'
        return df_coin
    except Exception as e:
        st.error(f"코인 리스트 오류: {e}")
        return pd.DataFrame()

def calculate_advanced_factors(df):
    """거래량, ATR, 이격도 등 심화 팩터 계산"""
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Disparity'] = (df['Close'] / df['MA20']) * 100
    df['Next_5d_Return'] = (df['Close'].shift(-5) / df['Close'] - 1) * 100
    return df

def find_candle_patterns(df):
    """캔들 패턴 정의 및 필터 적용"""
    df['is_hammer'] = False
    df['is_bullish_engulfing'] = False
    df['is_bearish_engulfing'] = False
    df['is_shooting_star'] = False

    for i in range(1, len(df)):
        open_p, close_p, high_p, low_p = df.iloc[i][['Open', 'Close', 'High', 'Low']]
        prev_open, prev_close = df.iloc[i-1][['Open', 'Close']]
        body_length = abs(close_p - open_p)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p
        
        is_reliable = df.iloc[i]['Vol_Ratio'] > 1.2
        
        if body_length > 0 and lower_shadow > 2 * body_length and upper_shadow < body_length:
            if close_p > open_p and df.iloc[i]['Disparity'] < 100:
                df.at[df.index[i], 'is_hammer'] = True
        
        if (prev_close < prev_open and close_p > open_p and 
            open_p < prev_close and close_p > prev_open and is_reliable):
            df.at[df.index[i], 'is_bullish_engulfing'] = True

        if (prev_close > prev_open and close_p < open_p and 
            open_p > prev_close and close_p < prev_open and is_reliable):
            df.at[df.index[i], 'is_bearish_engulfing'] = True

        if body_length > 0 and upper_shadow > 2 * body_length and lower_shadow < body_length:
            if close_p < open_p and df.iloc[i]['Disparity'] > 100:
                df.at[df.index[i], 'is_shooting_star'] = True
    return df

# ---------------------------------------------------------------------------------
# 2. UI 및 메인 로직
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="심화 캔들 패턴 분석기", layout="wide")
st.markdown("<h1 style='text-align: center;'>📈 심화 캔들 패턴 & 백테스팅 분석기</h1>", unsafe_allow_html=True)

# --- 상단 설정 섹션 ---
st.subheader("1. 분석 옵션 설정")
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    market_choice = st.radio("분석 시장", ['한국 주식 (KRX)', '미국 주식 (NASDAQ)', '코인 (Upbit)'], horizontal=True)
with col2:
    period = st.radio("봉 주기", ['일봉', '주봉'], horizontal=True)

if '주식' in market_choice:
    listing = get_stock_listing('KRX' if '한국' in market_choice else 'NASDAQ')
else:
    listing = get_coin_listing()

with col3:
    if not listing.empty:
        selected_label = st.selectbox("종목 선택", listing['label'].tolist())
        ticker = listing[listing['label'] == selected_label]['Code'].values[0]

col4, col5 = st.columns(2)
with col4:
    start_date = st.date_input("시작일", datetime.date.today() - datetime.timedelta(days=365))
with col5:
    end_date = st.date_input("종료일", datetime.date.today())

st.divider()

# --- 실행 버튼 및 결과 ---
if st.button("🚀 데이터 분석 및 백테스팅 시작", use_container_width=True, type="primary"):
    with st.spinner('데이터를 분석 중입니다...'):
        if '주식' in market_choice:
            df = fdr.DataReader(ticker, start_date, end_date)
        else:
            interval = 'day' if period == '일봉' else 'week'
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=400)

        if df is not None and not df.empty:
            df = calculate_advanced_factors(df)
            df = find_candle_patterns(df)
            
            # --- 승률 리포트 ---
            st.subheader("📊 패턴별 5일 후 승률 리포트")
            pattern_cols = ['is_hammer', 'is_bullish_engulfing', 'is_bearish_engulfing', 'is_shooting_star']
            stat_cols = st.columns(len(pattern_cols))
            
            for idx, p_col in enumerate(pattern_cols):
                p_data = df[df[p_col] == True].copy()
                valid_p_data = p_data.dropna(subset=['Next_5d_Return'])
                if not valid_p_data.empty:
                    win_rate = (valid_p_data['Next_5d_Return'] > 0).mean() * 100
                    avg_ret = valid_p_data['Next_5d_Return'].mean()
                    stat_cols[idx].metric(p_col.replace('is_', '').upper(), f"{len(p_data)}회", f"{win_rate:.1f}% 승률")
                    stat_cols[idx].caption(f"평균 수익: {avg_ret:.2f}%")
                else:
                    stat_cols[idx].write(f"{p_col.split('_')[-1].upper()}\n발견 안됨")

            # --- 차트 시각화 ---
            st.subheader("🕯️ 캔들 패턴 시각화")
            apds = []
            if df['is_bullish_engulfing'].any():
                bull_data = df['Low'] * 0.97
                bull_data[~df['is_bullish_engulfing']] = np.nan
                apds.append(mpf.make_addplot(bull_data, type='scatter', markersize=120, marker='^', color='green'))
            
            if df['is_bearish_engulfing'].any():
                bear_data = df['High'] * 1.03
                bear_data[~df['is_bearish_engulfing']] = np.nan
                apds.append(mpf.make_addplot(bear_data, type='scatter', markersize=120, marker='v', color='red'))
            
            fig, axlist = mpf.plot(df, type='candle', volume=True, addplot=apds, 
                               style='charles', figratio=(16, 9), returnfig=True,
                               title=f"\n{selected_label} Analysis")
            st.pyplot(fig)
        else:
            st.error("데이터 로드 실패")

# --- 캔들 패턴 설명 (Expander) ---
st.divider()
with st.expander("캔들 패턴 참고자료 📖"):
    st.markdown("""
    이 앱에서 분석하는 주요 캔들 패턴에 대한 간단한 설명입니다.
    
    - **🔎도지(Doji)**: 시가와 종가가 거의 같은 십자형 캔들입니다. 매수자와 매도자가 서로 힘의 균형을 이루고 있다는 것을 나타내며, 추세 전환의 신호일 수 있습니다.

    - **🔎망치형 (Hammer)**: 긴 아래 꼬리와 짧은 몸통을 가진 캔들입니다. 하락 추세에서 나타나면 바닥을 확인하고 반등할 가능성을 시사합니다.

    - **장악형 (Engulfing)**: 현재 캔들이 이전 캔들의 몸통을 완전히 감싸는 형태입니다.

        - **📈상승 장악형 (Bullish Engulfing)**: 큰 양봉이 이전 음봉을 감싸는 형태로, 강한 매수세와 상승 반전을 예고합니다.

        - **📉하락 장악형 (Bearish Engulfing)**: 큰 음봉이 이전 양봉을 감싸는 형태로, 강한 매도세와 하락 반전을 예고합니다.

    - **📈모닝 스타 (Morning Star)**: 하락 추세에서 나타나는 3개의 캔들 패턴입니다. 큰 음봉, 작은 캔들, 그리고 큰 양봉이 순서대로 나타나며, 강력한 상승 반전 신호입니다.

    - **샛별형 (Star)**: 몸통이 이전 캔들의 몸통 위에 위치하는 캔들입니다.

        - **📈상승 샛별형 (Bullish Star)**: 큰 음봉 이후 작은 캔들이 나타나며, 상승 반전 가능성을 시사합니다.

        - **📉하락 샛별형 (Bearish Star)**: 큰 양봉 이후 작은 캔들이 나타나며, 하락 반전 가능성을 시사합니다.

    - **🔎십자 샛별형 (Doji Star)**: 샛별형의 작은 캔들이 도지 형태인 경우입니다. 추세 전환의 신호로 더 강력하게 해석됩니다.

    - **📈관통형 (Piercing Line)**: 하락 추세에서 첫 날 큰 음봉이 나타나고, 다음 날 양봉이 나타나는데, 이 양봉의 종가가 이전 날 음봉의 중간 지점을 뚫고 올라가는 형태입니다. 하락 추세가 끝날 수 있다는 긍정적인 신호로 해석됩니다.

    - **📉흑운형 (Dark Cloud Cover)**: 상승 추세에서 첫 날 양봉이 나타나고, 다음 날 음봉이 나타나는데, 이 음봉의 종가가 이전 날 양봉의 중간 지점을 뚫고 내려오는 형태입니다. 매도세가 강해져 상승 추세가 꺾일 수 있다는 부정적인 신호입니다.

    - **📈적삼병 (Three White Soldiers)**: 3일 연속 양봉이 나타나는 패턴입니다. 각 양봉의 종가가 이전 날의 종가보다 높게 끝나며, 강력한 상승 추세의 시작을 알리는 신호입니다.

    - **📉흑삼병 (Three Black Crows)**: 3일 연속 음봉이 나타나는 패턴입니다. 각 음봉의 종가가 이전 날의 종가보다 낮게 끝나며, 강력한 하락 추세의 시작을 알리는 신호입니다.

    - **📉유성형 (Shooting Star)**: 긴 위 꼬리와 짧은 몸통을 가진 캔들입니다. 상승 추세에서 나타나면 고점에서 매수세가 약해졌다는 것을 보여주며, 하락 반전 가능성을 시사합니다.

    - **📉교수형 (Hanging Man)**: 망치형과 모양은 비슷하지만, 상승 추세에서 나타납니다. 주가가 고점에서 하락할 가능성이 있다는 경고 신호로 해석됩니다.

    """)

with st.expander("📝 분석 상세 데이터 확인"):
    if 'df' in locals():
        st.dataframe(df.tail(50))
    else:
        st.info("먼저 분석을 시작해주세요.")
