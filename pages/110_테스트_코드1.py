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
    # 1. 거래량 필터 (20일 평균 대비 비율)
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # 2. ATR (변동성) 계산
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 3. 이격도 (추세 필터)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Disparity'] = (df['Close'] / df['MA20']) * 100
    
    # 4. 백테스팅용: 5거래일 후 수익률
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
        
        # 신뢰도 조건: 거래량 1.2배 이상
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
# 2. UI 및 메인 로직 (본문 위주 구성)
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="심화 캔들 패턴 분석기", layout="wide")
st.markdown("<h1 style='text-align: center;'>📈 심화 캔들 패턴 & 백테스팅 분석기</h1>", unsafe_allow_html=True)

# --- 1. 분석 옵션 (본문 구성) ---
st.subheader("1. 시장 및 종목 설정")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    market_choice = st.radio("분석 시장", ['한국 주식 (KRX)', '미국 주식 (NASDAQ)', '코인 (Upbit)'])

with col2:
    period = st.radio("봉 주기", ['일봉', '주봉'])

# 데이터 로드
if '주식' in market_choice:
    m_code = 'KRX' if '한국' in market_choice else 'NASDAQ'
    listing = get_stock_listing(m_code)
else:
    listing = get_coin_listing()

with col3:
    if not listing.empty:
        selected_label = st.selectbox("종목 선택", listing['label'].tolist())
        ticker = listing[listing['label'] == selected_label]['Code'].values[0]

# --- 2. 날짜 설정 (본문 구성) ---
st.subheader("2. 분석 기간 설정")
col4, col5 = st.columns(2)
with col4:
    start_date = st.date_input("시작일", datetime.date.today() - datetime.timedelta(days=365))
with col5:
    end_date = st.date_input("종료일", datetime.date.today())

st.divider()

if st.button("🚀 데이터 분석 및 백테스팅 시작", use_container_width=True, type="primary"):
    with st.spinner('데이터를 분석 중입니다...'):
        # 데이터 가져오기
        if '주식' in market_choice:
            df = fdr.DataReader(ticker, start_date, end_date)
        else:
            interval = 'day' if period == '일봉' else 'week'
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=400) # 넉넉히 가져옴

        if df is not None and not df.empty:
            # 팩터 계산 및 패턴 매칭
            df = calculate_advanced_factors(df)
            df = find_candle_patterns(df)
            
            # --- 3. 백테스팅 통계 섹션 ---
            st.subheader("📊 패턴별 5일 후 승률 리포트")
            pattern_cols = ['is_hammer', 'is_bullish_engulfing', 'is_bearish_engulfing', 'is_shooting_star']
            stat_cols = st.columns(len(pattern_cols))
            
            has_any_pattern = False
            for idx, p_col in enumerate(pattern_cols):
                p_data = df[df[p_col] == True].copy()
                if not p_data.empty:
                    has_any_pattern = True
                    # 유효한 수익률 데이터(마지막 5일 제외)만 계산
                    valid_p_data = p_data.dropna(subset=['Next_5d_Return'])
                    if not valid_p_data.empty:
                        win_rate = (valid_p_data['Next_5d_Return'] > 0).mean() * 100
                        avg_ret = valid_p_data['Next_5d_Return'].mean()
                        stat_cols[idx].metric(p_col.replace('is_', '').upper(), f"{len(p_data)}회", f"{win_rate:.1f}% 승률")
                        stat_cols[idx].caption(f"평균 수익: {avg_ret:.2f}%")
                    else:
                        stat_cols[idx].write(f"{p_col}\n데이터 부족")
                else:
                    stat_cols[idx].write(f"{p_col}\n발견 안됨")

            # --- 4. 차트 시각화 ---
            st.subheader("🕯️ 캔들 패턴 시각화 (마커 표시)")
            
            apds = []
            # 오류 수정한 마커 추가 부분 (scatter_limit 제거)
            if df['is_bullish_engulfing'].any():
                bull_data = df['Low'] * 0.97
                bull_data[~df['is_bullish_engulfing']] = np.nan
                apds.append(mpf.make_addplot(bull_data, type='scatter', markersize=120, marker='^', color='green'))
            
            if df['is_bearish_engulfing'].any():
                bear_data = df['High'] * 1.03
                bear_data[~df['is_bearish_engulfing']] = np.nan
                apds.append(mpf.make_addplot(bear_data, type='scatter', markersize=120, marker='v', color='red'))
            
            # 차트 그리기
            if not df.empty:
                fig, axlist = mpf.plot(df, type='candle', volume=True, addplot=apds, 
                                   style='charles', figratio=(16, 9), returnfig=True,
                                   title=f"\n{selected_label} Analysis")
                st.pyplot(fig)
            
            with st.expander("📝 분석 상세 데이터 확인"):
                st.dataframe(df.tail(20))
        else:
            st.error("데이터를 불러오지 못했습니다. 종목 코드나 기간을 확인해주세요.")
