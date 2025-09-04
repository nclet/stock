# Streamlit을 사용한 웹 애플리케이션 제작에 필요한 라이브러리
import streamlit as st

# 주식 데이터와 그래프를 다루는 데 필요한 라이브러리들
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import datetime

# ---------------------------------------------------------------------------------
# 1. Streamlit 앱 설정 및 데이터 로드 함수
# ---------------------------------------------------------------------------------
def get_korean_stock_data(ticker, start_date, end_date):
    """
    주어진 종목 코드와 날짜 범위에 대한 한국 주식 데이터를 가져옵니다.
    :param ticker: 주식 종목 코드
    :param start_date: 데이터 시작 날짜 (YYYY-MM-DD)
    :param end_date: 데이터 종료 날짜 (YYYY-MM-DD)
    :return: 주식 데이터가 담긴 Pandas DataFrame, 오류 발생 시 None 반환
    """
    try:
        # FinanceDataReader를 사용하여 주식 데이터를 가져옵니다.
        data = fdr.DataReader(ticker, start_date, end_date)
        if data.empty:
            st.warning(f"오류: [{ticker}] 종목에 대한 데이터를 찾을 수 없습니다. 종목 코드나 날짜 범위를 확인해 주세요.")
            return None
        return data
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return None

def find_candle_patterns(df):
    """
    주어진 주식 데이터 DataFrame에서 캔들 패턴을 찾아 결과를 반환합니다.
    """
    # 각 캔들 패턴을 표시하기 위한 새로운 열을 추가합니다.
    df['is_hammer'] = False
    df['is_inverted_hammer'] = False
    df['is_doji'] = False
    df['is_bullish_engulfing'] = False
    df['is_bearish_engulfing'] = False
    df['is_piercing_line'] = False
    df['is_dark_cloud_cover'] = False
    df['is_three_white_soldiers'] = False
    df['is_three_black_crows'] = False
    df['is_shooting_star'] = False
    df['is_hanging_man'] = False
    
    # 패턴 탐지 로직 (이전 코드와 동일)
    for i in range(len(df)):
        # --- 단일 캔들 패턴 (망치형, 도지, 유성형, 교수형) 찾기 ---
        open_p, close_p, high_p, low_p = df.iloc[i][['Open', 'Close', 'High', 'Low']]
        body_length = abs(close_p - open_p)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p
        
        # 망치형 (Hammer) 및 교수형 (Hanging Man)
        if lower_shadow > 2 * body_length and upper_shadow < body_length:
            if close_p > open_p:
                df.loc[df.index[i], 'is_hammer'] = True
            elif close_p < open_p:
                df.loc[df.index[i], 'is_hanging_man'] = True
        
        # 역망치형 (Inverted Hammer) 및 유성형 (Shooting Star)
        if upper_shadow > 2 * body_length and lower_shadow < body_length:
            if close_p > open_p:
                df.loc[df.index[i], 'is_inverted_hammer'] = True
            elif close_p < open_p:
                df.loc[df.index[i], 'is_shooting_star'] = True
                
        # 도지 (Doji)
        if body_length < (high_p - low_p) * 0.05:
            df.loc[df.index[i], 'is_doji'] = True

        # --- 이중 캔들 패턴 ---
        if i >= 1:
            prev_open, prev_close, prev_high, prev_low = df.iloc[i-1][['Open', 'Close', 'High', 'Low']]
            prev_body_midpoint = (prev_open + prev_close) / 2
            
            # 상승 장악형 (Bullish Engulfing)
            if (prev_close < prev_open and close_p > open_p and open_p < prev_close and close_p > prev_open):
                df.loc[df.index[i], 'is_bullish_engulfing'] = True
            # 하락 장악형 (Bearish Engulfing)
            if (prev_close > prev_open and close_p < open_p and open_p > prev_close and close_p < prev_open):
                df.loc[df.index[i], 'is_bearish_engulfing'] = True
            # 관통형 (Piercing Line)
            if (prev_close < prev_open and close_p > open_p and open_p < prev_low and close_p > prev_body_midpoint and close_p < prev_open):
                df.loc[df.index[i], 'is_piercing_line'] = True
            # 흑운형 (Dark Cloud Cover)
            if (prev_close > prev_open and close_p < open_p and open_p > prev_high and close_p < prev_body_midpoint and close_p > prev_open):
                df.loc[df.index[i], 'is_dark_cloud_cover'] = True

        # --- 삼중 캔들 패턴 ---
        if i >= 2:
            prev2_open, prev2_close = df.iloc[i-2][['Open', 'Close']]
            prev1_open, prev1_close = df.iloc[i-1][['Open', 'Close']]
            curr_open, curr_close = df.iloc[i][['Open', 'Close']]
            
            # 적삼병 (Three White Soldiers)
            if (prev2_close > prev2_open and prev1_close > prev1_open and curr_close > curr_open and
                prev1_close > prev2_close and curr_close > prev1_close and
                prev1_open >= prev2_close and curr_open >= prev1_close):
                df.loc[df.index[i], 'is_three_white_soldiers'] = True
                
            # 흑삼병 (Three Black Crows)
            if (prev2_close < prev2_open and prev1_close < prev1_open and curr_close < curr_open and
                prev1_close < prev2_close and curr_close < prev1_close and
                prev1_open <= prev2_close and curr_open <= prev1_close):
                df.loc[df.index[i], 'is_three_black_crows'] = True
    
    return df

# ---------------------------------------------------------------------------------
# 2. Streamlit 웹 인터페이스 구성
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="주식 캔들 패턴 분석기", layout="wide")

# 앱 제목 및 설명
st.markdown("<h1 style='text-align: center;'>주식 캔들 패턴 분석기</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>원하는 종목과 날짜 범위를 선택하여 차트를 분석하세요.</h3>", unsafe_allow_html=True)

# 종목 검색 및 선택
stock_list = fdr.StockListing('KRX')
# 이 부분을 'Symbol'에서 'Code'로 수정했습니다.
stock_ticker_map = stock_list.set_index('Code')['Name'].to_dict()

# 사용자 입력 위젯
st.subheader("1. 종목 선택")
col1, col2 = st.columns(2)
with col1:
    user_input = st.text_input("종목 코드 입력", placeholder="예: 005930 (삼성전자)")
with col2:
    selected_name = None
    if user_input in stock_ticker_map:
        selected_name = stock_ticker_map[user_input]
    st.write(f"선택된 종목명: **{selected_name if selected_name else '알 수 없음'}**")

st.subheader("2. 날짜 범위 선택")
today = datetime.date.today()
default_start_date = today - datetime.timedelta(days=180)

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("시작 날짜", default_start_date)
with col4:
    end_date = st.date_input("종료 날짜", today)

# 분석 버튼
st.markdown("---")
if st.button("차트 분석 시작", type="primary", use_container_width=True):
    if not user_input:
        st.warning("종목 코드를 입력해 주세요.")
    else:
        st.subheader("분석 중...")
        st.info("데이터를 불러오고 캔들 패턴을 분석하는 중입니다. 잠시만 기다려 주세요.")

        # 데이터 가져오기
        df = get_korean_stock_data(user_input, start_date, end_date)

        if df is not None:
            # 캔들 패턴 찾기
            df_with_patterns = find_candle_patterns(df.copy())
            
            # 차트에 표시할 패턴 마커 및 라인 설정
            apds = []
            
            pattern_types = {
                'is_hammer': ('Low', '^', 'red', 'Hammer (Bullish)'),
                'is_inverted_hammer': ('High', 'v', 'blue', 'Inverted Hammer (Bearish)'),
                'is_doji': ('Close', '*', 'orange', 'Doji'),
                'is_bullish_engulfing': ('Low', 'o', 'green', 'Bullish Engulfing'),
                'is_bearish_engulfing': ('High', 'x', 'purple', 'Bearish Engulfing'),
                'is_piercing_line': ('Low', 'D', 'darkgreen', 'Piercing Line'),
                'is_dark_cloud_cover': ('High', 'D', 'darkred', 'Dark Cloud Cover'),
                'is_shooting_star': ('High', 'v', 'magenta', 'Shooting Star'),
                'is_hanging_man': ('Low', 's', 'brown', 'Hanging Man')
            }

            total_patterns = 0
            st.subheader("3. 발견된 패턴 목록")
            pattern_results = {}
            for col_name, (y_pos, marker, color, label) in pattern_types.items():
                candles = df_with_patterns[df_with_patterns[col_name]]
                if not candles.empty:
                    pattern_data = pd.Series(index=df.index, dtype='float64')
                    for idx in candles.index:
                        pattern_data.loc[idx] = candles.loc[idx, y_pos]
                    
                    apds.append(mpf.make_addplot(pattern_data, 
                                                type='scatter', 
                                                markersize=150, 
                                                marker=marker, 
                                                color=color, 
                                                label=label))
                    count = len(candles)
                    pattern_results[label] = count
                    total_patterns += count

            # 삼중 캔들 패턴 라인 추가
            # 적삼병
            three_white_soldiers_series = pd.Series(index=df.index, dtype='float64')
            for i in range(2, len(df_with_patterns)):
                if df_with_patterns.loc[df_with_patterns.index[i], 'is_three_white_soldiers']:
                    min_low = min(df.iloc[i-2:i+1]['Low'])
                    three_white_soldiers_series.iloc[i-2:i+1] = min_low * 0.99
                    pattern_results['Three White Soldiers'] = pattern_results.get('Three White Soldiers', 0) + 1
                    total_patterns += 1

            if not three_white_soldiers_series.dropna().empty:
                apds.append(mpf.make_addplot(three_white_soldiers_series, 
                                            type='line', 
                                            linestyle='solid', 
                                            width=5, 
                                            color='red', 
                                            label='Three White Soldiers'))
            
            # 흑삼병
            three_black_crows_series = pd.Series(index=df.index, dtype='float64')
            for i in range(2, len(df_with_patterns)):
                if df_with_patterns.loc[df_with_patterns.index[i], 'is_three_black_crows']:
                    max_high = max(df.iloc[i-2:i+1]['High'])
                    three_black_crows_series.iloc[i-2:i+1] = max_high * 1.01
                    pattern_results['Three Black Crows'] = pattern_results.get('Three Black Crows', 0) + 1
                    total_patterns += 1
            
            if not three_black_crows_series.dropna().empty:
                apds.append(mpf.make_addplot(three_black_crows_series, 
                                            type='line', 
                                            linestyle='solid', 
                                            width=5, 
                                            color='blue', 
                                            label='Three Black Crows'))

            # 결과 표시
            if total_patterns > 0:
                for label, count in pattern_results.items():
                    st.write(f"- **{label}**: {count}개 발견")
            else:
                st.write("선택한 기간 동안 발견된 캔들 패턴이 없습니다.")

            # 차트 그리기
            st.subheader("4. 캔들 차트")
            mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridcolor='gray')
            
            fig, axlist = mpf.plot(
                df_with_patterns,
                type='candle',
                style=s,
                title=f'{stock_ticker_map.get(user_input, user_input)} 일봉 차트',
                ylabel='주가',
                volume=True,
                figratio=(15, 8),
                addplot=apds,
                returnfig=True
            )
            st.pyplot(fig)
