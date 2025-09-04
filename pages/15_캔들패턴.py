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
@st.cache_data
def get_stock_listing():
    """
    FinanceDataReader에서 한국 주식 종목 전체 목록을 가져오고 캐싱합니다.
    사용자가 종목을 쉽게 선택할 수 있도록 'Name (Code)' 형태의 'label'을 생성합니다.
    """
    try:
        # KRX (한국 거래소) 종목 전체 목록을 가져옵니다.
        df_krx = fdr.StockListing('KRX')
        # FinanceDataReader의 버전이나 환경에 따라 'Symbol' 또는 'Code'로 열 이름이
        # 반환되지만, 최신 버전에서는 'Code'를 사용하도록 수정되었습니다.
        # 따라서 'Code' 열을 직접 사용합니다.
        if 'Code' not in df_krx.columns:
            st.error("데이터에 'Code' 열이 없습니다. 라이브러리 버전을 확인해주세요.")
            return pd.DataFrame()
        
        # 'Code' 열을 문자열로 변환합니다.
        df_krx['Code'] = df_krx['Code'].astype(str)
        # 사용자 편의를 위해 '종목명 (코드)' 형태의 레이블을 만듭니다.
        df_krx['label'] = df_krx['Name'] + ' (' + df_krx['Code'] + ')'
        return df_krx
    except Exception as e:
        st.error(f"종목 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame() # 빈 데이터프레임을 반환하여 이후 오류 방지

def get_korean_stock_data(ticker, start_date, end_date):
    """
    주어진 종목 코드와 날짜 범위에 대한 한국 주식 데이터를 가져옵니다.
    :param ticker: 주식 종목 코드
    :param start_date: 데이터 시작 날짜 (YYYY-MM-DD)
    :param end_date: 데이터 종료 날짜 (YYYY-MM-DD)
    :return: 주식 데이터가 담긴 Pandas DataFrame, 오류 발생 시 None 반환
    """
    try:
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
        
        if lower_shadow > 2 * body_length and upper_shadow < body_length:
            if close_p > open_p:
                df.loc[df.index[i], 'is_hammer'] = True
            elif close_p < open_p:
                df.loc[df.index[i], 'is_hanging_man'] = True
        
        if upper_shadow > 2 * body_length and lower_shadow < body_length:
            if close_p > open_p:
                df.loc[df.index[i], 'is_inverted_hammer'] = True
            elif close_p < open_p:
                df.loc[df.index[i], 'is_shooting_star'] = True
                
        if body_length < (high_p - low_p) * 0.05:
            df.loc[df.index[i], 'is_doji'] = True

        # --- 이중 캔들 패턴 ---
        if i >= 1:
            prev_open, prev_close, prev_high, prev_low = df.iloc[i-1][['Open', 'Close', 'High', 'Low']]
            prev_body_midpoint = (prev_open + prev_close) / 2
            
            if (prev_close < prev_open and close_p > open_p and open_p < prev_close and close_p > prev_open):
                df.loc[df.index[i], 'is_bullish_engulfing'] = True
            if (prev_close > prev_open and close_p < open_p and open_p > prev_close and close_p < prev_open):
                df.loc[df.index[i], 'is_bearish_engulfing'] = True
            if (prev_close < prev_open and close_p > open_p and open_p < prev_low and close_p > prev_body_midpoint and close_p < prev_open):
                df.loc[df.index[i], 'is_piercing_line'] = True
            if (prev_close > prev_open and close_p < open_p and open_p > prev_high and close_p < prev_body_midpoint and close_p > prev_open):
                df.loc[df.index[i], 'is_dark_cloud_cover'] = True

        # --- 삼중 캔들 패턴 ---
        if i >= 2:
            prev2_open, prev2_close = df.iloc[i-2][['Open', 'Close']]
            prev1_open, prev1_close = df.iloc[i-1][['Open', 'Close']]
            curr_open, curr_close = df.iloc[i][['Open', 'Close']]
            
            if (prev2_close > prev2_open and prev1_close > prev1_open and curr_close > curr_open and
                prev1_close > prev2_close and curr_close > prev1_close and
                prev1_open >= prev2_close and curr_open >= prev1_close):
                df.loc[df.index[i], 'is_three_white_soldiers'] = True
                
            if (prev2_close < prev2_open and prev1_close < prev1_open and curr_close < curr_open and
                prev1_close < prev2_close and curr_close < prev1_close and
                prev1_open <= prev2_close and curr_open <= prev1_close):
                df.loc[df.index[i], 'is_three_black_crows'] = True
    
    return df

# ---------------------------------------------------------------------------------
# 2. Streamlit 웹 인터페이스 구성
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="주식 캔들 패턴 분석기", layout="wide")

st.markdown("<h1 style='text-align: center;'>주식 캔들 패턴 분석기</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>원하는 종목과 날짜 범위를 선택하여 차트를 분석하세요.</h3>", unsafe_allow_html=True)

# 종목 리스트 가져오기
df_company = get_stock_listing()

if not df_company.empty:
    # `st.selectbox`를 사용하여 사용자에게 종목을 선택하도록 합니다.
    selected_label = st.selectbox("📊 분석할 종목을 선택하세요", df_company["label"].tolist())
    # 선택된 레이블에서 종목 코드를 추출합니다.
    selected_code = df_company[df_company["label"] == selected_label]["Code"].values[0]

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
        st.subheader("분석 중...")
        st.info("데이터를 불러오고 캔들 패턴을 분석하는 중입니다. 잠시만 기다려 주세요.")

        df = get_korean_stock_data(selected_code, start_date, end_date)

        if df is not None:
            df_with_patterns = find_candle_patterns(df.copy())
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

            if total_patterns > 0:
                for label, count in pattern_results.items():
                    st.write(f"- **{label}**: {count}개 발견")
            else:
                st.write("선택한 기간 동안 발견된 캔들 패턴이 없습니다.")
                
            st.subheader("4. 캔들 차트")
            mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridcolor='gray')
            
            fig, axlist = mpf.plot(
                df_with_patterns,
                type='candle',
                style=s,
                title=f'{selected_label} 일봉 차트',
                ylabel='주가',
                volume=True,
                figratio=(15, 8),
                addplot=apds,
                returnfig=True
            )
            st.pyplot(fig)
