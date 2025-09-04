# Streamlit을 사용한 웹 애플리케이션 제작에 필요한 라이브러리
import streamlit as st

# 주식 데이터와 그래프를 다루는 데 필요한 라이브러리들
import FinanceDataReader as fdr
import pyupbit
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import datetime

# ---------------------------------------------------------------------------------
# 1. Streamlit 앱 설정 및 데이터 로드 함수
# ---------------------------------------------------------------------------------
@st.cache_data
def get_stock_listing():
    """FinanceDataReader에서 한국 주식 종목 전체 목록을 가져옵니다."""
    try:
        df_krx = fdr.StockListing('KRX')
        if 'Code' not in df_krx.columns:
            st.error("데이터에 'Code' 열이 없습니다. 라이브러리 버전을 확인해주세요.")
            return pd.DataFrame()
        
        df_krx['Code'] = df_krx['Code'].astype(str)
        df_krx['label'] = df_krx['Name'] + ' (' + df_krx['Code'] + ')'
        return df_krx
    except Exception as e:
        st.error(f"종목 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data
def get_coin_listing():
    """pyupbit에서 원화(KRW) 코인 목록을 가져옵니다."""
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")
        df_coin = pd.DataFrame(tickers, columns=['Code'])
        # 코인명과 티커를 함께 표시하기 위해 'label' 열을 생성합니다.
        df_coin['label'] = df_coin['Code'].str.replace('KRW-', '') + ' (' + df_coin['Code'] + ')'
        return df_coin
    except Exception as e:
        st.error(f"코인 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

def get_stock_data(ticker, start_date, end_date, period='1D'):
    """주식 데이터를 가져오고, 원하는 기간으로 리샘플링합니다."""
    try:
        data = fdr.DataReader(ticker, start_date, end_date)
        if data.empty:
            st.warning(f"오류: [{ticker}] 종목에 대한 데이터를 찾을 수 없습니다. 종목 코드나 날짜 범위를 확인해 주세요.")
            return None
        
        # 주봉 또는 월봉으로 데이터를 리샘플링합니다.
        if period == '1W':
            resampled_data = data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif period == '1M':
            resampled_data = data.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            resampled_data = data
            
        return resampled_data
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return None

def get_coin_data(ticker, start_date, end_date, period='day'):
    """코인 데이터를 가져오고, 원하는 기간으로 리샘플링합니다."""
    try:
        # pyupbit의 get_ohlcv 함수는 count 파라미터가 필수적입니다.
        # 날짜 범위에 맞게 count를 계산합니다.
        days_diff = (end_date - start_date).days
        count = days_diff + 1 if period == 'day' else int(days_diff / 7) + 1 if period == 'week' else int(days_diff / 30) + 1
        
        # pyupbit는 count를 200개로 제한하기 때문에, 200개가 넘어가면 자동으로 200개까지만 가져옵니다.
        # 이 한계를 해결하기 위해 반복문을 사용할 수 있지만, 간단한 예제이므로 `count`를 그대로 사용합니다.
        df = pyupbit.get_ohlcv(ticker=ticker, interval=period, count=count)

        if df is None or df.empty:
            st.warning(f"오류: [{ticker}] 코인에 대한 데이터를 찾을 수 없습니다. 티커나 날짜 범위를 확인해 주세요.")
            return None
            
        # FinanceDataReader의 데이터프레임과 열 이름을 통일합니다.
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'value']
        df.index.name = 'Date'
        
        return df
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return None

def find_candle_patterns(df):
    """
    주어진 주식 데이터 DataFrame에서 캔들 패턴을 찾아 결과를 반환합니다.
    """
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
    
    for i in range(len(df)):
        # 단일 캔들 패턴
        open_p, close_p, high_p, low_p = df.iloc[i][['Open', 'Close', 'High', 'Low']]
        body_length = abs(close_p - open_p)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p
        
        if body_length > 0 and lower_shadow > 2 * body_length and upper_shadow < body_length:
            if close_p > open_p:
                df.loc[df.index[i], 'is_hammer'] = True
            elif close_p < open_p:
                df.loc[df.index[i], 'is_hanging_man'] = True
        
        if body_length > 0 and upper_shadow > 2 * body_length and lower_shadow < body_length:
            if close_p > open_p:
                df.loc[df.index[i], 'is_inverted_hammer'] = True
            elif close_p < open_p:
                df.loc[df.index[i], 'is_shooting_star'] = True
                
        if body_length < (high_p - low_p) * 0.05:
            df.loc[df.index[i], 'is_doji'] = True

        # 이중 캔들 패턴
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

        # 삼중 캔들 패턴
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
st.set_page_config(page_title="주식 & 코인 캔들 패턴 분석기", layout="wide")

st.markdown("<h1 style='text-align: center;'>주식 & 코인 캔들 패턴 분석기</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>원하는 시장과 종목, 날짜 범위를 선택하여 차트를 분석하세요.</h3>", unsafe_allow_html=True)

st.subheader("1. 분석 옵션 선택")
selected_market = st.radio(
    "💰 분석할 시장을 선택하세요",
    ('주식 (KRX)', '코인 (Upbit)'),
    horizontal=True
)

if selected_market == '주식 (KRX)':
    df_listing = get_stock_listing()
    default_start_date = datetime.date.today() - datetime.timedelta(days=365)
    period_options = ('일봉', '주봉', '월봉')
    period_map = {'일봉': '1D', '주봉': '1W', '월봉': '1M'}
else: # 코인 (Upbit)
    df_listing = get_coin_listing()
    default_start_date = datetime.date.today() - datetime.timedelta(days=180) # 코인은 데이터가 많지 않으므로 기본 날짜를 줄였습니다.
    period_options = ('일봉', '주봉', '월봉')
    period_map = {'일봉': 'day', '주봉': 'week', '월봉': 'month'}
    
if not df_listing.empty:
    selected_label = st.selectbox(f"📊 분석할 {selected_market.split()[0]} 종목", df_listing["label"].tolist())
    selected_code = df_listing[df_listing["label"] == selected_label]["Code"].values[0]

    col1, col2 = st.columns(2)
    with col1:
        selected_period = st.radio(
            "⏳ 차트 기간",
            period_options,
            horizontal=True
        )
    with col2:
        all_pattern_options = {
            'is_hammer': '망치형 (상승)',
            'is_inverted_hammer': '역망치형 (하락)',
            'is_doji': '도지형',
            'is_bullish_engulfing': '상승장악형',
            'is_bearish_engulfing': '하락장악형',
            'is_piercing_line': '관통형',
            'is_dark_cloud_cover': '흑운형',
            'is_three_white_soldiers': '적삼병',
            'is_three_black_crows': '흑삼병',
            'is_shooting_star': '유성형',
            'is_hanging_man': '교수형'
        }
        selected_patterns = st.multiselect(
            "📈 표시할 캔들 패턴",
            list(all_pattern_options.values())
        )

    st.subheader("2. 날짜 범위 선택")
    today = datetime.date.today()
    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("시작 날짜", default_start_date)
    with col4:
        end_date = st.date_input("종료 날짜", today)

    st.markdown("---")
    if st.button("차트 분석 시작", type="primary", use_container_width=True):
        st.subheader("분석 중...")
        st.info("데이터를 불러오고 캔들 패턴을 분석하는 중입니다. 잠시만 기다려 주세요.")
        
        if selected_market == '주식 (KRX)':
            df = get_stock_data(selected_code, start_date, end_date, period_map[selected_period])
        else:
            df = get_coin_data(selected_code, start_date, end_date, period_map[selected_period])

        if df is not None and not df.empty:
            df_with_patterns = find_candle_patterns(df.copy())
            apds = []
            
            pattern_info = {
                '망치형 (상승)': ('is_hammer', 'Low', '^', 'red', 140),
                '역망치형 (하락)': ('is_inverted_hammer', 'High', 'v', 'blue', 140),
                '도지형': ('is_doji', 'Close', '*', 'orange', 140),
                '상승장악형': ('is_bullish_engulfing', 'Low', 'o', 'green', 140),
                '하락장악형': ('is_bearish_engulfing', 'High', 'x', 'purple', 140),
                '관통형': ('is_piercing_line', 'Low', 'D', 'darkgreen', 140),
                '흑운형': ('is_dark_cloud_cover', 'High', 'D', 'darkred', 140),
                '유성형': ('is_shooting_star', 'High', 'v', 'magenta', 140),
                '교수형': ('is_hanging_man', 'Low', 's', 'brown', 140),
                '적삼병': ('is_three_white_soldiers', 'Low', 'D', 'darkgreen', 140),
                '흑삼병': ('is_three_black_crows', 'High', 'D', 'darkred', 140)
            }
            
            total_patterns = 0
            st.subheader("3. 발견된 패턴 목록")
            pattern_results = {}
            
            for pattern_label in selected_patterns:
                if pattern_label in pattern_info:
                    col_name, y_pos, marker, color, size = pattern_info[pattern_label]
                    
                    if col_name in ['is_three_white_soldiers', 'is_three_black_crows']:
                        if pattern_label == '적삼병':
                            series = pd.Series(index=df.index, dtype='float64')
                            for i in range(2, len(df_with_patterns)):
                                if df_with_patterns.loc[df_with_patterns.index[i], 'is_three_white_soldiers']:
                                    min_low = min(df.iloc[i-2:i+1]['Low'])
                                    series.iloc[i-2:i+1] = min_low * 0.99
                                    pattern_results[pattern_label] = pattern_results.get(pattern_label, 0) + 1
                                    total_patterns += 1
                            if not series.dropna().empty:
                                apds.append(mpf.make_addplot(series, 
                                            type='line', linestyle='solid', width=5, color='red', label='적삼병'))
                        
                        elif pattern_label == '흑삼병':
                            series = pd.Series(index=df.index, dtype='float64')
                            for i in range(2, len(df_with_patterns)):
                                if df_with_patterns.loc[df_with_patterns.index[i], 'is_three_black_crows']:
                                    max_high = max(df.iloc[i-2:i+1]['High'])
                                    series.iloc[i-2:i+1] = max_high * 1.01
                                    pattern_results[pattern_label] = pattern_results.get(pattern_label, 0) + 1
                                    total_patterns += 1
                            if not series.dropna().empty:
                                apds.append(mpf.make_addplot(series, 
                                            type='line', linestyle='solid', width=5, color='blue', label='흑삼병'))
                    else:
                        candles = df_with_patterns[df_with_patterns[col_name]]
                        if not candles.empty:
                            pattern_data = pd.Series(index=df.index, dtype='float64')
                            for idx in candles.index:
                                pattern_data.loc[idx] = candles.loc[idx, y_pos]
                            
                            apds.append(mpf.make_addplot(pattern_data, 
                                                        type='scatter', 
                                                        markersize=size, 
                                                        marker=marker, 
                                                        color=color, 
                                                        label=pattern_label))
                            count = len(candles)
                            pattern_results[pattern_label] = count
                            total_patterns += count

            if total_patterns > 0:
                for label, count in pattern_results.items():
                    st.write(f"- **{label}**: {count}개 발견")
            else:
                st.write("선택한 기간 동안 발견된 캔들 패턴이 없습니다.")
                
            st.subheader("4. 캔들 차트")
            mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridcolor='gray')
            
            title = f'{selected_label} {selected_period} 차트'
            fig, axlist = mpf.plot(
                df_with_patterns,
                type='candle',
                style=s,
                title=title,
                ylabel='가격',
                volume=True,
                figratio=(15, 8),
                addplot=apds,
                returnfig=True
            )
            st.pyplot(fig)
        else:
            st.error("데이터를 가져오는 데 실패했습니다. 종목 코드나 날짜 범위를 다시 확인해 주세요.")
