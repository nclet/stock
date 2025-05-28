# pages/6_ripple_advanced_backtest.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# --- Streamlit 설정 및 데이터 다운로드 ---
st.set_page_config(layout="wide")
st.title("리플(XRP) 백테스팅(장·단기 이평선, 모멘텀, RSI)")
st.write("##### RSI와 모멘텀은 좌측의 메뉴아래서 사용이 가능합니다.")
# 기본 날짜 설정
default_end_date = datetime.date.today()
default_start_date_5_years_ago = default_end_date - datetime.timedelta(days=365 * 9) # 9년치 데이터

# ccxt는 2017년 이후 데이터가 많으므로 시작일을 조정하는 것이 좋습니다.
min_valid_date_for_most_exchanges = datetime.date(2017, 7, 1)

st.sidebar.header("데이터 및 전략 설정")
start_date = st.sidebar.date_input("시작 날짜", default_start_date_5_years_ago)
end_date = st.sidebar.date_input("종료 날짜", default_end_date)

if start_date >= end_date:
    st.sidebar.error("종료 날짜는 시작 날짜보다 미래여야 합니다.")
    st.stop()
elif start_date < min_valid_date_for_most_exchanges:
    st.sidebar.warning(f"대부분의 주요 암호화폐 거래소는 {min_valid_date_for_most_exchanges} 이후부터 데이터가 존재합니다. 해당 날짜 이후로 설정하시면 더 많은 데이터를 얻을 수 있습니다.")

@st.cache_data
def load_crypto_data(symbol, timeframe, start_date_obj, end_date_obj):
    exchange = ccxt.binance() # 원하는 거래소를 선택 (예: ccxt.upbit(), ccxt.coinbasepro(), etc.)
    
    start_datetime = datetime.datetime(start_date_obj.year, start_date_obj.month, start_date_obj.day)
    end_datetime = datetime.datetime(end_date_obj.year, end_date_obj.month, end_date_obj.day)

    since_timestamp_ms = exchange.parse8601(start_datetime.isoformat())
    
    ohlcv = []
    while True:
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp_ms, limit=1000)
            if not chunk:
                break
            ohlcv.extend(chunk)
            since_timestamp_ms = chunk[-1][0] + (24 * 60 * 60 * 1000) # 1일 = 86400000ms

            if since_timestamp_ms > end_datetime.timestamp() * 1000:
                break
            
            time.sleep(0.05) # Rate Limit 준수

        except ccxt.NetworkError as e:
            st.warning(f"네트워크 오류: {e}. 잠시 후 다시 시도합니다.")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            st.error(f"거래소 오류: {e}. 데이터 가져오기를 중단합니다. Rate Limit에 도달했을 수 있습니다. 잠시 후 다시 시도해보세요.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"알 수 없는 오류: {e}. 데이터 가져오기를 중단합니다.")
            return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.loc[start_datetime:end_datetime] 
    df['Adj Close'] = df['close'] # 'Adj Close' 컬럼 생성

    return df # OHLCV 전체 데이터를 반환합니다.

st.write(f"##### 데이터 다운로드: XRP/USDT ({start_date} ~ {end_date})")
ohlcv_data = load_crypto_data("XRP/USDT", "1d", start_date, end_date)

# 디버깅을 위한 데이터 로딩 정보 출력
st.write(f"로드된 데이터의 행 개수: {ohlcv_data.shape[0]}개")
# st.write(f"로드된 데이터의 처음 5개 행:\n {ohlcv_data.head()}")
# st.write(f"로드된 데이터의 마지막 5개 행:\n {ohlcv_data.tail()}")


if ohlcv_data.empty:
    st.error("지정된 날짜 범위에 대한 데이터를 다운로드할 수 없습니다. 날짜 범위나 선택된 거래소의 데이터 유무를 확인해주세요.")
    st.stop()


### 지표 계산 함수

def calculate_indicators(df, use_sma, use_momentum, use_rsi,
                         short_ma_period, long_ma_period, rsi_period, momentum_period):

    # 이동평균선
    if use_sma:
        df['SMA_Short'] = df['Adj Close'].rolling(window=short_ma_period).mean()
        df['SMA_Long'] = df['Adj Close'].rolling(window=long_ma_period).mean()
    else:
        df['SMA_Short'] = np.nan # 사용하지 않으면 NaN으로 초기화
        df['SMA_Long'] = np.nan

    # RSI
    if use_rsi:
        delta = df['Adj Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        # 0으로 나누는 오류 방지
        rs = np.where(loss == 0, np.inf, gain / loss) 
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = np.nan

    # 모멘텀 (현재 종가 / N일 전 종가 * 100)
    if use_momentum:
        df['Momentum'] = df['Adj Close'].pct_change(momentum_period) * 100
    else:
        df['Momentum'] = np.nan
    
    return df

st.sidebar.subheader("이동평균선 설정")
use_sma = st.sidebar.checkbox("이동평균선 사용", value=True)
short_ma_period = st.sidebar.slider("단기 이동평균선 기간 (일)", 5, 50, 20) if use_sma else 0
long_ma_period = st.sidebar.slider("장기 이동평균선 기간 (일)", 30, 200, 60) if use_sma else 0

if use_sma and short_ma_period >= long_ma_period:
    st.sidebar.error("단기 이동평균선 기간은 장기 이동평균선 기간보다 작아야 합니다.")
    st.stop()

st.sidebar.subheader("모멘텀 지표 설정")
use_momentum = st.sidebar.checkbox("모멘텀 사용", value=False)
momentum_period = st.sidebar.slider("모멘텀 기간 (일)", 5, 30, 14) if use_momentum else 0
momentum_buy_threshold = st.sidebar.slider("모멘텀 매수 임계값 (%)", -10.0, 10.0, 0.5, step=0.1) if use_momentum else 0
momentum_sell_threshold = st.sidebar.slider("모멘텀 매도 임계값 (%)", -10.0, 10.0, -0.5, step=0.1) if use_momentum else 0

st.sidebar.subheader("RSI 지표 설정")
use_rsi = st.sidebar.checkbox("RSI 사용", value=False)
rsi_period = st.sidebar.slider("RSI 기간 (일)", 5, 30, 14) if use_rsi else 0
rsi_buy_threshold = st.sidebar.slider("RSI 매수 임계값", 20, 40, 30) if use_rsi else 0 # 과매도
rsi_sell_threshold = st.sidebar.slider("RSI 매도 임계값", 60, 80, 70) if use_rsi else 0 # 과매수

# 지표 계산
processed_data = calculate_indicators(ohlcv_data.copy(), 
                                      use_sma, use_momentum, use_rsi,
                                      short_ma_period, long_ma_period, rsi_period, momentum_period)

# 이 부분은 calculate_indicators()에서 계산된 지표가 없는 경우를 걸러냅니다.
# 하지만 dropna()가 backtest_strategy 내부로 옮겨졌기 때문에, 이 체크는
# 지표 계산에 필요한 데이터가 최소한 있는지를 보는 것이 됩니다.
# 실제로 비어있다면, ohlcv_data.empty에서 걸러졌을 것입니다.
# 따라서 이 부분의 조건은 `ohlcv_data.empty`와 거의 동일하게 작동합니다.
if processed_data.empty: # 또는 processed_data['Adj Close'].empty:
    st.error("지표 계산에 필요한 데이터를 처리할 수 없습니다. 데이터 기간을 확인해주세요.")
    st.stop()
    
    
def backtest_strategy(df, use_sma, use_momentum, use_rsi,
                      short_ma_period, long_ma_period,
                      momentum_buy_threshold, momentum_sell_threshold,
                      rsi_buy_threshold, rsi_sell_threshold):
    
    # 전략에 필요한 컬럼 목록을 동적으로 생성
    cols_to_check = ['Adj Close']
    if use_sma:
        cols_to_check.extend(['SMA_Short', 'SMA_Long'])
    if use_momentum:
        cols_to_check.append('Momentum')
    if use_rsi:
        cols_to_check.append('RSI')

    # 필요한 컬럼들에서 NaN이 있는 행만 제거
    df.dropna(subset=cols_to_check, inplace=True) # <-- 이 부분 수정!

    if df.empty:
        return pd.DataFrame()

    df['Position'] = 0
    df['Strategy_Return'] = 0.0
    df['Cumulative_Strategy_Return'] = 1.0
    df['Cumulative_Buy_And_Hold_Return'] = 1.0
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    
        # in_position 변수를 for 루프 시작 전에 초기화해야 합니다.
    in_position = False # <-- 이 줄이 여기에 있는지 확인해 주세요!

    # 백테스팅 시작 시점의 누적 수익률 초기화 (dropna 후 첫 인덱스 기준)
    # 데이터프레임이 비어있지 않다면, 첫 번째 행은 항상 존재합니다.
    # 단, df.index[0]가 유효하려면 for 루프가 최소 두 번 이상 돌 수 있는 데이터가 있어야 합니다.
    # for 루프가 (1, len(df)) 이므로 len(df)가 최소 2여야 합니다.
    if len(df) > 0: # 데이터 프레임이 비어있지 않은지 다시 한번 확인
        df.loc[df.index[0], 'Cumulative_Strategy_Return'] = 1.0
        df.loc[df.index[0], 'Cumulative_Buy_And_Hold_Return'] = 1.0
    else: # 이 경우는 df.empty에서 걸러져야 하지만, 혹시 모를 상황 대비
        return pd.DataFrame()
    

    for i in range(1, len(df)):
        current_date = df.index[i]
        # prev_date = df.index[i-1] # 사용되지 않으므로 제거 가능

        # 매수 조건
        buy_condition_sma = False
        if use_sma:
            if df['SMA_Short'].iloc[i-1] < df['SMA_Long'].iloc[i-1] and \
               df['SMA_Short'].iloc[i] >= df['SMA_Long'].iloc[i]:
                buy_condition_sma = True
        else:
            buy_condition_sma = True # SMA를 사용하지 않으면 조건 항상 충족

        buy_condition_momentum = False
        if use_momentum:
            if df['Momentum'].iloc[i] > momentum_buy_threshold:
                buy_condition_momentum = True
        else:
            buy_condition_momentum = True

        buy_condition_rsi = False
        if use_rsi:
            if df['RSI'].iloc[i] < rsi_buy_threshold: # RSI가 낮으면 과매도 -> 매수 신호
                buy_condition_rsi = True
        else:
            buy_condition_rsi = True
        
        # 모든 활성화된 지표의 매수 조건이 동시에 충족될 때 매수
        if not in_position and \
           buy_condition_sma and \
           buy_condition_momentum and \
           buy_condition_rsi:
            df.loc[current_date, 'Position'] = 1
            df.loc[current_date, 'Buy_Signal'] = True
            in_position = True
            
        # 매도 조건
        sell_condition_sma = False
        if use_sma:
            if df['SMA_Short'].iloc[i-1] > df['SMA_Long'].iloc[i-1] and \
               df['SMA_Short'].iloc[i] <= df['SMA_Long'].iloc[i]:
                sell_condition_sma = True
        else:
            sell_condition_sma = True

        sell_condition_momentum = False
        if use_momentum:
            if df['Momentum'].iloc[i] < momentum_sell_threshold:
                sell_condition_momentum = True
        else:
            sell_condition_momentum = True
            
        sell_condition_rsi = False
        if use_rsi:
            if df['RSI'].iloc[i] > rsi_sell_threshold: # RSI가 높으면 과매수 -> 매도 신호
                sell_condition_rsi = True
        else:
            sell_condition_rsi = True
        
        # 모든 활성화된 지표의 매도 조건이 동시에 충족될 때 매도
        if in_position and \
           sell_condition_sma and \
           sell_condition_momentum and \
           sell_condition_rsi:
            df.loc[current_date, 'Position'] = 0
            df.loc[current_date, 'Sell_Signal'] = True
            in_position = False

        # 수익률 계산
        daily_return = (df['Adj Close'].iloc[i] / df['Adj Close'].iloc[i-1]) - 1

        if in_position:
            df.loc[current_date, 'Strategy_Return'] = daily_return
        else:
            df.loc[current_date, 'Strategy_Return'] = 0.0

        # 누적 수익률 계산
        # prev_date 대신 이전 행의 누적 수익률을 참조
        df.loc[current_date, 'Cumulative_Strategy_Return'] = \
            df['Cumulative_Strategy_Return'].iloc[i-1] * (1 + df.loc[current_date, 'Strategy_Return'])
        df.loc[current_date, 'Cumulative_Buy_And_Hold_Return'] = \
            df['Cumulative_Buy_And_Hold_Return'].iloc[i-1] * (1 + daily_return)
            
    return df

st.write("### 백테스팅 결과")
results = backtest_strategy(processed_data.copy(), use_sma, use_momentum, use_rsi,
                            short_ma_period, long_ma_period,
                            momentum_buy_threshold, momentum_sell_threshold,
                            rsi_buy_threshold, rsi_sell_threshold)

if results.empty:
    st.error("백테스팅 결과를 생성할 수 없습니다. 데이터 기간 및 지표 기간을 다시 확인해주세요.")
    st.stop()
    
# --- 결과 시각화 ---
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1]) # 3:1:1 비율로 세 개의 서브플롯

ax1 = fig.add_subplot(gs[0]) # 가격 및 이동평균선, 신호
ax2 = fig.add_subplot(gs[1], sharex=ax1) # 누적 수익률
ax3 = fig.add_subplot(gs[2], sharex=ax1) # 지표 그래프 (RSI, 모멘텀)

# 상단 그래프 (가격, MA, 신호)
ax1.plot(results.index, results['Adj Close'], label='ripple price', color='lightgray', linewidth=1)
if use_sma:
    ax1.plot(results.index, results['SMA_Short'], label=f'short term MA ({short_ma_period}day)', color='orange', linewidth=1.5)
    ax1.plot(results.index, results['SMA_Long'], label=f'long term MA ({long_ma_period}day)', color='purple', linewidth=1.5)

buy_signals = results[results['Buy_Signal'] == True]
ax1.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', s=100, label='buy signal', zorder=5)

sell_signals = results[results['Sell_Signal'] == True]
ax1.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', s=100, label='sell signal', zorder=5)

ax1.set_ylabel("($USDT)")
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_title("Ripple Price, Moving Average, and Trading Signals")


# 중간 그래프 (누적 수익률)
ax2.plot(results.index, results['Cumulative_Strategy_Return'], label='Strategic Accumulated Return', color='blue', linewidth=2)
ax2.plot(results.index, results['Cumulative_Buy_And_Hold_Return'], label='cumulative return on holdings after purchase', color='green', linestyle='--', linewidth=2)
ax2.set_ylabel("cumulative return")
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_title("Comparison of cumulative returns")


# 하단 그래프 (지표)
if use_rsi:
    ax3.plot(results.index, results['RSI'], label='RSI', color='cyan', linewidth=1)
    ax3.axhline(y=rsi_buy_threshold, color='green', linestyle='--', label=f'RSI buy ({rsi_buy_threshold})')
    ax3.axhline(y=rsi_sell_threshold, color='red', linestyle='--', label=f'RSI sell ({rsi_sell_threshold})')
if use_momentum:
    ax3.plot(results.index, results['Momentum'], label='Momentum', color='magenta', linewidth=1)
    ax3.axhline(y=momentum_buy_threshold, color='green', linestyle=':', label=f'Momentum Buy ({momentum_buy_threshold})')
    ax3.axhline(y=momentum_sell_threshold, color='red', linestyle=':', label=f'Momentum Sell ({momentum_sell_threshold})')


ax3.set_xlabel("data")
ax3.set_ylabel("Indicator value")
ax3.legend(loc='upper left')
ax3.grid(True)
ax3.set_title("Technical indicators")

fig.autofmt_xdate()
st.pyplot(fig)

# 최종 수익률 요약
final_strategy_return = (results['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
final_buy_and_hold_return = (results['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) * 100

st.metric(label="최종 전략 누적 수익률", value=f"{final_strategy_return:.2f}%")
st.metric(label="최종 매수 후 보유 (Buy & Hold) 누적 수익률", value=f"{final_buy_and_hold_return:.2f}%")

st.write("---")
st.write("### 백테스팅 상세 데이터 (일부)")
st.dataframe(results[['Adj Close', 'SMA_Short', 'SMA_Long', 'RSI', 'Momentum', 'Buy_Signal', 'Sell_Signal', 'Position', 'Strategy_Return', 'Cumulative_Strategy_Return', 'Cumulative_Buy_And_Hold_Return']].tail(20))

st.write("---")
st.write("### 참고")
st.write("""
- 이 백테스팅 결과는 과거 데이터를 기반으로 하며, 미래 수익을 보장하지 않습니다.
- 실제 거래에서는 거래 수수료, 슬리피지, 유동성 등 다양한 요인이 고려되어야 합니다.
- CCXT를 통한 데이터 수집 시 API Rate Limit 등으로 인해 모든 데이터를 가져오지 못할 수 있습니다.
- **각 지표의 임계값은 예시이며, 최적의 조합과 값은 끊임없는 연구와 백테스팅을 통해 찾아야 합니다.**
""")
