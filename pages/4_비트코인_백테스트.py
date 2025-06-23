import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import traceback

# --- Streamlit 설정 및 데이터 다운로드 ---
st.set_page_config(layout="wide")
st.title("📈 비트코인(BTC) 백테스팅 (SMA, MACD, RSI, OBV)")
st.write("##### 각 지표 설정은 좌측 사이드바 메뉴에서 가능합니다.")

# 기본 날짜 설정
default_end_date = datetime.date.today()
# 업비트 BTC/KRW는 2017년 9월 25일부터 데이터가 있습니다.
min_valid_date_for_upbit_btc = datetime.date(2017, 9, 25)
# 기본 시작 날짜는 5년 전 또는 업비트 최소 유효 날짜 중 더 늦은 날짜로 설정
default_start_date = max(min_valid_date_for_upbit_btc, default_end_date - datetime.timedelta(days=365 * 5))

st.sidebar.header("데이터 및 전략 설정")
start_date = st.sidebar.date_input("시작 날짜", default_start_date)
end_date = st.sidebar.date_input("종료 날짜", default_end_date)

if start_date >= end_date:
    st.sidebar.error("❌ 종료 날짜는 시작 날짜보다 미래여야 합니다.")
    st.stop()
elif start_date < min_valid_date_for_upbit_btc:
    st.sidebar.warning(f"⚠️ 업비트 BTC/KRW 데이터는 {min_valid_date_for_upbit_btc} 이후부터 존재합니다. 해당 날짜 이후로 설정하시면 더 많은 데이터를 얻을 수 있습니다.")


@st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
def load_crypto_data(symbol, timeframe, start_date_obj, end_date_obj):
    # --- 업비트 API 키 로드 ---
    try:
        upbit_access_key = st.secrets["UPBIT_ACCESS_KEY"]
        upbit_secret_key = st.secrets["UPBIT_SECRET_KEY"]
    except KeyError as e:
        st.error(f"❌ 업비트 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
        st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 UPBIT_ACCESS_KEY와 UPBIT_SECRET_KEY를 설정해야 합니다.")
        return pd.DataFrame()

    # --- CCXT Upbit 초기화 ---
    exchange = ccxt.upbit({
        'apiKey': upbit_access_key,
        'secret': upbit_secret_key,
        'enableRateLimit': True,
        'options': {
            'createMarketAutomatically': False,
        },
    })
    exchange.options['cachedCurrencies'] = None # 캐시된 통화 정보 초기화

    try:
        st.info("🔄 업비트 시장 정보를 로드하는 중...")
        exchange.load_markets()
        st.info(f"🔄 업비트에서 **{symbol}** ({timeframe}) 데이터를 수집하고 있습니다...")
    except Exception as e:
        st.error(f"❌ 업비트 시장 정보를 로드하는 중 오류가 발생했습니다: {e}")
        st.info("API 키를 확인하거나 잠시 후 다시 시도해보세요. Traceback: " + traceback.format_exc())
        return pd.DataFrame()

    start_timestamp_ms = exchange.parse8601(start_date_obj.isoformat() + 'T00:00:00Z')
    end_timestamp_ms = exchange.parse8601(end_date_obj.isoformat() + 'T23:59:59Z') # 종료일 포함

    ohlcv = []
    current_timestamp_ms = start_timestamp_ms
    one_day_in_ms = 24 * 60 * 60 * 1000

    progress_bar = st.progress(0)
    status_text = st.empty()
    limit_per_call = 200 # 업비트 fetch_ohlcv limit

    while current_timestamp_ms <= end_timestamp_ms:
        try:
            display_date = datetime.datetime.fromtimestamp(current_timestamp_ms / 1000).strftime('%Y-%m-%d')
            status_text.text(f"데이터 수집 중: {display_date} 부터...")

            chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp_ms, limit=limit_per_call)

            if not chunk: # 더 이상 데이터가 없으면 중단
                status_text.text("데이터 수집 완료 (추가 데이터 없음).")
                break

            ohlcv.extend(chunk)

            current_timestamp_ms = chunk[-1][0] + one_day_in_ms

            progress_percentage = (current_timestamp_ms - start_timestamp_ms) / (end_timestamp_ms - start_timestamp_ms + one_day_in_ms)
            progress_bar.progress(min(1.0, progress_percentage))

            time.sleep(0.15) # Rate Limit 회피

        except ccxt.NetworkError as e:
            st.warning(f"네트워크 오류: {e}. 잠시 후 다시 시도합니다.")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            st.error(f"거래소 오류: {e}. 데이터 가져오기를 중단합니다. Rate Limit에 도달했을 수 있습니다. 잠시 후 다시 시도해보세요. Traceback: {traceback.format_exc()}")
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()
        except Exception as e:
            st.error(f"알 수 없는 오류: {e}. 데이터 가져오기를 중단합니다. Traceback: {traceback.format_exc()}")
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()

    progress_bar.empty()
    status_text.empty()

    if not ohlcv:
        st.warning("⚠️ 지정된 기간 동안 데이터를 가져오지 못했습니다. 날짜 범위 또는 심볼을 확인하세요.")
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 데이터 필터링: 요청한 날짜 범위에 맞게 정확히 필터링
    df = df.loc[pd.to_datetime(start_date_obj):pd.to_datetime(end_date_obj)].copy()

    df['Adj Close'] = df['close']

    st.success(f"✅ **{symbol}** 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
    return df

# BTC/KRW 사용 (업비트)
st.write(f"##### 데이터 다운로드: **BTC/KRW** ({start_date} ~ {end_date})")
ohlcv_data = load_crypto_data("BTC/KRW", "1d", start_date, end_date)

if ohlcv_data.empty:
    st.error("지정된 날짜 범위에 대한 데이터를 다운로드할 수 없습니다. 날짜 범위나 선택된 거래소의 데이터 유무를 확인해주세요.")
    st.stop()


# --- 지표 계산 함수 ---
def calculate_indicators(df, use_sma, use_momentum, use_rsi, use_macd, use_obv,
                         short_ma_period, long_ma_period, rsi_period, momentum_period,
                         macd_fast_period, macd_slow_period, macd_signal_period):

    # SMA
    if use_sma:
        df['SMA_Short'] = df['Adj Close'].rolling(window=short_ma_period).mean()
        df['SMA_Long'] = df['Adj Close'].rolling(window=long_ma_period).mean()
    else:
        df['SMA_Short'] = np.nan # 사용하지 않으면 NaN으로 채움
        df['SMA_Long'] = np.nan

    # RSI
    if use_rsi:
        delta = df['Adj Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = np.where(loss == 0, np.inf, gain / loss)
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = np.nan

    # Momentum
    if use_momentum:
        df['Momentum'] = df['Adj Close'].pct_change(momentum_period) * 100
    else:
        df['Momentum'] = np.nan

    # MACD
    if use_macd:
        exp1 = df['Adj Close'].ewm(span=macd_fast_period, adjust=False).mean()
        exp2 = df['Adj Close'].ewm(span=macd_slow_period, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=macd_signal_period, adjust=False).mean()
    else:
        df['MACD'] = np.nan
        df['MACD_Signal'] = np.nan

    # OBV (On-Balance Volume)
    if use_obv:
        obv_values = np.zeros(len(df))
        if len(df) > 0:
            obv_values[0] = df['volume'].iloc[0]

        for k in range(1, len(df)):
            if df['Adj Close'].iloc[k] > df['Adj Close'].iloc[k-1]:
                obv_values[k] = obv_values[k-1] + df['volume'].iloc[k]
            elif df['Adj Close'].iloc[k] < df['Adj Close'].iloc[k-1]:
                obv_values[k] = obv_values[k-1] - df['volume'].iloc[k]
            else:
                obv_values[k] = obv_values[k-1]
        df['OBV'] = obv_values
    else:
        df['OBV'] = np.nan

    return df

# --- 사이드바 UI 구성 ---
st.sidebar.subheader("📊 이동평균선 설정")
use_sma = st.sidebar.checkbox("이동평균선 사용", value=True)
short_ma_period = st.sidebar.slider("단기 이동평균선 기간 (일)", 5, 50, 20) if use_sma else 0
long_ma_period = st.sidebar.slider("장기 이동평균선 기간 (일)", 30, 200, 60) if use_sma else 0

if use_sma and short_ma_period >= long_ma_period:
    st.sidebar.error("❌ 단기 이동평균선 기간은 장기 이동평균선 기간보다 작아야 합니다.")
    st.stop()

st.sidebar.subheader("📈 모멘텀 지표 설정")
use_momentum = st.sidebar.checkbox("모멘텀 사용", value=False)
momentum_period = st.sidebar.slider("모멘텀 기간 (일)", 5, 30, 14) if use_momentum else 0
momentum_buy_threshold = st.sidebar.slider("모멘텀 매수 임계값 (%)", -10.0, 10.0, 0.5, step=0.1) if use_momentum else 0
momentum_sell_threshold = st.sidebar.slider("모멘텀 매도 임계값 (%)", -10.0, 10.0, -0.5, step=0.1) if use_momentum else 0

st.sidebar.subheader("📉 RSI 지표 설정")
use_rsi = st.sidebar.checkbox("RSI 사용", value=False)
rsi_period = st.sidebar.slider("RSI 기간 (일)", 5, 30, 14) if use_rsi else 0
rsi_buy_threshold = st.sidebar.slider("RSI 매수 임계값 (과매도)", 20, 40, 30) if use_rsi else 0
rsi_sell_threshold = st.sidebar.slider("RSI 매도 임계값 (과매수)", 60, 80, 70) if use_rsi else 0

st.sidebar.subheader("📊 MACD 지표 설정")
use_macd = st.sidebar.checkbox("MACD 사용", value=False)
macd_fast_period = st.sidebar.slider("MACD 단기 EMA 기간 (일)", 5, 30, 12) if use_macd else 0
macd_slow_period = st.sidebar.slider("MACD 장기 EMA 기간 (일)", 20, 50, 26) if use_macd else 0
macd_signal_period = st.sidebar.slider("MACD 시그널 EMA 기간 (일)", 5, 15, 9) if use_macd else 0

if use_macd and macd_fast_period >= macd_slow_period:
    st.sidebar.error("❌ MACD 단기 EMA 기간은 장기 EMA 기간보다 작아야 합니다.")
    st.stop()

st.sidebar.subheader("📈 OBV 지표 설정")
use_obv = st.sidebar.checkbox("OBV 사용", value=False)


# 지표 계산
processed_data = calculate_indicators(ohlcv_data.copy(),
                                      use_sma, use_momentum, use_rsi, use_macd, use_obv,
                                      short_ma_period, long_ma_period, rsi_period, momentum_period,
                                      macd_fast_period, macd_slow_period, macd_signal_period)

# --- NaN 값 처리 (여기서 중요!) ---
# 실제로 사용되는 지표 컬럼만 선택하여 NaN을 제거합니다.
columns_to_check = ['Adj Close', 'volume'] # 기본적으로 가격, 거래량 데이터는 항상 있어야 함

if use_sma:
    columns_to_check.extend(['SMA_Short', 'SMA_Long'])
if use_rsi:
    columns_to_check.append('RSI')
if use_momentum:
    columns_to_check.append('Momentum')
if use_macd:
    columns_to_check.extend(['MACD', 'MACD_Signal'])
if use_obv:
    columns_to_check.append('OBV')

# 필요한 컬럼에만 NaN이 없는지 확인
initial_rows = len(processed_data)
processed_data.dropna(subset=columns_to_check, inplace=True)
rows_after_dropna = len(processed_data)

st.info(f"지표 계산 후 NaN 제거: 초기 {initial_rows}개 행에서 {rows_after_dropna}개 행으로 감소했습니다 (총 {initial_rows - rows_after_dropna}개 행 제거).")


if processed_data.empty or len(processed_data) < 2:
    st.error("지표 계산 후 유효한 데이터가 너무 적거나 없습니다. 시작 날짜를 조정하거나 지표 기간을 짧게 설정해보세요.")
    st.stop()


# --- 백테스팅 함수 ---
def backtest_strategy(df, use_sma, use_momentum, use_rsi, use_macd, use_obv,
                      short_ma_period, long_ma_period,
                      momentum_buy_threshold, momentum_sell_threshold,
                      rsi_buy_threshold, rsi_sell_threshold):

    if df.empty or len(df) < 2:
        st.warning("백테스팅에 필요한 데이터가 충분하지 않습니다.")
        return pd.DataFrame()

    df['Position'] = 0
    df['Strategy_Return'] = 0.0
    df['Cumulative_Strategy_Return'] = 1.0
    df['Cumulative_Buy_And_Hold_Return'] = 1.0
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False

    in_position = False

    # 첫 날의 누적 수익률 초기화 (dropna로 인해 인덱스가 변경될 수 있으므로, 현재 데이터프레임의 첫 인덱스 사용)
    first_index = df.index[0]
    df.loc[first_index, 'Cumulative_Strategy_Return'] = 1.0
    df.loc[first_index, 'Cumulative_Buy_And_Hold_Return'] = 1.0

    buy_signal_count = 0
    sell_signal_count = 0

    st.info("백테스팅 시뮬레이션을 시작합니다...")

    for i in range(1, len(df)):
        current_date = df.index[i]

        # --- 각 지표별 조건 설정 ---
        # SMA 조건 (추세 필터)
        sma_buy_ok = True # SMA 사용 안 할 경우 기본 True
        sma_sell_ok = True
        if use_sma:
            sma_buy_ok = (df['SMA_Short'].iloc[i-1] < df['SMA_Long'].iloc[i-1] and \
                          df['SMA_Short'].iloc[i] >= df['SMA_Long'].iloc[i]) # 골든 크로스
            sma_sell_ok = (df['SMA_Short'].iloc[i-1] > df['SMA_Long'].iloc[i-1] and \
                           df['SMA_Short'].iloc[i] <= df['SMA_Long'].iloc[i]) # 데드 크로스

        # MACD 조건
        macd_buy_ok = False
        macd_sell_ok = False
        if use_macd:
            if df['MACD'].iloc[i-1] < df['MACD_Signal'].iloc[i-1] and \
               df['MACD'].iloc[i] >= df['MACD_Signal'].iloc[i]:
                macd_buy_ok = True
            if df['MACD'].iloc[i-1] > df['MACD_Signal'].iloc[i-1] and \
               df['MACD'].iloc[i] <= df['MACD_Signal'].iloc[i]:
                macd_sell_ok = True

        # 모멘텀 조건
        momentum_buy_ok = False
        momentum_sell_ok = False
        if use_momentum:
            if df['Momentum'].iloc[i] > momentum_buy_threshold:
                momentum_buy_ok = True
            if df['Momentum'].iloc[i] < momentum_sell_threshold:
                momentum_sell_ok = True

        # RSI 조건
        rsi_buy_ok = False
        rsi_sell_ok = False
        if use_rsi:
            if df['RSI'].iloc[i] < rsi_buy_threshold: # 과매도
                rsi_buy_ok = True
            if df['RSI'].iloc[i] > rsi_sell_threshold: # 과매수
                rsi_sell_ok = True

        # OBV 조건 (OBV는 증가/감소로 신호 판단)
        obv_buy_ok = False
        obv_sell_ok = False
        if use_obv:
            if df['OBV'].iloc[i] > df['OBV'].iloc[i-1]: # OBV 상승
                obv_buy_ok = True
            if df['OBV'].iloc[i] < df['OBV'].iloc[i-1]: # OBV 하락
                obv_sell_ok = True

        # --- 최종 매수 신호 로직 ---
        buy_signal_triggered = False
        if not in_position:
            # 보조 지표 (MACD, 모멘텀, RSI, OBV) 중 하나라도 만족하면 True
            active_secondary_buy_conditions = []
            if use_macd: active_secondary_buy_conditions.append(macd_buy_ok)
            if use_momentum: active_secondary_buy_conditions.append(momentum_buy_ok)
            if use_rsi: active_secondary_buy_conditions.append(rsi_buy_ok)
            if use_obv: active_secondary_buy_conditions.append(obv_buy_ok)

            # 보조 지표 중 하나라도 True이거나, 보조 지표를 전혀 사용하지 않을 경우 True
            # (즉, secondary_indicators_buy_ok가 True가 되려면 active_secondary_buy_conditions가 비어있거나, 그 중 하나라도 True여야 함)
            secondary_indicators_buy_ok = (not active_secondary_buy_conditions) or any(active_secondary_buy_conditions)

            # 최종 매수 조건: SMA가 사용되면 SMA 조건 AND 나머지 지표 OR 조건
            if (not use_sma or sma_buy_ok) and secondary_indicators_buy_ok:
                buy_signal_triggered = True

        # --- 최종 매도 신호 로직 ---
        sell_signal_triggered = False
        if in_position:
            active_secondary_sell_conditions = []
            if use_macd: active_secondary_sell_conditions.append(macd_sell_ok)
            if use_momentum: active_secondary_sell_conditions.append(momentum_sell_ok)
            if use_rsi: active_secondary_sell_conditions.append(rsi_sell_ok)
            if use_obv: active_secondary_sell_conditions.append(obv_sell_ok)

            secondary_indicators_sell_ok = (not active_secondary_sell_conditions) or any(active_secondary_sell_conditions)

            if (not use_sma or sma_sell_ok) and secondary_indicators_sell_ok:
                sell_signal_triggered = True

        # --- 포지션 및 수익률 업데이트 ---
        # 매수/매도 신호는 단 하루에 한 번만 발생하도록 `elif` 사용
        if buy_signal_triggered:
            df.loc[current_date, 'Position'] = 1
            df.loc[current_date, 'Buy_Signal'] = True
            in_position = True
            buy_signal_count += 1
        elif sell_signal_triggered:
            df.loc[current_date, 'Position'] = 0
            df.loc[current_date, 'Sell_Signal'] = True
            in_position = False
            sell_signal_count += 1
        else:
            # 이전 포지션 유지 (변동 없음)
            df.loc[current_date, 'Position'] = df['Position'].iloc[i-1]

        daily_return = (df['Adj Close'].iloc[i] / df['Adj Close'].iloc[i-1]) - 1

        if df.loc[current_date, 'Position'] == 1: # 현재 포지션이 롱(보유)이면 수익률 반영
            df.loc[current_date, 'Strategy_Return'] = daily_return
        else: # 포지션 없으면 수익률 0
            df.loc[current_date, 'Strategy_Return'] = 0.0

        df.loc[current_date, 'Cumulative_Strategy_Return'] = \
            df['Cumulative_Strategy_Return'].iloc[i-1] * (1 + df.loc[current_date, 'Strategy_Return'])
        df.loc[current_date, 'Cumulative_Buy_And_Hold_Return'] = \
            df['Cumulative_Buy_And_Hold_Return'].iloc[i-1] * (1 + daily_return)

    st.info(f"백테스팅 완료! 총 매수 신호: {buy_signal_count}회, 총 매도 신호: {sell_signal_count}회.")
    st.info(f"마지막 포지션 상태: {'보유 중' if in_position else '포지션 없음'}")

    return df

st.write("### 📈 백테스팅 결과")
results = backtest_strategy(processed_data.copy(), use_sma, use_momentum, use_rsi, use_macd, use_obv,
                            short_ma_period, long_ma_period,
                            momentum_buy_threshold, momentum_sell_threshold,
                            rsi_buy_threshold, rsi_sell_threshold)

if results.empty:
    st.error("백테스팅 결과를 생성할 수 없습니다. 데이터 기간 및 지표 기간 설정을 다시 확인해주세요.")
    st.stop()

# --- 결과 시각화 ---
fig = plt.figure(figsize=(14, 14)) # 차트 높이 조정
gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1]) # MACD와 OBV를 위한 공간 추가

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[3], sharex=ax1) # MACD
ax5 = fig.add_subplot(gs[4], sharex=ax1) # OBV


# --- 가격 및 SMA 차트 ---
ax1.plot(results.index, results['Adj Close'], label='비트코인 가격', color='lightgray', linewidth=1)
if use_sma:
    ax1.plot(results.index, results['SMA_Short'], label=f'단기 이평선 ({short_ma_period}일)', color='orange', linewidth=1.5)
    ax1.plot(results.index, results['SMA_Long'], label=f'장기 이평선 ({long_ma_period}일)', color='purple', linewidth=1.5)

buy_signals = results[results['Buy_Signal'] == True]
ax1.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', s=100, label='매수 신호', zorder=5)

sell_signals = results[results['Sell_Signal'] == True]
ax1.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', s=100, label='매도 신호', zorder=5)

ax1.set_ylabel("가격(₩KRW)")
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_title("비트코인 가격, 이동평균선 및 매매 신호")


# --- 누적 수익률 차트 ---
ax2.plot(results.index, (results['Cumulative_Strategy_Return'] - 1) * 100, label='전략 누적 수익률 (%)', color='blue', linewidth=2)
ax2.plot(results.index, (results['Cumulative_Buy_And_Hold_Return'] - 1) * 100, label='매수 후 보유 누적 수익률 (%)', color='green', linestyle='--', linewidth=2)
ax2.set_ylabel("누적 수익률 (%)")
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_title("누적 수익률 비교")


# --- 지표 차트 (RSI, 모멘텀, MACD, OBV) ---
# ax3는 RSI와 모멘텀 중 사용되는 것을 우선적으로 그림
if use_rsi and use_momentum: # 둘 다 사용 시
    ax3.plot(results.index, results['RSI'], label='RSI', color='cyan', linewidth=1)
    ax3.axhline(y=rsi_buy_threshold, color='green', linestyle='--', label=f'RSI 매수 ({rsi_buy_threshold})')
    ax3.axhline(y=rsi_sell_threshold, color='red', linestyle='--', label=f'RSI 매도 ({rsi_sell_threshold})')
    ax3.plot(results.index, results['Momentum'], label='모멘텀', color='magenta', linewidth=1)
    ax3.axhline(y=momentum_buy_threshold, color='lime', linestyle=':', label=f'모멘텀 매수 ({momentum_buy_threshold})')
    ax3.axhline(y=momentum_sell_threshold, color='darkred', linestyle=':', label=f'모멘텀 매도 ({momentum_sell_threshold})')
    ax3.set_title("RSI 및 모멘텀 지표")
    ax3.set_ylabel("값")
    ax3.legend(loc='upper left')
    ax3.grid(True)
elif use_rsi: # RSI만 사용 시
    ax3.plot(results.index, results['RSI'], label='RSI', color='cyan', linewidth=1)
    ax3.axhline(y=rsi_buy_threshold, color='green', linestyle='--', label=f'RSI 매수 ({rsi_buy_threshold})')
    ax3.axhline(y=rsi_sell_threshold, color='red', linestyle='--', label=f'RSI 매도 ({rsi_sell_threshold})')
    ax3.set_title("RSI 지표")
    ax3.set_ylabel("RSI")
    ax3.legend(loc='upper left')
    ax3.grid(True)
elif use_momentum: # 모멘텀만 사용 시
    ax3.plot(results.index, results['Momentum'], label='모멘텀', color='magenta', linewidth=1)
    ax3.axhline(y=momentum_buy_threshold, color='green', linestyle=':', label=f'모멘텀 매수 ({momentum_buy_threshold})')
    ax3.axhline(y=momentum_sell_threshold, color='red', linestyle=':', label=f'모멘텀 매도 ({momentum_sell_threshold})')
    ax3.set_title("모멘텀 지표")
    ax3.set_ylabel("모멘텀 (%)")
    ax3.legend(loc='upper left')
    ax3.grid(True)
else:
    ax3.set_visible(False) # 둘 다 사용 안 하면 차트 공간 숨김


# --- MACD 차트 ---
if use_macd:
    ax4.plot(results.index, results['MACD'], label='MACD Line', color='blue', linewidth=1)
    ax4.plot(results.index, results['MACD_Signal'], label='Signal Line', color='red', linestyle='--', linewidth=1)
    # 히스토그램은 MACD - Signal 값으로 그립니다.
    macd_hist = results['MACD'] - results['MACD_Signal']
    colors = ['green' if x >= 0 else 'red' for x in macd_hist]
    ax4.bar(results.index, macd_hist, label='MACD Histogram', color=colors, alpha=0.5, width=0.8) # bar width 조절
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax4.set_ylabel("MACD")
    ax4.legend(loc='upper left')
    ax4.grid(True)
    ax4.set_title("MACD 지표")
else:
    ax4.set_visible(False)

# --- OBV 차트 ---
if use_obv:
    ax5.plot(results.index, results['OBV'], label='OBV', color='darkgreen', linewidth=1)
    ax5.set_ylabel("OBV")
    ax5.legend(loc='upper left')
    ax5.grid(True)
    ax5.set_title("OBV 지표")
else:
    ax5.set_visible(False)

# x축 레이블 겹침 방지
fig.autofmt_xdate()
st.pyplot(fig)

final_strategy_return = (results['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
final_buy_and_hold_return = (results['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) * 100

st.metric(label="최종 전략 누적 수익률", value=f"{final_strategy_return:.2f}%")
st.metric(label="최종 매수 후 보유 (Buy & Hold) 누적 수익률", value=f"{final_buy_and_hold_return:.2f}%")

st.write("---")
st.write("### 📝 참고")
st.write("""
- **데이터 출처**: 이 앱은 **업비트(Upbit) BTC/KRW 일봉 데이터**를 기반으로 작동합니다.
- **백테스팅 모델의 한계**: 제시된 수익률은 백테스팅 결과이며, 실제 투자 결과와는 다를 수 있습니다. 거래 수수료, 슬리피지(Slippage), 세금, 시스템 지연 등의 실제 거래 환경 요소를 고려하지 않은 단순 시뮬레이션입니다.
- **면책 조항**: 본 정보는 투자 자문이 아니며, 여기에 제시된 내용은 오직 정보 제공을 목적으로 합니다. 투자 결정은 사용자 본인의 판단과 책임 하에 이루어져야 합니다.
""")
st.write("---")
st.write("### 백테스팅 상세 데이터 (최근 20일)")
# 필요한 컬럼만 보여주도록 명시 (use_ 지표에 따라 NaN이 있을 수 있으므로 주의)
display_cols = ['Adj Close', 'Buy_Signal', 'Sell_Signal', 'Position', 'Strategy_Return', 'Cumulative_Strategy_Return', 'Cumulative_Buy_And_Hold_Return']
if use_sma: display_cols.extend(['SMA_Short', 'SMA_Long'])
if use_rsi: display_cols.append('RSI')
if use_momentum: display_cols.append('Momentum')
if use_macd: display_cols.extend(['MACD', 'MACD_Signal'])
if use_obv: display_cols.append('OBV')

st.dataframe(results[display_cols].tail(20))


# import streamlit as st
# import ccxt
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import datetime
# import time
# import traceback # 오류 상세 내용을 출력하기 위해 임포트

# # --- Streamlit 설정 및 데이터 다운로드 ---
# st.set_page_config(layout="wide")
# st.title("📈 비트코인(BTC) 백테스팅 (장·단기 이평선, 모멘텀, RSI)")
# st.write("##### RSI와 모멘텀 지표 설정은 좌측 사이드바 메뉴에서 가능합니다.") 

# # 기본 날짜 설정
# default_end_date = datetime.date.today()
# # 업비트 BTC/KRW는 2017년 9월 25일부터 데이터가 있습니다.
# min_valid_date_for_upbit_btc = datetime.date(2017, 9, 25)
# # 기본 시작 날짜는 5년 전 또는 업비트 최소 유효 날짜 중 더 늦은 날짜로 설정
# default_start_date = max(min_valid_date_for_upbit_btc, default_end_date - datetime.timedelta(days=365 * 5)) 


# st.sidebar.header("데이터 및 전략 설정")
# start_date = st.sidebar.date_input("시작 날짜", default_start_date)
# end_date = st.sidebar.date_input("종료 날짜", default_end_date)

# if start_date >= end_date:
#     st.sidebar.error("❌ 종료 날짜는 시작 날짜보다 미래여야 합니다.")
#     st.stop()
# elif start_date < min_valid_date_for_upbit_btc: # 최소 유효 날짜 경고
#     st.sidebar.warning(f"⚠️ 업비트 BTC/KRW 데이터는 {min_valid_date_for_upbit_btc} 이후부터 존재합니다. 해당 날짜 이후로 설정하시면 더 많은 데이터를 얻을 수 있습니다.")


# @st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
# def load_crypto_data(symbol, timeframe, start_date_obj, end_date_obj):
#     # --- 업비트 API 키 로드 ---
#     try:
#         upbit_access_key = st.secrets["UPBIT_ACCESS_KEY"]
#         upbit_secret_key = st.secrets["UPBIT_SECRET_KEY"]
#     except KeyError as e:
#         st.error(f"❌ 업비트 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
#         st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 UPBIT_ACCESS_KEY와 UPBIT_SECRET_KEY를 설정해야 합니다.")
#         return pd.DataFrame()

#     # --- CCXT Upbit 초기화 ---
#     exchange = ccxt.upbit({
#         'apiKey': upbit_access_key,
#         'secret': upbit_secret_key,
#         'enableRateLimit': True, # 초당 요청 제한 준수
#         'options': {
#             'createMarketAutomatically': False, # 이 부분을 추가해주세요!
#         },
#     })
#     exchange.options['cachedCurrencies'] = None
#     # 명시적으로 시장 정보 로드 (추가)
#     try:
#         st.info("🔄 업비트 시장 정보를 로드하는 중...")
#         exchange.load_markets()
#         st.info(f"🔄 업비트에서 **{symbol}** ({timeframe}) 데이터를 수집하고 있습니다...")
#     except Exception as e:
#         st.error(f"❌ 업비트 시장 정보를 로드하는 중 오류가 발생했습니다: {e}")
#         st.info("API 키를 확인하거나 잠시 후 다시 시도해보세요. Traceback: " + traceback.format_exc())
#         return pd.DataFrame()

#     # # --- CCXT Upbit 초기화 ---
#     # exchange = ccxt.upbit({
#     #     'apiKey': upbit_access_key,
#     #     'secret': upbit_secret_key,
#     #     'enableRateLimit': True, # 초당 요청 제한 준수
#     # })
    
#     # st.info(f"🔄 업비트에서 **{symbol}** ({timeframe}) 데이터를 수집하고 있습니다...")

#     # 시작 및 종료 날짜를 타임스탬프 (밀리초)로 변환
#     # UTC 기준 00:00:00 (시작일) 및 23:59:59 (종료일)
#     start_timestamp_ms = exchange.parse8601(start_date_obj.isoformat() + 'T00:00:00Z')
#     end_timestamp_ms = exchange.parse8601(end_date_obj.isoformat() + 'T23:59:59Z') # 종료일 포함

#     ohlcv = []
#     current_timestamp_ms = start_timestamp_ms # 현재 요청할 데이터의 시작 타임스탬프

#     # 1일의 밀리초
#     one_day_in_ms = 24 * 60 * 60 * 1000 

#     # 진행률 표시를 위한 위젯 추가
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     # fetch_ohlcv의 limit은 거래소마다 다름. 업비트는 최대 200개 (1분봉, 일봉 등).
#     # 따라서 limit=200으로 설정하고, 더 많은 데이터를 가져오기 위해 반복 요청.
#     limit_per_call = 200 

#     while current_timestamp_ms <= end_timestamp_ms:
#         try:
#             display_date = datetime.datetime.fromtimestamp(current_timestamp_ms / 1000).strftime('%Y-%m-%d')
#             status_text.text(f"데이터 수집 중: {display_date} 부터...")
            
#             # 업비트의 fetch_ohlcv는 market 필터링을 지원하지 않을 수 있으므로, 
#             # CCXT가 내부적으로 처리할 수 있도록 심볼을 정확히 전달
#             chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp_ms, limit=limit_per_call)
            
#             if not chunk: # 더 이상 데이터가 없으면 중단
#                 status_text.text("데이터 수집 완료 (추가 데이터 없음).")
#                 break 

#             ohlcv.extend(chunk)
            
#             # 다음 요청을 위한 since_timestamp_ms 업데이트 (가져온 마지막 데이터의 다음 봉)
#             # 마지막 봉의 타임스탬프 + 1일 (일봉 기준)
#             current_timestamp_ms = chunk[-1][0] + one_day_in_ms 

#             # 진행률 업데이트 (대략적인 계산)
#             progress_percentage = (current_timestamp_ms - start_timestamp_ms) / (end_timestamp_ms - start_timestamp_ms + one_day_in_ms)
#             progress_bar.progress(min(1.0, progress_percentage)) 
            
#             # 업비트는 Binance보다 Rate Limit이 더 엄격할 수 있습니다.
#             # CCXT의 enableRateLimit: True가 어느 정도 처리해주지만, 
#             # 확실하게 요청 간격을 주기 위해 추가적인 sleep을 줍니다.
#             # 업비트 public API는 초당 10회, private API는 초당 10회 (체결) ~ 30회 (주문) 등으로 다릅니다.
#             # ohlcv는 public API이므로 초당 10회 이내로 맞추는 것이 좋습니다. (0.1초 이상 대기)
#             time.sleep(0.15) # 0.15초 대기 (안전하게 0.1초보다 크게)

#         except ccxt.NetworkError as e:
#             st.warning(f"네트워크 오류: {e}. 잠시 후 다시 시도합니다.")
#             time.sleep(5) 
#         except ccxt.ExchangeError as e:
#             st.error(f"거래소 오류: {e}. 데이터 가져오기를 중단합니다. Rate Limit에 도달했을 수 있습니다. 잠시 후 다시 시도해보세요. Traceback: {traceback.format_exc()}")
#             progress_bar.empty()
#             status_text.empty()
#             return pd.DataFrame()
#         except Exception as e:
#             st.error(f"알 수 없는 오류: {e}. 데이터 가져오기를 중단합니다. Traceback: {traceback.format_exc()}")
#             progress_bar.empty()
#             status_text.empty()
#             return pd.DataFrame()
            
#     progress_bar.empty() 
#     status_text.empty() 

#     if not ohlcv:
#         st.warning("⚠️ 지정된 기간 동안 데이터를 가져오지 못했습니다. 날짜 범위 또는 심볼을 확인하세요.")
#         return pd.DataFrame()

#     df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     df.set_index('timestamp', inplace=True)
    
#     # 데이터 필터링: 요청한 날짜 범위에 맞게 정확히 필터링
#     df = df.loc[pd.to_datetime(start_date_obj):pd.to_datetime(end_date_obj)].copy()
    
#     df['Adj Close'] = df['close'] 

#     st.success(f"✅ **{symbol}** 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
#     return df

# # BTC/KRW 사용 (업비트)
# st.write(f"##### 데이터 다운로드: **BTC/KRW** ({start_date} ~ {end_date})") # 심볼 표기 변경
# ohlcv_data = load_crypto_data("BTC/KRW", "1d", start_date, end_date) # 심볼 변경

# # 디버깅 정보는 실제 배포 시에는 삭제하거나 주석 처리하는 것이 좋습니다.
# # st.write(f"로드된 데이터의 행 개수: {ohlcv_data.shape[0]}개")
# # st.write(f"로드된 데이터의 처음 5개 행:\n {ohlcv_data.head()}")
# # st.write(f"로드된 데이터의 마지막 5개 행:\n {ohlcv_data.tail()}")

# if ohlcv_data.empty:
#     st.error("지정된 날짜 범위에 대한 데이터를 다운로드할 수 없습니다. 날짜 범위나 선택된 거래소의 데이터 유무를 확인해주세요.")
#     st.stop()


# # --- 지표 계산 함수 ---
# # (이 부분은 변경 없음)
# def calculate_indicators(df, use_sma, use_momentum, use_rsi,
#                          short_ma_period, long_ma_period, rsi_period, momentum_period):

#     if use_sma:
#         df['SMA_Short'] = df['Adj Close'].rolling(window=short_ma_period).mean()
#         df['SMA_Long'] = df['Adj Close'].rolling(window=long_ma_period).mean()
#     else:
#         df['SMA_Short'] = np.nan 
#         df['SMA_Long'] = np.nan

#     if use_rsi:
#         delta = df['Adj Close'].diff(1)
#         gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
#         rs = np.where(loss == 0, np.inf, gain / loss) 
#         df['RSI'] = 100 - (100 / (1 + rs))
#     else:
#         df['RSI'] = np.nan

#     if use_momentum:
#         df['Momentum'] = df['Adj Close'].pct_change(momentum_period) * 100
#     else:
#         df['Momentum'] = np.nan
    
#     return df

# st.sidebar.subheader("📊 이동평균선 설정") 
# use_sma = st.sidebar.checkbox("이동평균선 사용", value=True)
# short_ma_period = st.sidebar.slider("단기 이동평균선 기간 (일)", 5, 50, 20) if use_sma else 0
# long_ma_period = st.sidebar.slider("장기 이동평균선 기간 (일)", 30, 200, 60) if use_sma else 0

# if use_sma and short_ma_period >= long_ma_period:
#     st.sidebar.error("❌ 단기 이동평균선 기간은 장기 이동평균선 기간보다 작아야 합니다.")
#     st.stop()

# st.sidebar.subheader("📈 모멘텀 지표 설정") 
# use_momentum = st.sidebar.checkbox("모멘텀 사용", value=False)
# momentum_period = st.sidebar.slider("모멘텀 기간 (일)", 5, 30, 14) if use_momentum else 0
# momentum_buy_threshold = st.sidebar.slider("모멘텀 매수 임계값 (%)", -10.0, 10.0, 0.5, step=0.1) if use_momentum else 0
# momentum_sell_threshold = st.sidebar.slider("모멘텀 매도 임계값 (%)", -10.0, 10.0, -0.5, step=0.1) if use_momentum else 0

# st.sidebar.subheader("📉 RSI 지표 설정") 
# use_rsi = st.sidebar.checkbox("RSI 사용", value=False)
# rsi_period = st.sidebar.slider("RSI 기간 (일)", 5, 30, 14) if use_rsi else 0
# rsi_buy_threshold = st.sidebar.slider("RSI 매수 임계값 (과매도)", 20, 40, 30) if use_rsi else 0 
# rsi_sell_threshold = st.sidebar.slider("RSI 매도 임계값 (과매수)", 60, 80, 70) if use_rsi else 0 

# # 지표 계산
# processed_data = calculate_indicators(ohlcv_data.copy(), 
#                                       use_sma, use_momentum, use_rsi,
#                                       short_ma_period, long_ma_period, rsi_period, momentum_period)

# if processed_data.empty:
#     st.error("지표 계산에 필요한 데이터를 처리할 수 없습니다. 기간 설정을 다시 확인해주세요.") 
#     st.stop()
    
# # --- 백테스팅 함수 ---
# # (이 부분도 변경 없음)
# def backtest_strategy(df, use_sma, use_momentum, use_rsi,
#                       short_ma_period, long_ma_period, 
#                       momentum_buy_threshold, momentum_sell_threshold,
#                       rsi_buy_threshold, rsi_sell_threshold):
    
#     cols_to_check = ['Adj Close']
#     if use_sma:
#         cols_to_check.extend(['SMA_Short', 'SMA_Long'])
#     if use_momentum:
#         cols_to_check.append('Momentum')
#     if use_rsi:
#         cols_to_check.append('RSI')

#     df.dropna(subset=cols_to_check, inplace=True) 

#     if df.empty:
#         st.warning("선택된 지표를 적용한 후 유효한 데이터가 없습니다. 기간 설정을 다시 확인해주세요.") 
#         return pd.DataFrame()

#     df['Position'] = 0
#     df['Strategy_Return'] = 0.0
#     df['Cumulative_Strategy_Return'] = 1.0
#     df['Cumulative_Buy_And_Hold_Return'] = 1.0
#     df['Buy_Signal'] = False
#     df['Sell_Signal'] = False
    
#     in_position = False 

#     if len(df) > 0: 
#         df.loc[df.index[0], 'Cumulative_Strategy_Return'] = 1.0
#         df.loc[df.index[0], 'Cumulative_Buy_And_Hold_Return'] = 1.0
#     else: 
#         return pd.DataFrame()
    

#     for i in range(1, len(df)):
#         current_date = df.index[i]

#         buy_condition_sma = not use_sma or (df['SMA_Short'].iloc[i-1] < df['SMA_Long'].iloc[i-1] and \
#                                             df['SMA_Short'].iloc[i] >= df['SMA_Long'].iloc[i])

#         buy_condition_momentum = not use_momentum or (df['Momentum'].iloc[i] > momentum_buy_threshold)
        
#         buy_condition_rsi = not use_rsi or (df['RSI'].iloc[i] < rsi_buy_threshold)
        
#         if not in_position and buy_condition_sma and buy_condition_momentum and buy_condition_rsi:
#             df.loc[current_date, 'Position'] = 1
#             df.loc[current_date, 'Buy_Signal'] = True
#             in_position = True
            
#         sell_condition_sma = not use_sma or (df['SMA_Short'].iloc[i-1] > df['SMA_Long'].iloc[i-1] and \
#                                              df['SMA_Short'].iloc[i] <= df['SMA_Long'].iloc[i])

#         sell_condition_momentum = not use_momentum or (df['Momentum'].iloc[i] < momentum_sell_threshold)
            
#         sell_condition_rsi = not use_rsi or (df['RSI'].iloc[i] > rsi_sell_threshold)
        
#         if in_position and sell_condition_sma and sell_condition_momentum and sell_condition_rsi:
#             df.loc[current_date, 'Position'] = 0
#             df.loc[current_date, 'Sell_Signal'] = True
#             in_position = False

#         daily_return = (df['Adj Close'].iloc[i] / df['Adj Close'].iloc[i-1]) - 1

#         if in_position:
#             df.loc[current_date, 'Strategy_Return'] = daily_return
#         else:
#             df.loc[current_date, 'Strategy_Return'] = 0.0

#         df.loc[current_date, 'Cumulative_Strategy_Return'] = \
#             df['Cumulative_Strategy_Return'].iloc[i-1] * (1 + df.loc[current_date, 'Strategy_Return'])
#         df.loc[current_date, 'Cumulative_Buy_And_Hold_Return'] = \
#             df['Cumulative_Buy_And_Hold_Return'].iloc[i-1] * (1 + daily_return)
            
#     return df

# st.write("### 📈 백테스팅 결과") 
# results = backtest_strategy(processed_data.copy(), use_sma, use_momentum, use_rsi,
#                             short_ma_period, long_ma_period,
#                             momentum_buy_threshold, momentum_sell_threshold,
#                             rsi_buy_threshold, rsi_sell_threshold)

# if results.empty:
#     st.error("백테스팅 결과를 생성할 수 없습니다. 데이터 기간 및 지표 기간 설정을 다시 확인해주세요.") 
#     st.stop()
    
# # --- 결과 시각화 ---
# fig = plt.figure(figsize=(14, 10))
# gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1]) 

# ax1 = fig.add_subplot(gs[0]) 
# ax2 = fig.add_subplot(gs[1], sharex=ax1) 
# ax3 = fig.add_subplot(gs[2], sharex=ax1) 

# ax1.plot(results.index, results['Adj Close'], label='Bitcoin Price', color='lightgray', linewidth=1) 
# if use_sma:
#     ax1.plot(results.index, results['SMA_Short'], label=f'Short_Ma ({short_ma_period}day)', color='orange', linewidth=1.5) 
#     ax1.plot(results.index, results['SMA_Long'], label=f'Long_Ma ({long_ma_period}day)', color='purple', linewidth=1.5) 

# buy_signals = results[results['Buy_Signal'] == True]
# ax1.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', s=100, label='buy signal', zorder=5) 

# sell_signals = results[results['Sell_Signal'] == True]
# ax1.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', s=100, label='sell signal', zorder=5) 

# ax1.set_ylabel("price(₩KRW)") # 업비트이므로 KRW로 변경
# ax1.legend(loc='upper left')
# ax1.grid(True)
# ax1.set_title("Bitcoin Price, Moving Average Trading Signal") 


# ax2.plot(results.index, (results['Cumulative_Strategy_Return'] - 1) * 100, label='Strategic Accumulated Return (%)', color='blue', linewidth=2) 
# ax2.plot(results.index, (results['Cumulative_Buy_And_Hold_Return'] - 1) * 100, label='cumulative returns after buying (%)', color='green', linestyle='--', linewidth=2) 
# ax2.set_ylabel("cumulative return (%)") 
# ax2.legend(loc='upper left')
# ax2.grid(True)
# ax2.set_title("Comparison of cumulative returns") 


# if use_rsi:
#     ax3.plot(results.index, results['RSI'], label='RSI', color='cyan', linewidth=1)
#     ax3.axhline(y=rsi_buy_threshold, color='green', linestyle='--', label=f'RSI buy ({rsi_buy_threshold})') 
#     ax3.axhline(y=rsi_sell_threshold, color='red', linestyle='--', label=f'RSI sell ({rsi_sell_threshold})') 
# if use_momentum:
#     ax3.plot(results.index, results['Momentum'], label='Momentum', color='magenta', linewidth=1) 
#     ax3.axhline(y=momentum_buy_threshold, color='green', linestyle=':', label=f'Momentum buying ({momentum_buy_threshold})') 
#     ax3.axhline(y=momentum_sell_threshold, color='red', linestyle=':', label=f'Momentum selling ({momentum_sell_threshold})') 


# ax3.set_xlabel("Date") 
# ax3.set_ylabel("Indicator value")
# ax3.legend(loc='upper left')
# ax3.grid(True)
# ax3.set_title("Technical indicators") 

# fig.autofmt_xdate()
# st.pyplot(fig)

# final_strategy_return = (results['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
# final_buy_and_hold_return = (results['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) * 100

# st.metric(label="최종 전략 누적 수익률", value=f"{final_strategy_return:.2f}%")
# st.metric(label="최종 매수 후 보유 (Buy & Hold) 누적 수익률", value=f"{final_buy_and_hold_return:.2f}%")

# st.write("---")
# st.write("### 📝 참고")
# st.write("""
# - **데이터 출처**: 이 앱은 **업비트(Upbit) BTC/KRW 일봉 데이터**를 기반으로 작동합니다.
# - **백테스팅 모델의 한계**: 제시된 수익률은 백테스팅 결과이며, 실제 투자 결과와는 다를 수 있습니다. 거래 수수료, 슬리피지(Slippage), 세금, 시스템 지연 등의 실제 거래 환경 요소를 고려하지 않은 단순 시뮬레이션입니다.
# - **면책 조항**: 본 정보는 투자 자문이 아니며, 여기에 제시된 내용은 오직 정보 제공을 목적으로 합니다. 투자 결정은 사용자 본인의 판단과 책임 하에 이루어져야 합니다.
# """)
# st.write("---")
# st.write("### 백테스팅 상세 데이터 (최근 20일)") 
# st.dataframe(results[['Adj Close', 'SMA_Short', 'SMA_Long', 'RSI', 'Momentum', 'Buy_Signal', 'Sell_Signal', 'Position', 'Strategy_Return', 'Cumulative_Strategy_Return', 'Cumulative_Buy_And_Hold_Return']].tail(20))
