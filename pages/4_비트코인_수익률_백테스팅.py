import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import traceback # 오류 상세 내용을 출력하기 위해 임포트

# --- Streamlit 설정 및 데이터 다운로드 ---
st.set_page_config(layout="wide")
st.title("📈 비트코인(BTC) 백테스팅 (장·단기 이평선, 모멘텀, RSI)")
st.write("##### RSI와 모멘텀 지표 설정은 좌측 사이드바 메뉴에서 가능합니다.") 

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
elif start_date < min_valid_date_for_upbit_btc: # 최소 유효 날짜 경고
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
        'enableRateLimit': True, # 초당 요청 제한 준수
        'options': {
            'createMarketAutomatically': False, # 이 부분을 추가해주세요!
        },
    })
    exchange.options['cachedCurrencies'] = None
    # 명시적으로 시장 정보 로드 (추가)
    try:
        st.info("🔄 업비트 시장 정보를 로드하는 중...")
        exchange.load_markets()
        st.info(f"🔄 업비트에서 **{symbol}** ({timeframe}) 데이터를 수집하고 있습니다...")
    except Exception as e:
        st.error(f"❌ 업비트 시장 정보를 로드하는 중 오류가 발생했습니다: {e}")
        st.info("API 키를 확인하거나 잠시 후 다시 시도해보세요. Traceback: " + traceback.format_exc())
        return pd.DataFrame()

    # 시작 및 종료 날짜를 타임스탬프 (밀리초)로 변환
    # UTC 기준 00:00:00 (시작일) 및 23:59:59 (종료일)
    start_timestamp_ms = exchange.parse8601(start_date_obj.isoformat() + 'T00:00:00Z')
    end_timestamp_ms = exchange.parse8601(end_date_obj.isoformat() + 'T23:59:59Z') # 종료일 포함

    ohlcv = []
    current_timestamp_ms = start_timestamp_ms # 현재 요청할 데이터의 시작 타임스탬프

    # 1일의 밀리초
    one_day_in_ms = 24 * 60 * 60 * 1000 

    # 진행률 표시를 위한 위젯 추가
    progress_bar = st.progress(0)
    status_text = st.empty()

    # fetch_ohlcv의 limit은 거래소마다 다름. 업비트는 최대 200개 (1분봉, 일봉 등).
    # 따라서 limit=200으로 설정하고, 더 많은 데이터를 가져오기 위해 반복 요청.
    limit_per_call = 200 

    while current_timestamp_ms <= end_timestamp_ms:
        try:
            display_date = datetime.datetime.fromtimestamp(current_timestamp_ms / 1000).strftime('%Y-%m-%d')
            status_text.text(f"데이터 수집 중: {display_date} 부터...")
            
            # 업비트의 fetch_ohlcv는 market 필터링을 지원하지 않을 수 있으므로, 
            # CCXT가 내부적으로 처리할 수 있도록 심볼을 정확히 전달
            chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp_ms, limit=limit_per_call)
            
            if not chunk: # 더 이상 데이터가 없으면 중단
                status_text.text("데이터 수집 완료 (추가 데이터 없음).")
                break 

            ohlcv.extend(chunk)
            
            # 다음 요청을 위한 since_timestamp_ms 업데이트 (가져온 마지막 데이터의 다음 봉)
            # 마지막 봉의 타임스탬프 + 1일 (일봉 기준)
            current_timestamp_ms = chunk[-1][0] + one_day_in_ms 

            # 진행률 업데이트 (대략적인 계산)
            progress_percentage = (current_timestamp_ms - start_timestamp_ms) / (end_timestamp_ms - start_timestamp_ms + one_day_in_ms)
            progress_bar.progress(min(1.0, progress_percentage)) 
            
            # 업비트는 Binance보다 Rate Limit이 더 엄격할 수 있습니다.
            # CCXT의 enableRateLimit: True가 어느 정도 처리해주지만, 
            # 확실하게 요청 간격을 주기 위해 추가적인 sleep을 줍니다.
            # 업비트 public API는 초당 10회, private API는 초당 10회 (체결) ~ 30회 (주문) 등으로 다릅니다.
            # ohlcv는 public API이므로 초당 10회 이내로 맞추는 것이 좋습니다. (0.1초 이상 대기)
            time.sleep(0.15) # 0.15초 대기 (안전하게 0.1초보다 크게)

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
st.write(f"##### 데이터 다운로드: **BTC/KRW** ({start_date} ~ {end_date})") # 심볼 표기 변경
ohlcv_data = load_crypto_data("BTC/KRW", "1d", start_date, end_date) # 심볼 변경

# 디버깅 정보는 실제 배포 시에는 삭제하거나 주석 처리하는 것이 좋습니다.
# st.write(f"로드된 데이터의 행 개수: {ohlcv_data.shape[0]}개")
# st.write(f"로드된 데이터의 처음 5개 행:\n {ohlcv_data.head()}")
# st.write(f"로드된 데이터의 마지막 5개 행:\n {ohlcv_data.tail()}")

if ohlcv_data.empty:
    st.error("지정된 날짜 범위에 대한 데이터를 다운로드할 수 없습니다. 날짜 범위나 선택된 거래소의 데이터 유무를 확인해주세요.") 
    st.stop()


# --- 지표 계산 함수 ---
# (이 부분은 변경 없음)
def calculate_indicators(df, use_sma, use_momentum, use_rsi,
                         short_ma_period, long_ma_period, rsi_period, momentum_period):

    if use_sma:
        df['SMA_Short'] = df['Adj Close'].rolling(window=short_ma_period).mean()
        df['SMA_Long'] = df['Adj Close'].rolling(window=long_ma_period).mean()
    else:
        df['SMA_Short'] = np.nan 
        df['SMA_Long'] = np.nan

    if use_rsi:
        delta = df['Adj Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        # 손실이 0인 경우를 처리하여 NaN 대신 무한대가 되도록 함
        rs = np.where(loss == 0, np.inf, gain / loss) 
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = np.nan

    if use_momentum:
        df['Momentum'] = df['Adj Close'].pct_change(momentum_period) * 100
    else:
        df['Momentum'] = np.nan
    
    return df

st.sidebar.subheader("📊 이동평균선 설정") 
use_sma = st.sidebar.checkbox("이동평균선 사용", value=True)
short_ma_period = st.sidebar.slider("단기 이동평균선 기간 (일)", 5, 50, 20) if use_sma else 0
long_ma_period = st.sidebar.slider("장기 이동평균선 기간 (일)", 30, 200, 60) if use_sma else 0

if use_sma and short_ma_period >= long_ma_period:
    st.sidebar.error("❌ 단기 이동평균선 기간은 장기 이동평균선 기간보다 작아야 합니다.")
    st.sidebar.stop() # st.stop() 대신 st.sidebar.stop() 사용
    # st.stop() # 오류 메시지 후에 앱이 더 이상 진행되지 않도록

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

# 지표 계산
processed_data = calculate_indicators(ohlcv_data.copy(), 
                                      use_sma, use_momentum, use_rsi,
                                      short_ma_period, long_ma_period, rsi_period, momentum_period)

# 지표 계산 후 NaN 값 처리, 특히 RSI와 Momentum은 첫 N개 기간 동안 NaN이 발생
# 이로 인해 백테스팅 시작 시 데이터가 너무 적을 수 있으므로, 초기 NaN을 제거
if processed_data.empty or processed_data.dropna().empty:
    st.error("지표 계산 후 유효한 데이터가 충분하지 않습니다. 시작 날짜를 조정하거나 지표 기간을 짧게 설정해보세요.") 
    st.stop()
    
# --- 백테스팅 함수 ---
def backtest_strategy(df, use_sma, use_momentum, use_rsi,
                      short_ma_period, long_ma_period, 
                      momentum_buy_threshold, momentum_sell_threshold,
                      rsi_buy_threshold, rsi_sell_threshold):
    
    cols_to_check = ['Adj Close']
    if use_sma:
        cols_to_check.extend(['SMA_Short', 'SMA_Long'])
    if use_momentum:
        cols_to_check.append('Momentum')
    if use_rsi:
        cols_to_check.append('RSI')

    df.dropna(subset=cols_to_check, inplace=True) 

    if df.empty:
        st.warning("선택된 지표를 적용한 후 유효한 데이터가 없습니다. 기간 설정을 다시 확인해주세요.") 
        return pd.DataFrame()

    df['Position'] = 0
    df['Strategy_Return'] = 0.0
    df['Cumulative_Strategy_Return'] = 1.0
    df['Cumulative_Buy_And_Hold_Return'] = 1.0
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    
    in_position = False 

    if len(df) > 0: 
        # 첫 날의 누적 수익률을 1.0으로 초기화
        df.loc[df.index[0], 'Cumulative_Strategy_Return'] = 1.0
        df.loc[df.index[0], 'Cumulative_Buy_And_Hold_Return'] = 1.0
    else: 
        return pd.DataFrame()
    
    # 디버깅을 위한 신호 카운터
    buy_signal_count = 0
    sell_signal_count = 0

    st.info("백테스팅 시뮬레이션을 시작합니다...")
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        
        # --- 매수 조건 정의 ---
        sma_buy_condition = False
        if use_sma:
            if df['SMA_Short'].iloc[i-1] < df['SMA_Long'].iloc[i-1] and \
               df['SMA_Short'].iloc[i] >= df['SMA_Long'].iloc[i]:
                sma_buy_condition = True
        else: # SMA를 사용하지 않으면 항상 True
            sma_buy_condition = True

        momentum_buy_condition = False
        if use_momentum:
            if df['Momentum'].iloc[i] > momentum_buy_threshold:
                momentum_buy_condition = True
        else: # 모멘텀을 사용하지 않으면 항상 True
            momentum_buy_condition = True
        
        rsi_buy_condition = False
        if use_rsi:
            if df['RSI'].iloc[i] < rsi_buy_threshold:
                rsi_buy_condition = True
        else: # RSI를 사용하지 않으면 항상 True
            rsi_buy_condition = True
        
        # --- 매수 신호 로직 ---
        # 전략: SMA 조건은 AND로 묶고, 모멘텀과 RSI는 OR로 묶습니다.
        # 즉, (SMA 조건) AND ((모멘텀 조건) OR (RSI 조건))
        # 만약 모멘텀과 RSI를 둘 다 사용하지 않는다면, 이 부분은 자동으로 True가 되어 SMA만으로 작동
        
        combined_indicator_buy_condition = False
        if not use_momentum and not use_rsi: # 모멘텀, RSI 둘 다 사용 안 함 -> SMA만 사용
            combined_indicator_buy_condition = True
        elif use_momentum and not use_rsi: # 모멘텀만 사용
            combined_indicator_buy_condition = momentum_buy_condition
        elif not use_momentum and use_rsi: # RSI만 사용
            combined_indicator_buy_condition = rsi_buy_condition
        elif use_momentum and use_rsi: # 모멘텀, RSI 둘 다 사용 -> OR 조건
            combined_indicator_buy_condition = momentum_buy_condition or rsi_buy_condition


        # 최종 매수 신호
        if not in_position and sma_buy_condition and combined_indicator_buy_condition:
            df.loc[current_date, 'Position'] = 1
            df.loc[current_date, 'Buy_Signal'] = True
            in_position = True
            buy_signal_count += 1
            # st.info(f"매수 신호 발생: {current_date.strftime('%Y-%m-%d')}") # 디버깅용

        # --- 매도 조건 정의 ---
        sma_sell_condition = False
        if use_sma:
            if df['SMA_Short'].iloc[i-1] > df['SMA_Long'].iloc[i-1] and \
               df['SMA_Short'].iloc[i] <= df['SMA_Long'].iloc[i]:
                sma_sell_condition = True
        else: # SMA를 사용하지 않으면 항상 True
            sma_sell_condition = True

        momentum_sell_condition = False
        if use_momentum:
            if df['Momentum'].iloc[i] < momentum_sell_threshold:
                momentum_sell_condition = True
        else: # 모멘텀을 사용하지 않으면 항상 True
            momentum_sell_condition = True
            
        rsi_sell_condition = False
        if use_rsi:
            if df['RSI'].iloc[i] > rsi_sell_threshold:
                rsi_sell_condition = True
        else: # RSI를 사용하지 않으면 항상 True
            rsi_sell_condition = True

        # --- 매도 신호 로직 ---
        # 전략: SMA 조건은 AND로 묶고, 모멘텀과 RSI는 OR로 묶습니다.
        combined_indicator_sell_condition = False
        if not use_momentum and not use_rsi:
            combined_indicator_sell_condition = True
        elif use_momentum and not use_rsi:
            combined_indicator_sell_condition = momentum_sell_condition
        elif not use_momentum and use_rsi:
            combined_indicator_sell_condition = rsi_sell_condition
        elif use_momentum and use_rsi:
            combined_indicator_sell_condition = momentum_sell_condition or rsi_sell_condition

        # 최종 매도 신호
        if in_position and sma_sell_condition and combined_indicator_sell_condition:
            df.loc[current_date, 'Position'] = 0
            df.loc[current_date, 'Sell_Signal'] = True
            in_position = False
            sell_signal_count += 1
            # st.info(f"매도 신호 발생: {current_date.strftime('%Y-%m-%d')}") # 디버깅용


        daily_return = (df['Adj Close'].iloc[i] / df['Adj Close'].iloc[i-1]) - 1

        # 포지션이 있는 경우에만 수익률 계산
        if in_position:
            df.loc[current_date, 'Strategy_Return'] = daily_return
        else:
            df.loc[current_date, 'Strategy_Return'] = 0.0

        # 누적 수익률 계산
        df.loc[current_date, 'Cumulative_Strategy_Return'] = \
            df['Cumulative_Strategy_Return'].iloc[i-1] * (1 + df.loc[current_date, 'Strategy_Return'])
        df.loc[current_date, 'Cumulative_Buy_And_Hold_Return'] = \
            df['Cumulative_Buy_And_Hold_Return'].iloc[i-1] * (1 + daily_return)
            
    st.info(f"백테스팅 완료! 총 매수 신호: {buy_signal_count}회, 총 매도 신호: {sell_signal_count}회.")
    st.info(f"마지막 포지션 상태: {'보유 중' if in_position else '포지션 없음'}")

    return df

st.write("### 📈 백테스팅 결과") 
results = backtest_strategy(processed_data.copy(), use_sma, use_momentum, use_rsi,
                            short_ma_period, long_ma_period,
                            momentum_buy_threshold, momentum_sell_threshold,
                            rsi_buy_threshold, rsi_sell_threshold)

if results.empty:
    st.error("백테스팅 결과를 생성할 수 없습니다. 데이터 기간 및 지표 기간 설정을 다시 확인해주세요.") 
    st.stop()
    
# --- 결과 시각화 ---
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1]) 

ax1 = fig.add_subplot(gs[0]) 
ax2 = fig.add_subplot(gs[1], sharex=ax1) 
ax3 = fig.add_subplot(gs[2], sharex=ax1) 

ax1.plot(results.index, results['Adj Close'], label='비트코인 가격', color='lightgray', linewidth=1) 
if use_sma:
    ax1.plot(results.index, results['SMA_Short'], label=f'단기 이평선 ({short_ma_period}일)', color='orange', linewidth=1.5) 
    ax1.plot(results.index, results['SMA_Long'], label=f'장기 이평선 ({long_ma_period}일)', color='purple', linewidth=1.5) 

buy_signals = results[results['Buy_Signal'] == True]
ax1.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', s=100, label='매수 신호', zorder=5) 

sell_signals = results[results['Sell_Signal'] == True]
ax1.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', s=100, label='매도 신호', zorder=5) 

ax1.set_ylabel("가격(₩KRW)") # 업비트이므로 KRW로 변경
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_title("비트코인 가격, 이동평균선 및 매매 신호") 


ax2.plot(results.index, (results['Cumulative_Strategy_Return'] - 1) * 100, label='전략 누적 수익률 (%)', color='blue', linewidth=2) 
ax2.plot(results.index, (results['Cumulative_Buy_And_Hold_Return'] - 1) * 100, label='매수 후 보유 누적 수익률 (%)', color='green', linestyle='--', linewidth=2) 
ax2.set_ylabel("누적 수익률 (%)") 
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_title("누적 수익률 비교") 


if use_rsi:
    ax3.plot(results.index, results['RSI'], label='RSI', color='cyan', linewidth=1)
    ax3.axhline(y=rsi_buy_threshold, color='green', linestyle='--', label=f'RSI 매수 ({rsi_buy_threshold})') 
    ax3.axhline(y=rsi_sell_threshold, color='red', linestyle='--', label=f'RSI 매도 ({rsi_sell_threshold})') 
if use_momentum:
    ax3.plot(results.index, results['Momentum'], label='모멘텀', color='magenta', linewidth=1) 
    ax3.axhline(y=momentum_buy_threshold, color='green', linestyle=':', label=f'모멘텀 매수 ({momentum_buy_threshold})') 
    ax3.axhline(y=momentum_sell_threshold, color='red', linestyle=':', label=f'모멘텀 매도 ({momentum_sell_threshold})') 


ax3.set_xlabel("날짜") 
ax3.set_ylabel("지표 값")
ax3.legend(loc='upper left')
ax3.grid(True)
ax3.set_title("기술 지표") 

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
st.dataframe(results[['Adj Close', 'SMA_Short', 'SMA_Long', 'RSI', 'Momentum', 'Buy_Signal', 'Sell_Signal', 'Position', 'Strategy_Return', 'Cumulative_Strategy_Return', 'Cumulative_Buy_And_Hold_Return']].tail(20))

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
