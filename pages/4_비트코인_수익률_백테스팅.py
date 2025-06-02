# pages/6_bitcoin_advanced_backtest.py
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
# 업비트 KRW-BTC 데이터는 2017년 9월 25일부터 시작됩니다.
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
    st.sidebar.warning(f"⚠️ 업비트 BTC 데이터는 {min_valid_date_for_upbit_btc} 이후부터 존재합니다. 해당 날짜 이후로 설정하시면 더 많은 데이터를 얻을 수 있습니다.")


@st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시하여 API 호출 최소화
def load_crypto_data(symbol, timeframe, start_date_obj, end_date_obj):
    # Streamlit Secrets에서 API 키를 불러옵니다.
    try:
        upbit_access_key = st.secrets["UPBIT_ACCESS_KEY"]
        upbit_secret_key = st.secrets["UPBIT_SECRET_KEY"]
    except KeyError:
        st.error("❌ 업비트 API 키(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)가 Streamlit Secrets에 설정되어 있지 않습니다.")
        st.info("로컬에서 실행하는 경우 `.streamlit/secrets.toml` 파일에 키를 추가하고, Streamlit Cloud에 배포하는 경우 대시보드에서 Secrets를 설정해야 합니다.")
        return pd.DataFrame() # 오류 발생 시 빈 DataFrame 반환하여 실행 중단

    exchange = ccxt.upbit({
        'apiKey': upbit_access_key,
        'secret': upbit_secret_key,
        'enableRateLimit': True, # 초당 요청 제한 준수
    })
    
    st.info(f"🔄 업비트에서 **{symbol}** ({timeframe}) 데이터를 가져오는 중...")

    # 시작 및 종료 날짜를 타임스탬프 (밀리초)로 변환
    # UTC 기준 00:00:00 (시작일) 및 23:59:59 (종료일)
    start_timestamp_ms = exchange.parse8601(start_date_obj.isoformat() + 'T00:00:00Z')
    end_timestamp_ms = exchange.parse8601(end_date_obj.isoformat() + 'T23:59:59Z')
    
    ohlcv = []
    current_timestamp_ms = start_timestamp_ms

    # 업비트의 fetch_ohlcv는 한 번에 최대 200개의 봉을 반환합니다.
    # 따라서 데이터를 잘게 나눠서 가져와야 합니다.
    one_day_in_ms = 24 * 60 * 60 * 1000 # 1일의 밀리초

    # 진행률 표시를 위한 위젯
    progress_bar = st.progress(0)
    status_text = st.empty()

    fetch_count = 0
    # 현재 타임스탬프가 종료 타임스탬프를 넘지 않을 때까지 반복
    while current_timestamp_ms <= end_timestamp_ms:
        try:
            display_date = datetime.datetime.fromtimestamp(current_timestamp_ms / 1000).strftime('%Y-%m-%d')
            status_text.text(f"데이터 수집 중: {display_date} 부터...")
            
            chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp_ms, limit=200)
            
            if not chunk: # 더 이상 데이터가 없으면 중단
                status_text.text("데이터 수집 완료 (추가 데이터 없음).")
                break 

            ohlcv.extend(chunk)
            
            # 다음 요청을 위한 since_timestamp_ms 업데이트 (가져온 마지막 데이터의 다음 날)
            current_timestamp_ms = chunk[-1][0] + one_day_in_ms 

            # 진행률 업데이트 (대략적인 계산)
            progress_percentage = (current_timestamp_ms - start_timestamp_ms) / (end_timestamp_ms - start_timestamp_ms + one_day_in_ms)
            progress_bar.progress(min(1.0, progress_percentage)) # 최대 100%를 넘지 않도록
            
            fetch_count += 1
            if fetch_count % 5 == 0: # 너무 자주 sleep 하지 않도록 (5번 호출당 1회)
                time.sleep(exchange.rateLimit / 1000) # 거래소 rateLimit 준수 (밀리초를 초로 변환)
            
        except ccxt.NetworkError as e:
            st.warning(f"네트워크 오류: {e}. 잠시 후 다시 시도합니다.")
            time.sleep(5) # 5초 대기 후 재시도
        except ccxt.ExchangeError as e:
            st.error(f"거래소 오류: {e}. 데이터 가져오기를 중단합니다. Rate Limit에 도달했을 수 있습니다. 잠시 후 다시 시도해보세요.")
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()
        except Exception as e:
            st.error(f"알 수 없는 오류: {e}. 데이터 가져오기를 중단합니다. Traceback: {traceback.format_exc()}")
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()
            
    progress_bar.empty() # 진행률 바 숨김
    status_text.empty() # 상태 텍스트 숨김

    if not ohlcv:
        st.warning("⚠️ 지정된 기간 동안 데이터를 가져오지 못했습니다. 날짜 범위 또는 심볼을 확인하세요.")
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # 시간대 조정 (UTC -> KST) 및 시간 정보 제거 (날짜만 남김)
    # Upbit은 KST 기준이지만, ccxt는 UTC로 반환할 수 있으므로 변환하는 것이 안전
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul').normalize()

    # 요청한 날짜 범위에 맞게 필터링 (불필요한 미래 데이터 제거)
    # start_date_obj와 end_date_obj는 datetime.date 객체이므로, pd.to_datetime으로 변환
    df = df.loc[pd.to_datetime(start_date_obj):pd.to_datetime(end_date_obj)].copy()
    
    # 분석에 사용할 'Adj Close' 컬럼 생성 (여기서는 'close'와 동일)
    df['Adj Close'] = df['close'] 

    st.success(f"✅ **{symbol}** 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
    return df

# BTC/USDT를 KRW-BTC로 변경 (업비트 원화 마켓)
st.write(f"##### 데이터 다운로드: **KRW-BTC** ({start_date} ~ {end_date})")
ohlcv_data = load_crypto_data("KRW-BTC", "1d", start_date, end_date)


if ohlcv_data.empty:
    st.error("지정된 날짜 범위에 대한 데이터를 다운로드할 수 없습니다. 날짜 범위나 선택된 거래소의 데이터 유무를 확인해주세요.")
    st.stop()


# --- 지표 계산 함수 ---
def calculate_indicators(df, use_sma, use_momentum, use_rsi,
                         short_ma_period, long_ma_period, rsi_period, momentum_period):

    # 이동평균선 (Simple Moving Average)
    if use_sma:
        df['SMA_Short'] = df['Adj Close'].rolling(window=short_ma_period).mean()
        df['SMA_Long'] = df['Adj Close'].rolling(window=long_ma_period).mean()
    else:
        df['SMA_Short'] = np.nan # 사용하지 않으면 NaN으로 초기화
        df['SMA_Long'] = np.nan

    # RSI (Relative Strength Index)
    if use_rsi:
        delta = df['Adj Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        # 0으로 나누는 오류 방지: loss가 0이면 rs를 무한대로 설정 (RSI는 100)
        rs = np.where(loss == 0, np.inf, gain / loss) 
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = np.nan

    # 모멘텀 (현재 종가 / N일 전 종가 * 100 - 100)
    # 퍼센트 변화를 나타내기 위해 (현재 - 과거) / 과거 * 100 또는 pct_change * 100 사용
    if use_momentum:
        df['Momentum'] = df['Adj Close'].pct_change(momentum_period) * 100
    else:
        df['Momentum'] = np.nan
    
    return df

# 사이드바에서 지표 설정
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

# 지표 계산 실행
processed_data = calculate_indicators(ohlcv_data.copy(), 
                                      use_sma, use_momentum, use_rsi,
                                      short_ma_period, long_ma_period, rsi_period, momentum_period)

if processed_data.empty:
    st.error("지표 계산에 필요한 데이터가 충분하지 않습니다. 데이터 기간을 확인해주세요.")
    st.stop()
    
# --- 백테스팅 함수 ---
def backtest_strategy(df, use_sma, use_momentum, use_rsi,
                      short_ma_period, long_ma_period, # 이평선 기간은 전략 로직에 사용되지 않으므로 사실상 불필요하지만 인자 유지
                      momentum_buy_threshold, momentum_sell_threshold,
                      rsi_buy_threshold, rsi_sell_threshold):
    
    # 전략에 필요한 컬럼 목록을 동적으로 생성하여 NaN 제거
    cols_to_check = ['Adj Close']
    if use_sma:
        cols_to_check.extend(['SMA_Short', 'SMA_Long'])
    if use_momentum:
        cols_to_check.append('Momentum')
    if use_rsi:
        cols_to_check.append('RSI')

    df.dropna(subset=cols_to_check, inplace=True) # 필요한 지표 계산 후 NaN이 있는 초기 행 제거

    if df.empty:
        st.warning("선택된 지표를 적용한 후 유효한 데이터가 없습니다. 기간 설정을 다시 확인해주세요.")
        return pd.DataFrame()

    # 결과 저장을 위한 컬럼 초기화
    df['Position'] = 0 # 0: 현금 보유, 1: 포지션 보유
    df['Strategy_Return'] = 0.0
    df['Cumulative_Strategy_Return'] = 1.0 # 초기 자산 1로 시작
    df['Cumulative_Buy_And_Hold_Return'] = 1.0 # 초기 자산 1로 시작
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    
    in_position = False # 현재 포지션 보유 여부

    # 백테스팅 시작 시점의 누적 수익률 초기화
    if len(df) > 0:
        df.loc[df.index[0], 'Cumulative_Strategy_Return'] = 1.0
        df.loc[df.index[0], 'Cumulative_Buy_And_Hold_Return'] = 1.0
    else:
        return pd.DataFrame()
    
    # 날짜별 반복하여 전략 실행
    for i in range(1, len(df)):
        current_date = df.index[i]
        
        # 각 지표별 매수/매도 조건 검토
        buy_condition_sma = not use_sma or (df['SMA_Short'].iloc[i-1] < df['SMA_Long'].iloc[i-1] and \
                                            df['SMA_Short'].iloc[i] >= df['SMA_Long'].iloc[i])

        buy_condition_momentum = not use_momentum or (df['Momentum'].iloc[i] > momentum_buy_threshold)
        
        buy_condition_rsi = not use_rsi or (df['RSI'].iloc[i] < rsi_buy_threshold)

        # 모든 활성화된 지표의 매수 조건이 동시에 충족될 때 매수 (AND 조건)
        if not in_position and buy_condition_sma and buy_condition_momentum and buy_condition_rsi:
            df.loc[current_date, 'Position'] = 1
            df.loc[current_date, 'Buy_Signal'] = True
            in_position = True
            
        sell_condition_sma = not use_sma or (df['SMA_Short'].iloc[i-1] > df['SMA_Long'].iloc[i-1] and \
                                             df['SMA_Short'].iloc[i] <= df['SMA_Long'].iloc[i])

        sell_condition_momentum = not use_momentum or (df['Momentum'].iloc[i] < momentum_sell_threshold)
            
        sell_condition_rsi = not use_rsi or (df['RSI'].iloc[i] > rsi_sell_threshold)
        
        # 모든 활성화된 지표의 매도 조건이 동시에 충족될 때 매도 (AND 조건)
        if in_position and sell_condition_sma and sell_condition_momentum and sell_condition_rsi:
            df.loc[current_date, 'Position'] = 0
            df.loc[current_date, 'Sell_Signal'] = True
            in_position = False

        # 일일 수익률 계산 (어제 대비 오늘 종가 변화율)
        daily_return = (df['Adj Close'].iloc[i] / df['Adj Close'].iloc[i-1]) - 1

        # 전략 수익률은 포지션을 보유했을 때만 발생
        if in_position:
            df.loc[current_date, 'Strategy_Return'] = daily_return
        else:
            df.loc[current_date, 'Strategy_Return'] = 0.0

        # 누적 수익률 계산
        df.loc[current_date, 'Cumulative_Strategy_Return'] = \
            df['Cumulative_Strategy_Return'].iloc[i-1] * (1 + df.loc[current_date, 'Strategy_Return'])
        df.loc[current_date, 'Cumulative_Buy_And_Hold_Return'] = \
            df['Cumulative_Buy_And_Hold_Return'].iloc[i-1] * (1 + daily_return)
            
    return df

st.write("### 📈 백테스팅 결과")
results = backtest_strategy(processed_data.copy(), use_sma, use_momentum, use_rsi,
                            short_ma_period, long_ma_period, # 지표 계산에만 사용
                            momentum_buy_threshold, momentum_sell_threshold,
                            rsi_buy_threshold, rsi_sell_threshold)

if results.empty:
    st.error("백테스팅 결과를 생성할 수 없습니다. 데이터 기간 및 지표 기간 설정을 다시 확인해주세요.")
    st.stop()
    
# --- 결과 시각화 ---
fig = plt.figure(figsize=(14, 10))
# GridSpec을 사용하여 3개의 서브플롯 생성: 가격/MA, 누적 수익률, 지표
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1]) 

ax1 = fig.add_subplot(gs[0]) # 가격 및 이동평균선, 매매 신호
ax2 = fig.add_subplot(gs[1], sharex=ax1) # 누적 수익률 비교
ax3 = fig.add_subplot(gs[2], sharex=ax1) # 기술 지표 (RSI, 모멘텀)

# 상단 그래프 (가격, MA, 신호)
ax1.plot(results.index, results['Adj Close'], label='비트코인 가격', color='lightgray', linewidth=1)
if use_sma:
    ax1.plot(results.index, results['SMA_Short'], label=f'단기 이평선 ({short_ma_period}일)', color='orange', linewidth=1.5)
    ax1.plot(results.index, results['SMA_Long'], label=f'장기 이평선 ({long_ma_period}일)', color='purple', linewidth=1.5)

# 매수/매도 신호는 실제 거래 시점 (봉의 끝)에 발생하는 것으로 가정하고 현재 봉의 종가에 표시
buy_signals = results[results['Buy_Signal'] == True]
ax1.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', s=100, label='매수 신호', zorder=5)

sell_signals = results[results['Sell_Signal'] == True]
ax1.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', s=100, label='매도 신호', zorder=5)

ax1.set_ylabel("가격 (KRW)")
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_title("비트코인 가격, 이동평균선 및 매매 신호")


# 중간 그래프 (누적 수익률)
ax2.plot(results.index, (results['Cumulative_Strategy_Return'] - 1) * 100, label='전략 누적 수익률 (%)', color='blue', linewidth=2)
ax2.plot(results.index, (results['Cumulative_Buy_And_Hold_Return'] - 1) * 100, label='매수 후 보유 누적 수익률 (%)', color='green', linestyle='--', linewidth=2)
ax2.set_ylabel("누적 수익률 (%)")
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_title("누적 수익률 비교")


# 하단 그래프 (지표)
# 각 지표가 활성화되었을 때만 그립니다.
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

# 날짜 라벨 겹침 방지를 위해 자동 포맷팅
fig.autofmt_xdate()
st.pyplot(fig)

# 최종 수익률 요약 메트릭
final_strategy_return = (results['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
final_buy_and_hold_return = (results['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) * 100

st.metric(label="최종 전략 누적 수익률", value=f"{final_strategy_return:.2f}%")
st.metric(label="최종 매수 후 보유 (Buy & Hold) 누적 수익률", value=f"{final_buy_and_hold_return:.2f}%")

st.write("---")
st.write("### 📝 참고")
st.write("""
- **데이터 출처**: 이 앱은 **업비트(Upbit) KRW-BTC 일봉 데이터**를 기반으로 작동합니다.
- **백테스팅 모델의 한계**: 제시된 수익률은 백테스팅 결과이며, 실제 투자 결과와는 다를 수 있습니다. 거래 수수료, 슬리피지(Slippage), 세금, 시스템 지연 등의 실제 거래 환경 요소를 고려하지 않은 단순 시뮬레이션입니다.
- **면책 조항**: 본 정보는 투자 자문이 아니며, 여기에 제시된 내용은 오직 정보 제공을 목적으로 합니다. 투자 결정은 사용자 본인의 판단과 책임 하에 이루어져야 합니다.
""")

# # pages/6_bitcoin_advanced_backtest.py
# import streamlit as st
# import ccxt
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import datetime
# import time

# # --- Streamlit 설정 및 데이터 다운로드 ---
# st.set_page_config(layout="wide")
# st.title("비트코인(BTC) 백테스팅 (장·단기 이평선, 모멘텀, RSI)")
# st.write("##### RSI와 모멘텀은 좌측의 메뉴 아래서 사용이 가능합니다.")

# # 기본 날짜 설정
# default_end_date = datetime.date.today()
# default_start_date_5_years_ago = default_end_date - datetime.timedelta(days=365 * 9) # 9년치 데이터

# # ccxt는 2017년 이후 데이터가 많으므로 시작일을 조정하는 것이 좋습니다.
# min_valid_date_for_most_exchanges = datetime.date(2017, 7, 1)

# st.sidebar.header("데이터 및 전략 설정")
# start_date = st.sidebar.date_input("시작 날짜", default_start_date_5_years_ago)
# end_date = st.sidebar.date_input("종료 날짜", default_end_date)

# if start_date >= end_date:
#     st.sidebar.error("종료 날짜는 시작 날짜보다 미래여야 합니다.")
#     st.stop()
# elif start_date < min_valid_date_for_most_exchanges:
#     st.sidebar.warning(f"대부분의 주요 암호화폐 거래소는 {min_valid_date_for_most_exchanges} 이후부터 데이터가 존재합니다. 해당 날짜 이후로 설정하시면 더 많은 데이터를 얻을 수 있습니다.")

# @st.cache_data
# def load_crypto_data(symbol, timeframe, start_date_obj, end_date_obj):
#     exchange = ccxt.binance() # 원하는 거래소를 선택 (예: ccxt.upbit(), ccxt.coinbasepro(), etc.)
    
#     start_datetime = datetime.datetime(start_date_obj.year, start_date_obj.month, start_date_obj.day)
#     end_datetime = datetime.datetime(end_date_obj.year, end_date_obj.month, end_date_obj.day)

#     since_timestamp_ms = exchange.parse8601(start_datetime.isoformat())
    
#     ohlcv = []
#     while True:
#         try:
#             chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp_ms, limit=1000)
#             if not chunk:
#                 break
#             ohlcv.extend(chunk)
#             since_timestamp_ms = chunk[-1][0] + (24 * 60 * 60 * 1000) # 1일 = 86400000ms

#             if since_timestamp_ms > end_datetime.timestamp() * 1000:
#                 break
            
#             time.sleep(0.05) # Rate Limit 준수

#         except ccxt.NetworkError as e:
#             st.warning(f"네트워크 오류: {e}. 잠시 후 다시 시도합니다.")
#             time.sleep(5)
#         except ccxt.ExchangeError as e:
#             st.error(f"거래소 오류: {e}. 데이터 가져오기를 중단합니다. Rate Limit에 도달했을 수 있습니다. 잠시 후 다시 시도해보세요.")
#             return pd.DataFrame()
#         except Exception as e:
#             st.error(f"알 수 없는 오류: {e}. 데이터 가져오기를 중단합니다.")
#             return pd.DataFrame()

#     if not ohlcv:
#         return pd.DataFrame()

#     df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     df.set_index('timestamp', inplace=True)
#     df = df.loc[start_datetime:end_datetime] 
#     df['Adj Close'] = df['close'] # 'Adj Close' 컬럼 생성

#     return df # OHLCV 전체 데이터를 반환합니다.

# # XRP/USDT를 BTC/USDT로 변경
# st.write(f"##### 데이터 다운로드: BTC/USDT ({start_date} ~ {end_date})")
# ohlcv_data = load_crypto_data("BTC/USDT", "1d", start_date, end_date)

# # 디버깅을 위한 데이터 로딩 정보 출력
# st.write(f"로드된 데이터의 행 개수: {ohlcv_data.shape[0]}개")
# # st.write(f"로드된 데이터의 처음 5개 행:\n {ohlcv_data.head()}")
# # st.write(f"로드된 데이터의 마지막 5개 행:\n {ohlcv_data.tail()}")


# if ohlcv_data.empty:
#     st.error("지정된 날짜 범위에 대한 데이터를 다운로드할 수 없습니다. 날짜 범위나 선택된 거래소의 데이터 유무를 확인해주세요.")
#     st.stop()


# # --- 지표 계산 함수 ---
# def calculate_indicators(df, use_sma, use_momentum, use_rsi,
#                          short_ma_period, long_ma_period, rsi_period, momentum_period):

#     # 이동평균선
#     if use_sma:
#         df['SMA_Short'] = df['Adj Close'].rolling(window=short_ma_period).mean()
#         df['SMA_Long'] = df['Adj Close'].rolling(window=long_ma_period).mean()
#     else:
#         df['SMA_Short'] = np.nan # 사용하지 않으면 NaN으로 초기화
#         df['SMA_Long'] = np.nan

#     # RSI
#     if use_rsi:
#         delta = df['Adj Close'].diff(1)
#         gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
#         # 0으로 나누는 오류 방지
#         rs = np.where(loss == 0, np.inf, gain / loss) 
#         df['RSI'] = 100 - (100 / (1 + rs))
#     else:
#         df['RSI'] = np.nan

#     # 모멘텀 (현재 종가 / N일 전 종가 * 100)
#     if use_momentum:
#         df['Momentum'] = df['Adj Close'].pct_change(momentum_period) * 100
#     else:
#         df['Momentum'] = np.nan
    
#     return df

# st.sidebar.subheader("이동평균선 설정")
# use_sma = st.sidebar.checkbox("이동평균선 사용", value=True)
# short_ma_period = st.sidebar.slider("단기 이동평균선 기간 (일)", 5, 50, 20) if use_sma else 0
# long_ma_period = st.sidebar.slider("장기 이동평균선 기간 (일)", 30, 200, 60) if use_sma else 0

# if use_sma and short_ma_period >= long_ma_period:
#     st.sidebar.error("단기 이동평균선 기간은 장기 이동평균선 기간보다 작아야 합니다.")
#     st.stop()

# st.sidebar.subheader("모멘텀 지표 설정")
# use_momentum = st.sidebar.checkbox("모멘텀 사용", value=False)
# momentum_period = st.sidebar.slider("모멘텀 기간 (일)", 5, 30, 14) if use_momentum else 0
# momentum_buy_threshold = st.sidebar.slider("모멘텀 매수 임계값 (%)", -10.0, 10.0, 0.5, step=0.1) if use_momentum else 0
# momentum_sell_threshold = st.sidebar.slider("모멘텀 매도 임계값 (%)", -10.0, 10.0, -0.5, step=0.1) if use_momentum else 0

# st.sidebar.subheader("RSI 지표 설정")
# use_rsi = st.sidebar.checkbox("RSI 사용", value=False)
# rsi_period = st.sidebar.slider("RSI 기간 (일)", 5, 30, 14) if use_rsi else 0
# rsi_buy_threshold = st.sidebar.slider("RSI 매수 임계값", 20, 40, 30) if use_rsi else 0 # 과매도
# rsi_sell_threshold = st.sidebar.slider("RSI 매도 임계값", 60, 80, 70) if use_rsi else 0 # 과매수

# # 지표 계산
# processed_data = calculate_indicators(ohlcv_data.copy(), 
#                                       use_sma, use_momentum, use_rsi,
#                                       short_ma_period, long_ma_period, rsi_period, momentum_period)

# if processed_data.empty:
#     st.error("지표 계산에 필요한 데이터를 처리할 수 없습니다. 데이터 기간을 확인해주세요.")
#     st.stop()
    
# # --- 백테스팅 함수 ---
# def backtest_strategy(df, use_sma, use_momentum, use_rsi,
#                       short_ma_period, long_ma_period,
#                       momentum_buy_threshold, momentum_sell_threshold,
#                       rsi_buy_threshold, rsi_sell_threshold):
    
#     # 전략에 필요한 컬럼 목록을 동적으로 생성
#     cols_to_check = ['Adj Close']
#     if use_sma:
#         cols_to_check.extend(['SMA_Short', 'SMA_Long'])
#     if use_momentum:
#         cols_to_check.append('Momentum')
#     if use_rsi:
#         cols_to_check.append('RSI')

#     # 필요한 컬럼들에서 NaN이 있는 행만 제거
#     df.dropna(subset=cols_to_check, inplace=True) 

#     if df.empty:
#         return pd.DataFrame()

#     df['Position'] = 0
#     df['Strategy_Return'] = 0.0
#     df['Cumulative_Strategy_Return'] = 1.0
#     df['Cumulative_Buy_And_Hold_Return'] = 1.0
#     df['Buy_Signal'] = False
#     df['Sell_Signal'] = False
    
#     # in_position 변수를 for 루프 시작 전에 초기화합니다.
#     in_position = False 

#     # 백테스팅 시작 시점의 누적 수익률 초기화 (dropna 후 첫 인덱스 기준)
#     if len(df) > 0: # 데이터 프레임이 비어있지 않은지 다시 한번 확인
#         df.loc[df.index[0], 'Cumulative_Strategy_Return'] = 1.0
#         df.loc[df.index[0], 'Cumulative_Buy_And_Hold_Return'] = 1.0
#     else: # 이 경우는 df.empty에서 걸러져야 하지만, 혹시 모를 상황 대비
#         return pd.DataFrame()
    

#     for i in range(1, len(df)):
#         current_date = df.index[i]

#         # 매수 조건
#         buy_condition_sma = False
#         if use_sma:
#             if df['SMA_Short'].iloc[i-1] < df['SMA_Long'].iloc[i-1] and \
#                df['SMA_Short'].iloc[i] >= df['SMA_Long'].iloc[i]:
#                 buy_condition_sma = True
#         else:
#             buy_condition_sma = True # SMA를 사용하지 않으면 조건 항상 충족

#         buy_condition_momentum = False
#         if use_momentum:
#             if df['Momentum'].iloc[i] > momentum_buy_threshold:
#                 buy_condition_momentum = True
#         else:
#             buy_condition_momentum = True

#         buy_condition_rsi = False
#         if use_rsi:
#             if df['RSI'].iloc[i] < rsi_buy_threshold: # RSI가 낮으면 과매도 -> 매수 신호
#                 buy_condition_rsi = True
#         else:
#             buy_condition_rsi = True
        
#         # 모든 활성화된 지표의 매수 조건이 동시에 충족될 때 매수
#         if not in_position and \
#            buy_condition_sma and \
#            buy_condition_momentum and \
#            buy_condition_rsi:
#             df.loc[current_date, 'Position'] = 1
#             df.loc[current_date, 'Buy_Signal'] = True
#             in_position = True
            
#         # 매도 조건
#         sell_condition_sma = False
#         if use_sma:
#             if df['SMA_Short'].iloc[i-1] > df['SMA_Long'].iloc[i-1] and \
#                df['SMA_Short'].iloc[i] <= df['SMA_Long'].iloc[i]:
#                 sell_condition_sma = True
#         else:
#             sell_condition_sma = True

#         sell_condition_momentum = False
#         if use_momentum:
#             if df['Momentum'].iloc[i] < momentum_sell_threshold:
#                 sell_condition_momentum = True
#         else:
#             sell_condition_momentum = True
            
#         sell_condition_rsi = False
#         if use_rsi:
#             if df['RSI'].iloc[i] > rsi_sell_threshold: # RSI가 높으면 과매수 -> 매도 신호
#                 sell_condition_rsi = True
#         else:
#             sell_condition_rsi = True
        
#         # 모든 활성화된 지표의 매도 조건이 동시에 충족될 때 매도
#         if in_position and \
#            sell_condition_sma and \
#            sell_condition_momentum and \
#            sell_condition_rsi:
#             df.loc[current_date, 'Position'] = 0
#             df.loc[current_date, 'Sell_Signal'] = True
#             in_position = False

#         # 수익률 계산
#         daily_return = (df['Adj Close'].iloc[i] / df['Adj Close'].iloc[i-1]) - 1

#         if in_position:
#             df.loc[current_date, 'Strategy_Return'] = daily_return
#         else:
#             df.loc[current_date, 'Strategy_Return'] = 0.0

#         # 누적 수익률 계산
#         df.loc[current_date, 'Cumulative_Strategy_Return'] = \
#             df['Cumulative_Strategy_Return'].iloc[i-1] * (1 + df.loc[current_date, 'Strategy_Return'])
#         df.loc[current_date, 'Cumulative_Buy_And_Hold_Return'] = \
#             df['Cumulative_Buy_And_Hold_Return'].iloc[i-1] * (1 + daily_return)
            
#     return df

# st.write("### 백테스팅 결과")
# results = backtest_strategy(processed_data.copy(), use_sma, use_momentum, use_rsi,
#                             short_ma_period, long_ma_period,
#                             momentum_buy_threshold, momentum_sell_threshold,
#                             rsi_buy_threshold, rsi_sell_threshold)

# if results.empty:
#     st.error("백테스팅 결과를 생성할 수 없습니다. 데이터 기간 및 지표 기간을 다시 확인해주세요.")
#     st.stop()
    
# # --- 결과 시각화 ---
# fig = plt.figure(figsize=(14, 10))
# gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1]) # 3:1:1 비율로 세 개의 서브플롯

# ax1 = fig.add_subplot(gs[0]) # 가격 및 이동평균선, 신호
# ax2 = fig.add_subplot(gs[1], sharex=ax1) # 누적 수익률
# ax3 = fig.add_subplot(gs[2], sharex=ax1) # 지표 그래프 (RSI, 모멘텀)

# # 상단 그래프 (가격, MA, 신호)
# # 'ripple price' -> 'Bitcoin Price' 로 변경
# ax1.plot(results.index, results['Adj Close'], label='Bitcoin Price', color='lightgray', linewidth=1)
# if use_sma:
#     ax1.plot(results.index, results['SMA_Short'], label=f'short MA ({short_ma_period}day)', color='orange', linewidth=1.5)
#     ax1.plot(results.index, results['SMA_Long'], label=f'long MA ({long_ma_period}day)', color='purple', linewidth=1.5)

# buy_signals = results[results['Buy_Signal'] == True]
# ax1.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', s=100, label='buy signal', zorder=5)

# sell_signals = results[results['Sell_Signal'] == True]
# ax1.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', s=100, label='sell signal', zorder=5)

# # '($USDT)' -> '가격 (USDT)' 로 변경
# ax1.set_ylabel("$(USDT)")
# ax1.legend(loc='upper left')
# ax1.grid(True)
# # 'Ripple Price, Moving Average, and Trading Signals' -> '비트코인 가격, 이동평균선 및 매매 신호' 로 변경
# ax1.set_title("Bitcoin Price, Moving Average, and Trading Signals")


# # 중간 그래프 (누적 수익률)
# ax2.plot(results.index, results['Cumulative_Strategy_Return'], label='Strategic Accumulated Return', color='blue', linewidth=2)
# ax2.plot(results.index, results['Cumulative_Buy_And_Hold_Return'], label='cumulative return on holdings after purchase', color='green', linestyle='--', linewidth=2)
# ax2.set_ylabel("cumulative return")
# ax2.legend(loc='upper left')
# ax2.grid(True)
# ax2.set_title("Comparison of cumulative returns")


# # 하단 그래프 (지표)
# if use_rsi:
#     ax3.plot(results.index, results['RSI'], label='RSI', color='cyan', linewidth=1)
#     ax3.axhline(y=rsi_buy_threshold, color='green', linestyle='--', label=f'RSI buy ({rsi_buy_threshold})')
#     ax3.axhline(y=rsi_sell_threshold, color='red', linestyle='--', label=f'RSI sell ({rsi_sell_threshold})')
# if use_momentum:
#     ax3.plot(results.index, results['Momentum'], label='momentum', color='magenta', linewidth=1)
#     ax3.axhline(y=momentum_buy_threshold, color='green', linestyle=':', label=f'momentum buy ({momentum_buy_threshold})')
#     ax3.axhline(y=momentum_sell_threshold, color='red', linestyle=':', label=f'momentum sell ({momentum_sell_threshold})')


# ax3.set_xlabel("date") # 'data' -> '날짜' 로 변경
# ax3.set_ylabel("지표 값")
# ax3.legend(loc='upper left')
# ax3.grid(True)
# ax3.set_title("기술 지표")

# fig.autofmt_xdate()
# st.pyplot(fig)

# # 최종 수익률 요약
# final_strategy_return = (results['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
# final_buy_and_hold_return = (results['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) * 100

# st.metric(label="최종 전략 누적 수익률", value=f"{final_strategy_return:.2f}%")
# st.metric(label="최종 매수 후 보유 (Buy & Hold) 누적 수익률", value=f"{final_buy_and_hold_return:.2f}%")

# st.write("---")
# st.write("### 백테스팅 상세 데이터 (일부)")
# st.dataframe(results[['Adj Close', 'SMA_Short', 'SMA_Long', 'RSI', 'Momentum', 'Buy_Signal', 'Sell_Signal', 'Position', 'Strategy_Return', 'Cumulative_Strategy_Return', 'Cumulative_Buy_And_Hold_Return']].tail(20))

# st.write("---")
# st.write("### 참고")
# st.write("""
# - 이 백테스팅 결과는 과거 데이터를 기반으로 하며, 미래 수익을 보장하지 않습니다.
# - 실제 거래에서는 거래 수수료, 슬리피지, 유동성 등 다양한 요인이 고려되어야 합니다.
# - CCXT를 통한 데이터 수집 시 API Rate Limit 등으로 인해 모든 데이터를 가져오지 못할 수 있습니다.
# - **각 지표의 임계값은 예시이며, 최적의 조합과 값은 끊임없는 연구와 백테스팅을 통해 찾아야 합니다.**
# """)

