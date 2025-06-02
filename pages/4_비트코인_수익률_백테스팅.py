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
st.write("##### RSI와 모멘텀 지표 설정은 좌측 사이드바 메뉴에서 가능합니다.") # 문구 변경

# 기본 날짜 설정
default_end_date = datetime.date.today()
# 2017년 이후 데이터가 많으므로 시작일을 조정하는 것이 좋습니다.
# 바이낸스 BTC/USDT는 2017-08-17부터 데이터가 있습니다.
min_valid_date_for_binance_btc = datetime.date(2017, 8, 17)
# 기본 시작 날짜는 5년 전 또는 바이낸스 최소 유효 날짜 중 더 늦은 날짜로 설정
default_start_date = max(min_valid_date_for_binance_btc, default_end_date - datetime.timedelta(days=365 * 5)) 


st.sidebar.header("데이터 및 전략 설정")
start_date = st.sidebar.date_input("시작 날짜", default_start_date) # 기본 시작 날짜 변경
end_date = st.sidebar.date_input("종료 날짜", default_end_date)

if start_date >= end_date:
    st.sidebar.error("❌ 종료 날짜는 시작 날짜보다 미래여야 합니다.")
    st.stop()
elif start_date < min_valid_date_for_binance_btc: # 최소 유효 날짜 경고
    st.sidebar.warning(f"⚠️ 바이낸스 BTC/USDT 데이터는 {min_valid_date_for_binance_btc} 이후부터 존재합니다. 해당 날짜 이후로 설정하시면 더 많은 데이터를 얻을 수 있습니다.")

@st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
def load_crypto_data(symbol, timeframe, start_date_obj, end_date_obj):
    exchange = ccxt.binance({
        'enableRateLimit': True, # 초당 요청 제한 준수
    })
    
    st.info(f"🔄 바이낸스에서 **{symbol}** ({timeframe}) 데이터를 가져오는 중...")

    # 시작 및 종료 날짜를 타임스탬프 (밀리초)로 변환
    # UTC 기준 00:00:00 (시작일) 및 23:59:59 (종료일)
    # yyyy-MM-ddThh:mm:ssZ 형식으로 정확한 시간까지 포함하여 변환하는 것이 좋습니다.
    start_timestamp_ms = exchange.parse8601(start_date_obj.isoformat() + 'T00:00:00Z')
    end_timestamp_ms = exchange.parse8601(end_date_obj.isoformat() + 'T23:59:59Z') # 종료일 포함

    ohlcv = []
    current_timestamp_ms = start_timestamp_ms # 현재 요청할 데이터의 시작 타임스탬프

    # 1일의 밀리초
    one_day_in_ms = 24 * 60 * 60 * 1000 

    # 진행률 표시를 위한 위젯 추가
    progress_bar = st.progress(0)
    status_text = st.empty()

    fetch_count = 0
    # 현재 타임스탬프가 요청 종료 타임스탬프를 넘지 않을 때까지 반복
    while current_timestamp_ms <= end_timestamp_ms:
        try:
            display_date = datetime.datetime.fromtimestamp(current_timestamp_ms / 1000).strftime('%Y-%m-%d')
            status_text.text(f"데이터 수집 중: {display_date} 부터...")
            
            # Binance는 fetch_ohlcv의 limit이 최대 1000개입니다.
            chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp_ms, limit=1000)
            
            if not chunk: # 더 이상 데이터가 없으면 중단
                status_text.text("데이터 수집 완료 (추가 데이터 없음).")
                break 

            ohlcv.extend(chunk)
            
            # 다음 요청을 위한 since_timestamp_ms 업데이트 (가져온 마지막 데이터의 다음 봉)
            # 마지막 봉의 타임스탬프 + 1일 (일봉 기준)
            current_timestamp_ms = chunk[-1][0] + one_day_in_ms 

            # 진행률 업데이트 (대략적인 계산)
            progress_percentage = (current_timestamp_ms - start_timestamp_ms) / (end_timestamp_ms - start_timestamp_ms + one_day_in_ms)
            progress_bar.progress(min(1.0, progress_percentage)) # 최대 100%를 넘지 않도록
            
            fetch_count += 1
            # 바이낸스 public API는 Rate Limit이 비교적 넉넉하지만, 안전을 위해 sleep 추가
            time.sleep(0.05) # 0.05초 대기

        except ccxt.NetworkError as e:
            st.warning(f"네트워크 오류: {e}. 잠시 후 다시 시도합니다.")
            time.sleep(5) # 5초 대기 후 재시도
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
            
    progress_bar.empty() # 진행률 바 숨김
    status_text.empty() # 상태 텍스트 숨김

    if not ohlcv:
        st.warning("⚠️ 지정된 기간 동안 데이터를 가져오지 못했습니다. 날짜 범위 또는 심볼을 확인하세요.")
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # 데이터 필터링: 요청한 날짜 범위에 맞게 정확히 필터링 (불필요한 미래 데이터 제거)
    # yfinance나 다른 API와 다르게 ccxt는 요청한 'since'부터 데이터를 가져오므로 
    # end_date_obj를 기준으로 한번 더 필터링해주는 것이 좋습니다.
    df = df.loc[pd.to_datetime(start_date_obj):pd.to_datetime(end_date_obj)].copy()
    
    df['Adj Close'] = df['close'] # 'Adj Close' 컬럼 생성

    st.success(f"✅ **{symbol}** 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
    return df

# BTC/USDT 사용
st.write(f"##### 데이터 다운로드: **BTC/USDT** ({start_date} ~ {end_date})")
ohlcv_data = load_crypto_data("BTC/USDT", "1d", start_date, end_date)

# 디버깅 정보는 실제 배포 시에는 삭제하거나 주석 처리하는 것이 좋습니다.
# st.write(f"로드된 데이터의 행 개수: {ohlcv_data.shape[0]}개")
# st.write(f"로드된 데이터의 처음 5개 행:\n {ohlcv_data.head()}")
# st.write(f"로드된 데이터의 마지막 5개 행:\n {ohlcv_data.tail()}")


if ohlcv_data.empty:
    st.error("지정된 날짜 범위에 대한 데이터를 다운로드할 수 없습니다. 날짜 범위나 선택된 거래소의 데이터 유무를 확인해주세요.")
    st.stop()


# --- 지표 계산 함수 ---
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
        # 0으로 나누는 오류 방지: loss가 0이면 rs를 무한대로 설정 (RSI는 100)
        rs = np.where(loss == 0, np.inf, gain / loss) 
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = np.nan

    # 모멘텀 (현재 종가 / N일 전 종가 * 100 - 100)
    if use_momentum:
        df['Momentum'] = df['Adj Close'].pct_change(momentum_period) * 100
    else:
        df['Momentum'] = np.nan
    
    return df

st.sidebar.subheader("📊 이동평균선 설정") # 이모티콘 추가
use_sma = st.sidebar.checkbox("이동평균선 사용", value=True)
short_ma_period = st.sidebar.slider("단기 이동평균선 기간 (일)", 5, 50, 20) if use_sma else 0
long_ma_period = st.sidebar.slider("장기 이동평균선 기간 (일)", 30, 200, 60) if use_sma else 0

if use_sma and short_ma_period >= long_ma_period:
    st.sidebar.error("❌ 단기 이동평균선 기간은 장기 이동평균선 기간보다 작아야 합니다.")
    st.stop()

st.sidebar.subheader("📈 모멘텀 지표 설정") # 이모티콘 추가
use_momentum = st.sidebar.checkbox("모멘텀 사용", value=False)
momentum_period = st.sidebar.slider("모멘텀 기간 (일)", 5, 30, 14) if use_momentum else 0
momentum_buy_threshold = st.sidebar.slider("모멘텀 매수 임계값 (%)", -10.0, 10.0, 0.5, step=0.1) if use_momentum else 0
momentum_sell_threshold = st.sidebar.slider("모멘텀 매도 임계값 (%)", -10.0, 10.0, -0.5, step=0.1) if use_momentum else 0

st.sidebar.subheader("📉 RSI 지표 설정") # 이모티콘 추가
use_rsi = st.sidebar.checkbox("RSI 사용", value=False)
rsi_period = st.sidebar.slider("RSI 기간 (일)", 5, 30, 14) if use_rsi else 0
rsi_buy_threshold = st.sidebar.slider("RSI 매수 임계값 (과매도)", 20, 40, 30) if use_rsi else 0 # 과매도
rsi_sell_threshold = st.sidebar.slider("RSI 매도 임계값 (과매수)", 60, 80, 70) if use_rsi else 0 # 과매수

# 지표 계산
processed_data = calculate_indicators(ohlcv_data.copy(), 
                                      use_sma, use_momentum, use_rsi,
                                      short_ma_period, long_ma_period, rsi_period, momentum_period)

if processed_data.empty:
    st.error("지표 계산에 필요한 데이터를 처리할 수 없습니다. 기간 설정을 다시 확인해주세요.") # 오류 메시지 변경
    st.stop()
    
# --- 백테스팅 함수 ---
def backtest_strategy(df, use_sma, use_momentum, use_rsi,
                      short_ma_period, long_ma_period, # 지표 계산에만 사용되는 인자 유지
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

    # 필요한 컬럼들에서 NaN이 있는 행만 제거 (시작 부분)
    df.dropna(subset=cols_to_check, inplace=True) 

    if df.empty:
        st.warning("선택된 지표를 적용한 후 유효한 데이터가 없습니다. 기간 설정을 다시 확인해주세요.") # 경고 메시지 추가
        return pd.DataFrame()

    df['Position'] = 0
    df['Strategy_Return'] = 0.0
    df['Cumulative_Strategy_Return'] = 1.0
    df['Cumulative_Buy_And_Hold_Return'] = 1.0
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    
    in_position = False 

    # 백테스팅 시작 시점의 누적 수익률 초기화 (dropna 후 첫 인덱스 기준)
    # 데이터프레임이 비어있지 않은지 다시 한번 확인
    if len(df) > 0: 
        df.loc[df.index[0], 'Cumulative_Strategy_Return'] = 1.0
        df.loc[df.index[0], 'Cumulative_Buy_And_Hold_Return'] = 1.0
    else: 
        return pd.DataFrame() # 데이터가 없으면 빈 DataFrame 반환
    

    for i in range(1, len(df)):
        current_date = df.index[i]

        # 매수 조건
        buy_condition_sma = not use_sma or (df['SMA_Short'].iloc[i-1] < df['SMA_Long'].iloc[i-1] and \
                                            df['SMA_Short'].iloc[i] >= df['SMA_Long'].iloc[i])

        buy_condition_momentum = not use_momentum or (df['Momentum'].iloc[i] > momentum_buy_threshold)
        
        buy_condition_rsi = not use_rsi or (df['RSI'].iloc[i] < rsi_buy_threshold)
        
        # 모든 활성화된 지표의 매수 조건이 동시에 충족될 때 매수 (AND 조건)
        if not in_position and buy_condition_sma and buy_condition_momentum and buy_condition_rsi:
            df.loc[current_date, 'Position'] = 1
            df.loc[current_date, 'Buy_Signal'] = True
            in_position = True
            
        # 매도 조건
        sell_condition_sma = not use_sma or (df['SMA_Short'].iloc[i-1] > df['SMA_Long'].iloc[i-1] and \
                                             df['SMA_Short'].iloc[i] <= df['SMA_Long'].iloc[i])

        sell_condition_momentum = not use_momentum or (df['Momentum'].iloc[i] < momentum_sell_threshold)
            
        sell_condition_rsi = not use_rsi or (df['RSI'].iloc[i] > rsi_sell_threshold)
        
        # 모든 활성화된 지표의 매도 조건이 동시에 충족될 때 매도 (AND 조건)
        if in_position and sell_condition_sma and sell_condition_momentum and sell_condition_rsi:
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
        df.loc[current_date, 'Cumulative_Strategy_Return'] = \
            df['Cumulative_Strategy_Return'].iloc[i-1] * (1 + df.loc[current_date, 'Strategy_Return'])
        df.loc[current_date, 'Cumulative_Buy_And_Hold_Return'] = \
            df['Cumulative_Buy_And_Hold_Return'].iloc[i-1] * (1 + daily_return)
            
    return df

st.write("### 📈 백테스팅 결과") # 이모티콘 추가
results = backtest_strategy(processed_data.copy(), use_sma, use_momentum, use_rsi,
                            short_ma_period, long_ma_period,
                            momentum_buy_threshold, momentum_sell_threshold,
                            rsi_buy_threshold, rsi_sell_threshold)

if results.empty:
    st.error("백테스팅 결과를 생성할 수 없습니다. 데이터 기간 및 지표 기간 설정을 다시 확인해주세요.") # 오류 메시지 변경
    st.stop()
    
# --- 결과 시각화 ---
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1]) 

ax1 = fig.add_subplot(gs[0]) # 가격 및 이동평균선, 신호
ax2 = fig.add_subplot(gs[1], sharex=ax1) # 누적 수익률
ax3 = fig.add_subplot(gs[2], sharex=ax1) # 지표 그래프 (RSI, 모멘텀)

# 상단 그래프 (가격, MA, 신호)
ax1.plot(results.index, results['Adj Close'], label='비트코인 가격', color='lightgray', linewidth=1) # 라벨 변경
if use_sma:
    ax1.plot(results.index, results['SMA_Short'], label=f'단기 이평선 ({short_ma_period}일)', color='orange', linewidth=1.5) # 라벨 변경
    ax1.plot(results.index, results['SMA_Long'], label=f'장기 이평선 ({long_ma_period}일)', color='purple', linewidth=1.5) # 라벨 변경

buy_signals = results[results['Buy_Signal'] == True]
ax1.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', s=100, label='매수 신호', zorder=5) # 라벨 변경

sell_signals = results[results['Sell_Signal'] == True]
ax1.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', s=100, label='매도 신호', zorder=5) # 라벨 변경

ax1.set_ylabel("가격 (USDT)") # 라벨 변경 (USDT로 가정)
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_title("비트코인 가격, 이동평균선 및 매매 신호") # 제목 변경


# 중간 그래프 (누적 수익률)
ax2.plot(results.index, (results['Cumulative_Strategy_Return'] - 1) * 100, label='전략 누적 수익률 (%)', color='blue', linewidth=2) # 라벨 변경 및 % 변환
ax2.plot(results.index, (results['Cumulative_Buy_And_Hold_Return'] - 1) * 100, label='매수 후 보유 누적 수익률 (%)', color='green', linestyle='--', linewidth=2) # 라벨 변경 및 % 변환
ax2.set_ylabel("누적 수익률 (%)") # 라벨 변경
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_title("누적 수익률 비교") # 제목 변경


# 하단 그래프 (지표)
if use_rsi:
    ax3.plot(results.index, results['RSI'], label='RSI', color='cyan', linewidth=1)
    ax3.axhline(y=rsi_buy_threshold, color='green', linestyle='--', label=f'RSI 매수 ({rsi_buy_threshold})') # 라벨 변경
    ax3.axhline(y=rsi_sell_threshold, color='red', linestyle='--', label=f'RSI 매도 ({rsi_sell_threshold})') # 라벨 변경
if use_momentum:
    ax3.plot(results.index, results['Momentum'], label='모멘텀', color='magenta', linewidth=1) # 라벨 변경
    ax3.axhline(y=momentum_buy_threshold, color='green', linestyle=':', label=f'모멘텀 매수 ({momentum_buy_threshold})') # 라벨 변경
    ax3.axhline(y=momentum_sell_threshold, color='red', linestyle=':', label=f'모멘텀 매도 ({momentum_sell_threshold})') # 라벨 변경


ax3.set_xlabel("날짜") # 라벨 변경
ax3.set_ylabel("지표 값")
ax3.legend(loc='upper left')
ax3.grid(True)
ax3.set_title("기술 지표") # 제목 변경

fig.autofmt_xdate()
st.pyplot(fig)

# 최종 수익률 요약
final_strategy_return = (results['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
final_buy_and_hold_return = (results['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) * 100

st.metric(label="최종 전략 누적 수익률", value=f"{final_strategy_return:.2f}%")
st.metric(label="최종 매수 후 보유 (Buy & Hold) 누적 수익률", value=f"{final_buy_and_hold_return:.2f}%")

st.write("---")
st.write("### 📝 참고")
st.write("""
- **데이터 출처**: 이 앱은 **바이낸스(Binance) BTC/USDT 일봉 데이터**를 기반으로 작동합니다.
- **백테스팅 모델의 한계**: 제시된 수익률은 백테스팅 결과이며, 실제 투자 결과와는 다를 수 있습니다. 거래 수수료, 슬리피지(Slippage), 세금, 시스템 지연 등의 실제 거래 환경 요소를 고려하지 않은 단순 시뮬레이션입니다.
- **면책 조항**: 본 정보는 투자 자문이 아니며, 여기에 제시된 내용은 오직 정보 제공을 목적으로 합니다. 투자 결정은 사용자 본인의 판단과 책임 하에 이루어져야 합니다.
- **API 키 관리**: 바이낸스의 경우 Public API는 API 키 없이 OHLCV 데이터를 가져올 수 있습니다. 만약 개인 정보가 필요한 API를 사용한다면 `st.secrets`를 통해 안전하게 관리해야 합니다.
""")
st.write("---")
st.write("### 백테스팅 상세 데이터 (최근 20일)") # 제목 변경
st.dataframe(results[['Adj Close', 'SMA_Short', 'SMA_Long', 'RSI', 'Momentum', 'Buy_Signal', 'Sell_Signal', 'Position', 'Strategy_Return', 'Cumulative_Strategy_Return', 'Cumulative_Buy_And_Hold_Return']].tail(20))
