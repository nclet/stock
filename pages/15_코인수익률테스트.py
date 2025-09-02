import streamlit as st
import pandas as pd
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.parse
from json.decoder import JSONDecodeError
import time

# --- Streamlit 설정 및 데이터 다운로드 ---
st.set_page_config(layout="wide")
st.title("📈 암호화폐 투자 지표 백테스팅")
st.write("##### 업비트 KRW 마켓의 다양한 암호화폐에 대해 백테스팅을 실행할 수 있습니다.")

# ------------------------
# ✨ 한글 폰트 설정
# ------------------------
def get_korean_font():
    """시스템에 설치된 한글 폰트를 찾아 Matplotlib에 설정합니다."""
    font_path = ""
    for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'NanumGothic' in font:
            font_path = font
            break
        elif 'Malgun Gothic' in font:
            font_path = font
            break
        elif 'AppleGothic' in font:
            font_path = font
            break
    
    if font_path:
        fm.fontManager.addfont(font_path)
        plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
        plt.rc('axes', unicode_minus=False) # 마이너스 폰트 깨짐 방지
        st.info(f"✅ 한글 폰트 '{fm.FontProperties(fname=font_path).get_name()}'가 성공적으로 설정되었습니다.")
    else:
        st.warning("⚠️ 시스템에 한글 폰트(나눔고딕, 맑은고딕 등)가 설치되어 있지 않습니다. 차트의 한글이 깨질 수 있습니다.")

get_korean_font()


# ------------------------
# ✨ 암호화폐 종목 목록 로드 (Upbit API)
# ------------------------
@st.cache_data
def get_upbit_markets():
    """
    Upbit API에서 원화(KRW) 마켓에 있는 모든 암호화폐 목록을 가져옵니다.
    """
    url = "https://api.upbit.com/v1/market/all"
    try:
        response = requests.get(url, params={'isDetails': 'false'})
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
        markets = response.json()
        
        # KRW 마켓만 필터링하고 코인 이름으로 매핑
        krw_markets = {market['korean_name']: market['market'] for market in markets if market['market'].startswith('KRW-')}
        
        if not krw_markets:
            st.error("❌ Upbit API에서 원화 마켓 목록을 가져오지 못했습니다.")
            st.info("Upbit API 서버 상태를 확인하거나 잠시 후 다시 시도해주세요.")
            st.stop()
        
        return krw_markets
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Upbit API 연결 오류: {e}")
        st.info("인터넷 연결 상태를 확인하거나 Upbit 서버에 문제가 있을 수 있습니다.")
        st.stop()
        return {}
    except JSONDecodeError as e:
        st.error(f"❌ Upbit API 응답 파싱 오류: {e}")
        st.stop()
        return {}

crypto_list = get_upbit_markets()
company_names = list(crypto_list.keys())

# --- 데이터 및 전략 설정 ---
st.header("데이터 및 전략 설정")

# 암호화폐 종목 선택 UI
default_crypto = "비트코인"
if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
    st.session_state.selected_company = default_crypto if default_crypto in company_names else company_names[0]

company_name = st.selectbox(
    "✅ 분석할 암호화폐 선택",
    company_names,
    index=company_names.index(st.session_state.selected_company),
    key="selected_company"
)
symbol = crypto_list.get(st.session_state.selected_company)

# 날짜 설정
default_end_date = datetime.date.today()
default_start_date = default_end_date - datetime.timedelta(days=365 * 5)
start_date = st.date_input("시작 날짜", default_start_date)
end_date = st.date_input("종료 날짜", default_end_date)

if start_date >= end_date:
    st.error("❌ 종료 날짜는 시작 날짜보다 미래여야 합니다.")
    st.stop()

# ------------------------
# ✨ Upbit API 함수 (ccxt 대신 requests 사용)
# ------------------------
@st.cache_data(ttl=3600)
def load_crypto_data(symbol, start_date, end_date):
    """
    Upbit API를 통해 일별 캔들 데이터를 가져와 DataFrame으로 반환합니다.
    """
    base_url = "https://api.upbit.com/v1/candles/days"
    df_list = []
    current_date = end_date
    max_requests = 20 # 200일씩 20번 요청 (총 4000일, 약 10년치)
    requests_count = 0
    
    st.info(f"🔄 업비트에서 **{symbol}** 데이터를 수집하고 있습니다...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    while current_date >= start_date and requests_count < max_requests:
        params = {
            'market': symbol,
            'to': (current_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
            'count': 200
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            temp_df = pd.DataFrame(data)
            temp_df['timestamp'] = pd.to_datetime(temp_df['candle_date_time_kst'])
            temp_df = temp_df.rename(columns={'opening_price': 'open', 'high_price': 'high', 'low_price': 'low', 'trade_price': 'close', 'candle_acc_trade_volume': 'volume'})
            df_list.append(temp_df)
            
            current_date = temp_df['timestamp'].min().date() - datetime.timedelta(days=1)
            requests_count += 1
            
            progress_percentage = (end_date - current_date).days / (end_date - start_date).days
            progress_bar.progress(min(1.0, progress_percentage))
            status_text.text(f"데이터 수집 중: {current_date} 부터...")
            time.sleep(0.15)
        
        except requests.exceptions.RequestException as e:
            st.error(f"Upbit API 요청 실패: {e}")
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()
        except JSONDecodeError as e:
            st.error(f"Upbit API 응답 파싱 오류: {e}")
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame()

    progress_bar.empty()
    status_text.empty()

    if not df_list:
        st.warning("⚠️ 지정된 기간 동안 데이터를 가져오지 못했습니다. 날짜 범위를 확인하세요.")
        return pd.DataFrame()

    df_final = pd.concat(df_list, ignore_index=True)
    df_final = df_final.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)
    df_final = df_final[(df_final['timestamp'].dt.date >= start_date) & (df_final['timestamp'].dt.date <= end_date)].reset_index(drop=True)
    df_final.set_index('timestamp', inplace=True)
    
    # 원본 코드와의 호환성을 위해 Adj Close 컬럼 추가
    df_final['Adj Close'] = df_final['close']

    st.success(f"✅ **{company_name}** 데이터 로드 완료! ({df_final.index.min().date()} ~ {df_final.index.max().date()})")
    return df_final

# 지표 계산 함수
def calculate_indicators(df, use_sma, use_momentum, use_rsi, use_macd, use_obv,
                         short_ma_period, long_ma_period, rsi_period, momentum_period,
                         macd_fast_period, macd_slow_period, macd_signal_period):

    if df.empty:
        return pd.DataFrame()

    # SMA
    if use_sma:
        df['SMA_Short'] = df['Adj Close'].rolling(window=short_ma_period).mean()
        df['SMA_Long'] = df['Adj Close'].rolling(window=long_ma_period).mean()
    else:
        df['SMA_Short'] = np.nan
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

# --- 지표 설정 UI ---
st.subheader("📊 이동평균선 설정")
use_sma = st.checkbox("이동평균선 사용", value=True)
short_ma_period = st.slider("단기 이동평균선 기간 (일)", 5, 50, 20) if use_sma else 0
long_ma_period = st.slider("장기 이동평균선 기간 (일)", 30, 200, 60) if use_sma else 0

if use_sma and short_ma_period >= long_ma_period:
    st.error("❌ 단기 이동평균선 기간은 장기 이동평균선 기간보다 작아야 합니다.")
    st.stop()

st.subheader("📈 모멘텀 지표 설정")
use_momentum = st.checkbox("모멘텀 사용", value=False)
momentum_period = st.slider("모멘텀 기간 (일)", 5, 30, 14) if use_momentum else 0
momentum_buy_threshold = st.slider("모멘텀 매수 임계값 (%)", -10.0, 10.0, 0.5, step=0.1) if use_momentum else 0
momentum_sell_threshold = st.slider("모멘텀 매도 임계값 (%)", -10.0, 10.0, -0.5, step=0.1) if use_momentum else 0

st.subheader("📉 RSI 지표 설정")
use_rsi = st.checkbox("RSI 사용", value=False)
rsi_period = st.slider("RSI 기간 (일)", 5, 30, 14) if use_rsi else 0
rsi_buy_threshold = st.slider("RSI 매수 임계값 (과매도)", 20, 40, 30) if use_rsi else 0
rsi_sell_threshold = st.slider("RSI 매도 임계값 (과매수)", 60, 80, 70) if use_rsi else 0

st.subheader("📊 MACD 지표 설정")
use_macd = st.checkbox("MACD 사용", value=False)
macd_fast_period = st.slider("MACD 단기 EMA 기간 (일)", 5, 30, 12) if use_macd else 0
macd_slow_period = st.slider("MACD 장기 EMA 기간 (일)", 20, 50, 26) if use_macd else 0
macd_signal_period = st.slider("MACD 시그널 EMA 기간 (일)", 5, 15, 9) if use_macd else 0

if use_macd and macd_fast_period >= macd_slow_period:
    st.error("❌ MACD 단기 EMA 기간은 장기 EMA 기간보다 작아야 합니다.")
    st.stop()

st.subheader("📈 OBV 지표 설정")
use_obv = st.checkbox("OBV 사용", value=False)


# --- 백테스팅 실행 버튼 ---
if st.button("🚀 백테스팅 실행"):
    ohlcv_data = load_crypto_data(symbol, start_date, end_date)

    if ohlcv_data.empty:
        st.error("지정된 날짜 범위에 대한 데이터를 다운로드할 수 없습니다. 날짜 범위나 선택된 종목의 데이터 유무를 확인해주세요.")
        st.stop()
    
    # 지표 계산
    processed_data = calculate_indicators(ohlcv_data.copy(),
                                          use_sma, use_momentum, use_rsi, use_macd, use_obv,
                                          short_ma_period, long_ma_period, rsi_period, momentum_period,
                                          macd_fast_period, macd_slow_period, macd_signal_period)
    
    columns_to_check = ['Adj Close', 'volume']
    if use_sma: columns_to_check.extend(['SMA_Short', 'SMA_Long'])
    if use_rsi: columns_to_check.append('RSI')
    if use_momentum: columns_to_check.append('Momentum')
    if use_macd: columns_to_check.extend(['MACD', 'MACD_Signal'])
    if use_obv: columns_to_check.append('OBV')
    
    processed_data.dropna(subset=columns_to_check, inplace=True)
    
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
        df['High_Water_Mark'] = 1.0  # 최대 낙폭 계산을 위한 고점
        df['Drawdown'] = 0.0 # 최대 낙폭 계산용

        in_position = False
        first_index = df.index[0]
        df.loc[first_index, 'Cumulative_Strategy_Return'] = 1.0
        df.loc[first_index, 'Cumulative_Buy_And_Hold_Return'] = 1.0

        buy_signal_count = 0
        sell_signal_count = 0

        st.info("백테스팅 시뮬레이션을 시작합니다...")

        for i in range(1, len(df)):
            current_date = df.index[i]

            # --- 각 지표별 조건 설정 ---
            sma_buy_ok = True
            sma_sell_ok = True
            if use_sma:
                sma_buy_ok = (df['SMA_Short'].iloc[i-1] < df['SMA_Long'].iloc[i-1] and \
                              df['SMA_Short'].iloc[i] >= df['SMA_Long'].iloc[i])
                sma_sell_ok = (df['SMA_Short'].iloc[i-1] > df['SMA_Long'].iloc[i-1] and \
                               df['SMA_Short'].iloc[i] <= df['SMA_Long'].iloc[i])

            macd_buy_ok = False
            macd_sell_ok = False
            if use_macd:
                if df['MACD'].iloc[i-1] < df['MACD_Signal'].iloc[i-1] and \
                   df['MACD'].iloc[i] >= df['MACD_Signal'].iloc[i]:
                    macd_buy_ok = True
                if df['MACD'].iloc[i-1] > df['MACD_Signal'].iloc[i-1] and \
                   df['MACD'].iloc[i] <= df['MACD_Signal'].iloc[i]:
                    macd_sell_ok = True

            momentum_buy_ok = False
            momentum_sell_ok = False
            if use_momentum:
                if df['Momentum'].iloc[i] > momentum_buy_threshold:
                    momentum_buy_ok = True
                if df['Momentum'].iloc[i] < momentum_sell_threshold:
                    momentum_sell_ok = True

            rsi_buy_ok = False
            rsi_sell_ok = False
            if use_rsi:
                if df['RSI'].iloc[i] < rsi_buy_threshold:
                    rsi_buy_ok = True
                if df['RSI'].iloc[i] > rsi_sell_threshold:
                    rsi_sell_ok = True

            obv_buy_ok = False
            obv_sell_ok = False
            if use_obv:
                if df['OBV'].iloc[i] > df['OBV'].iloc[i-1]:
                    obv_buy_ok = True
                if df['OBV'].iloc[i] < df['OBV'].iloc[i-1]:
                    obv_sell_ok = True

            # --- 최종 매수 신호 로직 ---
            buy_signal_triggered = False
            if not in_position:
                active_secondary_buy_conditions = []
                if use_macd: active_secondary_buy_conditions.append(macd_buy_ok)
                if use_momentum: active_secondary_buy_conditions.append(momentum_buy_ok)
                if use_rsi: active_secondary_buy_conditions.append(rsi_buy_ok)
                if use_obv: active_secondary_buy_conditions.append(obv_buy_ok)

                secondary_indicators_buy_ok = (not active_secondary_buy_conditions) or any(active_secondary_buy_conditions)

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
                df.loc[current_date, 'Position'] = df['Position'].iloc[i-1]

            daily_return = (df['Adj Close'].iloc[i] / df['Adj Close'].iloc[i-1]) - 1

            if df.loc[current_date, 'Position'] == 1:
                df.loc[current_date, 'Strategy_Return'] = daily_return
            else:
                df.loc[current_date, 'Strategy_Return'] = 0.0

            df.loc[current_date, 'Cumulative_Strategy_Return'] = \
                df['Cumulative_Strategy_Return'].iloc[i-1] * (1 + df.loc[current_date, 'Strategy_Return'])
            df.loc[current_date, 'Cumulative_Buy_And_Hold_Return'] = \
                df['Cumulative_Buy_And_Hold_Return'].iloc[i-1] * (1 + daily_return)

            # --- 최대 낙폭(MDD) 계산 ---
            df.loc[current_date, 'High_Water_Mark'] = max(
                df['High_Water_Mark'].iloc[i-1], df.loc[current_date, 'Cumulative_Strategy_Return']
            )
            df.loc[current_date, 'Drawdown'] = (df.loc[current_date, 'High_Water_Mark'] - df.loc[current_date, 'Cumulative_Strategy_Return']) / df.loc[current_date, 'High_Water_Mark']

        st.info(f"백테스팅 완료! 총 매수 신호: {buy_signal_count}회, 총 매도 신호: {sell_signal_count}회.")
        st.info(f"마지막 포지션 상태: {'보유 중' if in_position else '포지션 없음'}")

        return df

    st.write("### 💸 백테스팅 결과")
    results = backtest_strategy(processed_data.copy(), use_sma, use_momentum, use_rsi, use_macd, use_obv,
                                 short_ma_period, long_ma_period,
                                 momentum_buy_threshold, momentum_sell_threshold,
                                 rsi_buy_threshold, rsi_sell_threshold)

    if results.empty:
        st.error("백테스팅 결과를 생성할 수 없습니다. 데이터 기간 및 지표 기간 설정을 다시 확인해주세요.")
        st.stop()
        
    # --- 핵심 성과 지표 계산 ---
    total_days = (results.index[-1] - results.index[0]).days
    final_strategy_return = (results['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
    final_buy_and_hold_return = (results['Cumulative_Buy_And_Hold_Return'].iloc[-1] - 1) * 100
    
    annualized_strategy_return = (1 + final_strategy_return / 100)**(365 / total_days) - 1
    max_drawdown = results['Drawdown'].max() * 100

    # --- 결과 요약 대시보드 ---
    st.subheader("📊 핵심 성과 지표")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="최종 전략 누적 수익률", value=f"{final_strategy_return:.2f}%")
    with col2:
        st.metric(label="최종 Buy & Hold 수익률", value=f"{final_buy_and_hold_return:.2f}%")
    with col3:
        st.metric(label="연평균 수익률 (전략)", value=f"{annualized_strategy_return:.2f}%")
    with col4:
        st.metric(label="최대 낙폭(MDD)", value=f"-{max_drawdown:.2f}%")

    # --- 결과 시각화 ---
    st.subheader("📈 백테스팅 시각화")

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 1, 1, 1, 2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax6 = fig.add_subplot(gs[5], sharex=ax1)

    # --- 가격 및 SMA 차트 ---
    ax1.plot(results.index, results['Adj Close'], label=f'{company_name} 가격', color='lightgray', linewidth=1)
    if use_sma:
        ax1.plot(results.index, results['SMA_Short'], label=f'단기 MA ({short_ma_period}일)', color='orange', linewidth=1.5)
        ax1.plot(results.index, results['SMA_Long'], label=f'장기 MA ({long_ma_period}일)', color='purple', linewidth=1.5)

    buy_signals = results[results['Buy_Signal'] == True]
    ax1.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', s=100, label='매수 신호', zorder=5)

    sell_signals = results[results['Sell_Signal'] == True]
    ax1.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', s=100, label='매도 신호', zorder=5)

    ax1.set_ylabel("가격(KRW)")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title(f"{company_name} 가격, 이동평균선 및 매매 신호")


    # --- 누적 수익률 차트 ---
    ax2.plot(results.index, (results['Cumulative_Strategy_Return'] - 1) * 100, label='전략 누적 수익률(%)', color='blue', linewidth=2)
    ax2.plot(results.index, (results['Cumulative_Buy_And_Hold_Return'] - 1) * 100, label='매수 후 보유(Buy & Hold) 수익률(%)', color='green', linestyle='--', linewidth=2)
    ax2.set_ylabel("누적 수익률 (%)")
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_title("누적 수익률 비교")


    # --- 지표 차트 (RSI, 모멘텀, MACD, OBV) ---
    if use_rsi or use_momentum:
        if use_rsi and use_momentum:
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
        elif use_rsi:
            ax3.plot(results.index, results['RSI'], label='RSI', color='cyan', linewidth=1)
            ax3.axhline(y=rsi_buy_threshold, color='green', linestyle='--', label=f'RSI 매수 ({rsi_buy_threshold})')
            ax3.axhline(y=rsi_sell_threshold, color='red', linestyle='--', label=f'RSI 매도 ({rsi_sell_threshold})')
            ax3.set_title("RSI 지표")
            ax3.set_ylabel("RSI")
            ax3.legend(loc='upper left')
            ax3.grid(True)
        elif use_momentum:
            ax3.plot(results.index, results['Momentum'], label='모멘텀', color='magenta', linewidth=1)
            ax3.axhline(y=momentum_buy_threshold, color='green', linestyle=':', label=f'모멘텀 매수 ({momentum_buy_threshold})')
            ax3.axhline(y=momentum_sell_threshold, color='red', linestyle=':', label=f'모멘텀 매도 ({momentum_sell_threshold})')
            ax3.set_title("모멘텀 지표")
            ax3.set_ylabel("모멘텀 (%)")
            ax3.legend(loc='upper left')
            ax3.grid(True)
    else:
        ax3.set_visible(False)

    # --- MACD 차트 ---
    if use_macd:
        ax4.plot(results.index, results['MACD'], label='MACD Line', color='blue', linewidth=1)
        ax4.plot(results.index, results['MACD_Signal'], label='Signal Line', color='red', linestyle='--', linewidth=1)
        macd_hist = results['MACD'] - results['MACD_Signal']
        colors = ['green' if x >= 0 else 'red' for x in macd_hist]
        ax4.bar(results.index, macd_hist, label='MACD 히스토그램', color=colors, alpha=0.5, width=0.8)
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

    # --- 누적 수익률과 최대 낙폭(MDD) 차트 ---
    ax6.plot(results.index, (results['Cumulative_Strategy_Return']-1) * 100, label='누적 수익률(%)', color='blue', linewidth=2)
    ax6.fill_between(results.index, (results['Cumulative_Strategy_Return']-1) * 100, (results['High_Water_Mark']-1) * 100, color='gray', alpha=0.2)
    ax6.set_ylabel("수익률 (%)")
    ax6.legend(loc='upper left')
    ax6.grid(True)
    ax6.set_title("수익률 vs. 최대 낙폭(MDD)")

    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("---")
    st.subheader("📝 최대 낙폭(MDD) 상세 정보")
    st.write("최대 낙폭은 고점 대비 최대 손실을 의미하며, 낮을수록 위험이 적습니다.")
    
    # 최대 낙폭 테이블
    if not results.empty:
        max_drawdown_value = results['Drawdown'].max()
        max_drawdown_end_date = results['Drawdown'].idxmax()
        
        # Drawdown 시작점 찾기
        high_water_mark_at_mdd_end = results.loc[max_drawdown_end_date, 'High_Water_Mark']
        max_drawdown_start_date = results[results['Cumulative_Strategy_Return'] == high_water_mark_at_mdd_end].index.max()
        
        mdd_data = {
            '기간': [f'{max_drawdown_start_date.strftime("%Y-%m-%d")} ~ {max_drawdown_end_date.strftime("%Y-%m-%d")}'],
            '최대 낙폭': [f'{-max_drawdown_value * 100:.2f}%']
        }
        mdd_df = pd.DataFrame(mdd_data)
        st.dataframe(mdd_df)
        


    st.write("---")
    st.write("### 📝 참고")
    st.write(f"""
    - **데이터 출처**: 이 앱은 **업비트(Upbit) {company_name} 일봉 데이터**를 기반으로 작동합니다.
    - **백테스팅 모델의 한계**: 제시된 수익률은 백테스팅 결과이며, 실제 투자 결과와는 다를 수 있습니다. 거래 수수료, 슬리피지(Slippage), 세금, 시스템 지연 등의 실제 거래 환경 요소를 고려하지 않은 단순 시뮬레이션입니다.
    - **면책 조항**: 본 정보는 투자 자문이 아니며, 여기에 제시된 내용은 오직 정보 제공을 목적으로 합니다. 투자 결정은 사용자 본인의 판단과 책임 하에 이루어져야 합니다.
    """)
    st.write("---")
    st.write("### 백테스팅 상세 데이터 (최근 20일)")
    display_cols = ['Adj Close', 'Buy_Signal', 'Sell_Signal', 'Position', 'Strategy_Return', 'Cumulative_Strategy_Return', 'Cumulative_Buy_And_Hold_Return', 'Drawdown']
    if use_sma: display_cols.extend(['SMA_Short', 'SMA_Long'])
    if use_rsi: display_cols.append('RSI')
    if use_momentum: display_cols.append('Momentum')
    if use_macd: display_cols.extend(['MACD', 'MACD_Signal'])
    if use_obv: display_cols.append('OBV')

    st.dataframe(results[display_cols].tail(20))
