import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

import FinanceDataReader as fdr
import pyupbit

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ======================================================
# 1. 캔들 패턴 탐지
# ======================================================
def find_candle_patterns(df):
    df = df.copy()
    df['body'] = abs(df['Close'] - df['Open'])
    df['range'] = df['High'] - df['Low']
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    df['is_hammer'] = (df['lower_shadow'] > 2 * df['body']) & (df['upper_shadow'] < df['body'])
    df['is_doji'] = (df['body'] / df['range'] < 0.05)

    df['is_bullish_engulfing'] = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1))
    )

    return df

PATTERN_COLS = [
    'is_hammer',
    'is_doji',
    'is_bullish_engulfing'
]

# ======================================================
# 2. 수치적 특징
# ======================================================
def add_numeric_features(df):
    df = df.copy()
    df['body_ratio'] = df['body'] / df['range']
    df['upper_ratio'] = df['upper_shadow'] / df['range']
    df['lower_ratio'] = df['lower_shadow'] / df['range']
    df['volatility'] = df['range'] / df['Close']
    return df

# ======================================================
# 3. 추세 & 맥락
# ======================================================
def add_context_features(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['trend'] = (df['MA20'] > df['MA60']).astype(int)

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# ======================================================
# 4. 패턴 강도
# ======================================================
def add_pattern_strength(df):
    df = df.copy()
    df['pattern_strength'] = (
        df['body_ratio'] +
        df['lower_ratio'] -
        df['upper_ratio']
    )
    return df

# ======================================================
# 5. 타겟 변수
# ======================================================
def add_targets(df):
    df = df.copy()
    df['ret_3'] = df['Close'].shift(-3) / df['Close'] - 1
    df['ret_5'] = df['Close'].shift(-5) / df['Close'] - 1
    df['ret_10'] = df['Close'].shift(-10) / df['Close'] - 1
    return df

# ======================================================
# 6. LightGBM 학습
# ======================================================
def train_lgbm(df, target):
    feature_cols = (
        PATTERN_COLS +
        ['body_ratio', 'upper_ratio', 'lower_ratio',
         'volatility', 'trend', 'RSI', 'pattern_strength']
    )

    df = df.dropna()
    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return model, rmse, X, y

# ======================================================
# 7. 패턴 승률 리포트
# ======================================================
def pattern_report(df):
    rows = []
    for p in PATTERN_COLS:
        trades = df[df[p]]
        if len(trades) > 10:
            winrate = (trades['ret_5'] > 0).mean() * 100
            rows.append({
                'Pattern': p,
                'Count': len(trades),
                'WinRate(%)': winrate
            })
    return pd.DataFrame(rows).sort_values('WinRate(%)', ascending=False)

# ======================================================
# 8. Streamlit UI
# ======================================================
st.set_page_config(layout="wide")
st.title("📊 AI 캔들 패턴 수익률 예측기 (LightGBM)")

market = st.radio("시장 선택", ["KRX 주식", "미국 주식", "코인(Upbit)"], horizontal=True)
ticker = st.text_input("티커", "AAPL")

target_map = {
    "3봉 수익률": "ret_3",
    "5봉 수익률": "ret_5",
    "10봉 수익률": "ret_10"
}
target_label = st.selectbox("예측 수익률 기준", list(target_map.keys()))
target_col = target_map[target_label]

if st.button("🚀 분석 실행"):

    with st.spinner("데이터 로딩 중..."):
        if market == "코인(Upbit)":
            df = pyupbit.get_ohlcv(ticker, count=500)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Value']
        else:
            df = fdr.DataReader(ticker)

    df = find_candle_patterns(df)
    df = add_numeric_features(df)
    df = add_context_features(df)
    df = add_pattern_strength(df)
    df = add_targets(df)

    model, rmse, X, y = train_lgbm(df, target_col)

    st.success(f"모델 학습 완료 | RMSE: {rmse:.4f}")

    # ======================================================
    # 예측
    # ======================================================
    latest_X = X.iloc[[-1]]
    pred = model.predict(latest_X)[0]

    st.metric("📈 예상 수익률", f"{pred*100:.2f}%")

    # ======================================================
    # 수익률 분포
    # ======================================================
    st.subheader("📊 수익률 분포")
    fig, ax = plt.subplots()
    ax.hist(y * 100, bins=30)
    ax.axvline(pred * 100, color='red', linestyle='--', label='Prediction')
    ax.legend()
    st.pyplot(fig)

    # ======================================================
    # 패턴 리포트
    # ======================================================
    st.subheader("🏆 패턴별 승률 리포트")
    report = pattern_report(df)
    st.dataframe(report)

    # ======================================================
    # SHAP
    # ======================================================
    st.subheader("🔍 SHAP 기여도")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[[-1]])

    shap_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP': shap_values[0]
    }).sort_values('SHAP', key=abs, ascending=False)

    st.dataframe(shap_df)


# # Streamlit을 사용한 웹 애플리케이션 제작에 필요한 라이브러리
# import streamlit as st
# import requests
# from json.decoder import JSONDecodeError

# # 주식 데이터와 그래프를 다루는 데 필요한 라이브러리들
# import FinanceDataReader as fdr
# import pyupbit
# import matplotlib.pyplot as plt
# import mplfinance as mpf
# import pandas as pd
# import datetime

# # ---------------------------------------------------------------------------------
# # 1. Streamlit 앱 설정 및 데이터 로드 함수
# # ---------------------------------------------------------------------------------
# # 캔들 패턴에 대한 한글명과 이니셜 매핑을 정의합니다.
# # 이니셜은 범례와 멀티셀렉트 옵션에 모두 사용됩니다.
# pattern_mapping = {
#     'is_hammer': {'label': '망치형 (상승)', 'initial': '[H]'},
#     'is_inverted_hammer': {'label': '역망치형 (하락)', 'initial': '[IH]'},
#     'is_doji': {'label': '도지형', 'initial': '[D]'},
#     'is_bullish_engulfing': {'label': '상승장악형', 'initial': '[BE]'},
#     'is_bearish_engulfing': {'label': '하락장악형', 'initial': '[BEE]'},
#     'is_piercing_line': {'label': '관통형', 'initial': '[PL]'},
#     'is_dark_cloud_cover': {'label': '흑운형', 'initial': '[DCC]'},
#     'is_three_white_soldiers': {'label': '적삼병', 'initial': '[TWS]'},
#     'is_three_black_crows': {'label': '흑삼병', 'initial': '[TBC]'},
#     'is_shooting_star': {'label': '유성형', 'initial': '[SS]'},
#     'is_hanging_man': {'label': '교수형', 'initial': '[HM]'}
# }

# # 매수/매도 신호 패턴 분류
# BUY_PATTERNS = ['is_hammer', 'is_bullish_engulfing', 'is_piercing_line', 'is_three_white_soldiers', 'is_inverted_hammer']
# SELL_PATTERNS = ['is_shooting_star', 'is_hanging_man', 'is_bearish_engulfing', 'is_dark_cloud_cover', 'is_three_black_crows']

# @st.cache_data
# def get_stock_listing(market):
#     """FinanceDataReader에서 주식 종목 전체 목록을 가져옵니다."""
#     try:
#         df = fdr.StockListing(market)
        
#         # 'Code' 열이 없는 경우 'Symbol' 열을 찾아 이름을 'Code'로 변경합니다.
#         if 'Code' not in df.columns and 'Symbol' in df.columns:
#             df.rename(columns={'Symbol': 'Code'}, inplace=True)
            
#         if 'Code' not in df.columns:
#             st.error("데이터에 'Code' 또는 'Symbol' 열이 없습니다. 라이브러리 버전을 확인해주세요.")
#             return pd.DataFrame()

#         df['Code'] = df['Code'].astype(str)
#         # 종목명과 티커를 결합하여 레이블 생성
#         df['label'] = df['Name'] + ' (' + df['Code'] + ')'
#         return df
#     except Exception as e:
#         st.error(f"종목 리스트를 가져오는 중 오류가 발생했습니다: {e}")
#         return pd.DataFrame()
        
# @st.cache_data
# def get_coin_listing():
#     """pyupbit에서 원화(KRW) 코인 목록을 가져오고 한글명을 매핑합니다."""
#     try:
#         # pyupbit.get_market_all() 대신 Upbit API를 직접 호출합니다.
#         url = "https://api.upbit.com/v1/market/all"
#         response = requests.get(url, params={'isDetails': 'false'})
#         response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
#         all_markets = response.json()
        
#         # 원화(KRW) 마켓만 필터링하고 데이터프레임으로 변환합니다.
#         krw_markets = [market for market in all_markets if market['market'].startswith('KRW-')]
#         df_coin = pd.DataFrame(krw_markets)
#         df_coin.rename(columns={'market': 'Code', 'korean_name': 'korean_name', 'english_name': 'english_name'}, inplace=True)
        
#         # 레이블을 '한글명 (영문티커)' 형식으로 생성
#         # 티커에서 'KRW-' 접두사를 제거합니다.
#         df_coin['label'] = df_coin['korean_name'] + ' (' + df_coin['Code'].str.replace('KRW-', '') + ')'
        
#         return df_coin
#     except requests.exceptions.RequestException as e:
#         st.error(f"❌ Upbit API 연결 오류: {e}")
#         st.info("인터넷 연결 상태를 확인하거나 Upbit 서버에 문제가 있을 수 있습니다.")
#         return pd.DataFrame()
#     except JSONDecodeError as e:
#         st.error(f"❌ Upbit API 응답 파싱 오류: {e}")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"코인 리스트를 가져오는 중 예상치 못한 오류가 발생했습니다: {e}")
#         return pd.DataFrame()

# def get_data(ticker, start_date, end_date, market, period='1D'):
#     """선택된 시장에 따라 주식 또는 코인 데이터를 가져옵니다."""
#     try:
#         if market in ['한국 주식 (KRX)', '미국 증시 (NYSE/NASDAQ)']:
#             data = fdr.DataReader(ticker, start_date, end_date)
            
#             # 주봉 또는 월봉으로 데이터를 리샘플링합니다.
#             if period == '주봉':
#                 resampled_data = data.resample('W').agg({
#                     'Open': 'first',
#                     'High': 'max',
#                     'Low': 'min',
#                     'Close': 'last',
#                     'Volume': 'sum'
#                 }).dropna()
#                 resampled_data.index.name = 'Date'
#                 return resampled_data
#             elif period == '월봉':
#                 resampled_data = data.resample('M').agg({
#                     'Open': 'first',
#                     'High': 'max',
#                     'Low': 'min',
#                     'Close': 'last',
#                     'Volume': 'sum'
#                 }).dropna()
#                 resampled_data.index.name = 'Date'
#                 return resampled_data
#             else: # 일봉
#                 data.index.name = 'Date'
#                 return data

#         elif market == '코인 (Upbit)':
#             # pyupbit는 날짜 범위가 아닌 count를 사용합니다.
#             upbit_period_map = {'일봉': 'day', '주봉': 'week', '월봉': 'month'}
#             days_diff = (end_date - start_date).days
#             count = days_diff + 1 if period == '일봉' else int(days_diff / 7) + 1 if period == '주봉' else int(days_diff / 30) + 1
            
#             df = pyupbit.get_ohlcv(ticker=ticker, interval=upbit_period_map[period], count=count)
#             if df is None or df.empty:
#                 st.warning(f"오류: [{ticker}] 코인에 대한 데이터를 찾을 수 없습니다. 티커나 날짜 범위를 확인해 주세요.")
#                 return None
                
#             df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'value']
#             df.index.name = 'Date'
#             return df
#     except Exception as e:
#         st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
#         return None

# def find_candle_patterns(df):
#     """
#     주어진 주식 데이터 DataFrame에서 캔들 패턴을 찾아 결과를 반환합니다.
#     """
#     df['is_hammer'] = False
#     df['is_inverted_hammer'] = False
#     df['is_doji'] = False
#     df['is_bullish_engulfing'] = False
#     df['is_bearish_engulfing'] = False
#     df['is_piercing_line'] = False
#     df['is_dark_cloud_cover'] = False
#     df['is_three_white_soldiers'] = False
#     df['is_three_black_crows'] = False
#     df['is_shooting_star'] = False
#     df['is_hanging_man'] = False
    
#     for i in range(len(df)):
#         # 단일 캔들 패턴
#         open_p, close_p, high_p, low_p = df.iloc[i][['Open', 'Close', 'High', 'Low']]
#         body_length = abs(close_p - open_p)
#         upper_shadow = high_p - max(open_p, close_p)
#         lower_shadow = min(open_p, close_p) - low_p
        
#         if body_length > 0 and lower_shadow > 2 * body_length and upper_shadow < body_length:
#             if close_p > open_p:
#                 df.loc[df.index[i], 'is_hammer'] = True
#             elif close_p < open_p:
#                 df.loc[df.index[i], 'is_hanging_man'] = True
        
#         if body_length > 0 and upper_shadow > 2 * body_length and lower_shadow < body_length:
#             if close_p > open_p:
#                 df.loc[df.index[i], 'is_inverted_hammer'] = True
#             elif close_p < open_p:
#                 df.loc[df.index[i], 'is_shooting_star'] = True
                
#         if body_length < (high_p - low_p) * 0.05:
#             df.loc[df.index[i], 'is_doji'] = True

#         # 이중 캔들 패턴
#         if i >= 1:
#             prev_open, prev_close, prev_high, prev_low = df.iloc[i-1][['Open', 'Close', 'High', 'Low']]
#             prev_body_midpoint = (prev_open + prev_close) / 2
            
#             if (prev_close < prev_open and close_p > open_p and open_p < prev_close and close_p > prev_open):
#                 df.loc[df.index[i], 'is_bullish_engulfing'] = True
#             if (prev_close > prev_open and close_p < open_p and open_p > prev_close and close_p < prev_open):
#                 df.loc[df.index[i], 'is_bearish_engulfing'] = True
#             if (prev_close < prev_open and close_p > open_p and open_p < prev_low and close_p > prev_body_midpoint and close_p < prev_open):
#                 df.loc[df.index[i], 'is_piercing_line'] = True
#             if (prev_close > prev_open and close_p < open_p and open_p > prev_high and close_p < prev_body_midpoint and close_p > prev_open):
#                 df.loc[df.index[i], 'is_dark_cloud_cover'] = True

#         # 삼중 캔들 패턴
#         if i >= 2:
#             prev2_open, prev2_close = df.iloc[i-2][['Open', 'Close']]
#             prev1_open, prev1_close = df.iloc[i-1][['Open', 'Close']]
#             curr_open, curr_close = df.iloc[i][['Open', 'Close']]
            
#             if (prev2_close > prev2_open and prev1_close > prev1_open and curr_close > curr_open and
#                 prev1_close > prev2_close and curr_close > prev1_close and
#                 prev1_open >= prev2_close and curr_open >= prev1_close):
#                 df.loc[df.index[i], 'is_three_white_soldiers'] = True
                
#             if (prev2_close < prev2_open and prev1_close < prev1_open and curr_close < curr_open and
#                 prev1_close < prev2_close and curr_close < prev1_close and
#                 prev1_open <= prev2_close and curr_open <= prev1_close):
#                 df.loc[df.index[i], 'is_three_black_crows'] = True
    
#     return df

# def calculate_and_add_indicators(df, show_ma, show_bb, show_rsi):
#     """선택된 기술적 지표들을 계산하고, mplfinance addplot 객체 리스트를 반환합니다."""
#     apds = []
    
#     # 이동평균선 (20일, 60일) 계산 및 추가
#     if show_ma:
#         df['MA20'] = df['Close'].rolling(window=20).mean()
#         df['MA60'] = df['Close'].rolling(window=60).mean()
#         apds.append(mpf.make_addplot(df['MA20'], color='blue', panel=0, label='단기 MA (20일)'))
#         apds.append(mpf.make_addplot(df['MA60'], color='red', panel=0, label='장기 MA (60일)'))
        
#     # 볼린저 밴드 계산 및 추가
#     if show_bb:
#         df['MA20'] = df['Close'].rolling(window=20).mean()
#         df['STD20'] = df['Close'].rolling(window=20).std()
#         df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
#         df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
        
#         apds.append(mpf.make_addplot(df['BB_Upper'], color='purple', linestyle=':', panel=0, label='BB_Upper'))
#         apds.append(mpf.make_addplot(df['BB_Lower'], color='#FFB6C1', linestyle=':', panel=0, label='BB_Lower'))

#     # RSI (상대강도지수) 계산 및 추가 (14일 기준)
#     if show_rsi:
#         delta = df['Close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
#         # 0으로 나누기 방지
#         rs = gain / (loss.replace(0, 1e-10))
#         df['RSI'] = 100 - (100 / (1 + rs))
        
#         # 새로운 패널에 RSI 그래프 추가
#         apds.append(mpf.make_addplot(df['RSI'], panel=2, color='orange', ylabel='RSI', label='RSI'))
        
#         # RSI 30, 70 라인 추가
#         apds.append(mpf.make_addplot([70] * len(df), panel=2, color='red', linestyle='--', width=1))
#         apds.append(mpf.make_addplot([30] * len(df), panel=2, color='green', linestyle='--', width=1))
        
#     return apds

# def run_backtest(df_with_patterns, selected_pattern_cols):
#     """
#     주어진 패턴 데이터프레임을 사용하여 백테스트를 실행하는 함수입니다.
#     `_Timestamp.__add__` 오류를 해결하기 위해 루프를 정수 인덱스 기반으로 수정했습니다.

#     Args:
#         df_with_patterns (pd.DataFrame): 캔들스틱 패턴이 식별된 데이터프레임.
#         selected_pattern_cols (list): 백테스트에 사용할 패턴 컬럼명 리스트.

#     Returns:
#         tuple: 백테스트 결과, 개별 거래 수익률, 백테스트용 데이터프레임.
#     """
#     df_bt = df_with_patterns.copy()
    
#     # 신호 컬럼 생성 (선택된 패턴 중 하나라도 True이면 1, 아니면 0)
#     df_bt['Signal'] = df_bt[selected_pattern_cols].any(axis=1).astype(int)
    
#     trades = []
#     in_trade = False
#     buy_price = 0
    
#     # 수정된 루프: 정수 인덱스 `i`를 사용합니다.
#     for i in range(len(df_bt)):
#         row = df_bt.iloc[i]
        
#         # 진입 신호 확인
#         if row['Signal'] == 1 and not in_trade:
#             buy_price = row['Close']
#             in_trade = True
            
#         # 청산 신호 확인 (다음 날의 시가로 청산)
#         # 데이터프레임의 끝에 도달했는지 확인합니다.
#         elif in_trade:
#             if i + 1 < len(df_bt):
#                 exit_price = df_bt.iloc[i + 1]['Open']
#             else:
#                 exit_price = row['Close'] # 마지막 행이면 현재 종가로 청산

#             trade_return = (exit_price - buy_price) / buy_price * 100
#             trades.append(trade_return)
#             in_trade = False

#     trade_returns = pd.Series(trades)
    
#     # 백테스트 결과 계산
#     total_return = (1 + trade_returns / 100).prod() - 1 if not trade_returns.empty else 0
#     num_trades = len(trades)
#     winning_rate = (trade_returns > 0).sum() / num_trades if num_trades > 0 else 0
    
#     # 결과를 담을 DataFrame 생성
#     backtest_results_df = pd.DataFrame([{
#         'Total Return (%)': total_return * 100,
#         'Number of Trades': num_trades,
#         'Winning Rate (%)': winning_rate * 100,
#         'Average Return per Trade (%)': trade_returns.mean() if num_trades > 0 else 0
#     }])
    
#     return backtest_results_df, trade_returns, df_bt
    
# # ---------------------------------------------------------------------------------
# # 2. Streamlit 웹 인터페이스 구성
# # ---------------------------------------------------------------------------------
# st.set_page_config(page_title="주식 & 코인 캔들 패턴 분석기", layout="wide")

# st.markdown("<h1 style='text-align: center;'>주식 & 코인 캔들 패턴 분석기</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: #4CAF50;'>원하는 시장과 종목, 날짜 범위를 선택하여 차트를 분석하세요.</h3>", unsafe_allow_html=True)

# st.subheader("1. 분석 옵션 선택")
# selected_market = st.radio(
#     "💰 분석할 시장을 선택하세요",
#     ('한국 주식 (KRX)', '미국 증시 (NYSE/NASDAQ)', '코인 (Upbit)'),
#     horizontal=True
# )

# df_listing = pd.DataFrame()
# default_start_date = datetime.date.today()
# period_options = ('일봉', '주봉', '월봉')

# if selected_market == '한국 주식 (KRX)':
#     df_listing = get_stock_listing('KRX')
#     default_start_date = datetime.date.today() - datetime.timedelta(days=365)
# elif selected_market == '미국 증시 (NYSE/NASDAQ)':
#     df_listing = get_stock_listing('NASDAQ')
#     default_start_date = datetime.date.today() - datetime.timedelta(days=365)
# else: # 코인 (Upbit)
#     df_listing = get_coin_listing()
#     default_start_date = datetime.date.today() - datetime.timedelta(days=180)
#     period_options = ('일봉', '주봉', '월봉')
    
# if not df_listing.empty:
#     selected_label = st.selectbox(f"📊 분석할 {selected_market.split()[0]} 종목", df_listing["label"].tolist())
#     # 'Code' 열이 항상 존재하도록 수정했기 때문에 아래 코드는 이제 안전합니다.
#     selected_code = df_listing[df_listing["label"] == selected_label]["Code"].values[0]

#     col1, col2 = st.columns(2)
#     with col1:
#         selected_period = st.radio(
#             "⏳ 차트 기간",
#             period_options,
#             horizontal=True
#         )
#     with col2:
#         # 패턴 옵션에 이니셜 추가
#         all_pattern_options = {
#             '망치형 (상승) [H]': 'is_hammer',
#             '역망치형 (하락) [IH]': 'is_inverted_hammer',
#             '도지형 [D]': 'is_doji',
#             '상승장악형 [BE]': 'is_bullish_engulfing',
#             '하락장악형 [BEE]': 'is_bearish_engulfing',
#             '관통형 [PL]': 'is_piercing_line',
#             '흑운형 [DCC]': 'is_dark_cloud_cover',
#             '적삼병 [TWS]': 'is_three_white_soldiers',
#             '흑삼병 [TBC]': 'is_three_black_crows',
#             '유성형 [SS]': 'is_shooting_star',
#             '교수형 [HM]': 'is_hanging_man'
#         }
#         selected_patterns = st.multiselect(
#             "📈 표시할 캔들 패턴",
#             list(all_pattern_options.keys())
#         )
    
#     st.subheader("2. 기술적 지표 선택")
#     col3, col4 = st.columns(2)
#     with col3:
#         show_ma = st.checkbox('이동평균선 (20일, 60일)')
#         show_bb = st.checkbox('볼린저 밴드')
#     with col4:
#         show_rsi = st.checkbox('상대강도지수 (RSI)')

#     st.subheader("3. 날짜 범위 선택")
#     today = datetime.date.today()
#     col5, col6 = st.columns(2)
#     with col5:
#         start_date = st.date_input("시작 날짜", default_start_date)
#     with col6:
#         end_date = st.date_input("종료 날짜", today)

#     st.markdown("---")
#     if st.button("차트 분석 시작", type="primary", use_container_width=True):
#         st.subheader("분석 중...")
#         st.info("데이터를 불러오고 캔들 패턴을 분석하는 중입니다. 잠시만 기다려 주세요.")
        
#         df = get_data(selected_code, start_date, end_date, selected_market, selected_period)
        
#         if df is not None and not df.empty:
#             df_with_patterns = find_candle_patterns(df.copy())
#             apds = []
            
#             marker_size = 50
            
#             # 차트 시각화에 사용할 패턴 정보 (컬럼명, 위치, 마커, 색상)
#             chart_pattern_info = {
#                 'is_hammer': ('Low', '^', 'red'),
#                 'is_inverted_hammer': ('High', 'v', 'blue'),
#                 'is_doji': ('Close', '*', 'orange'),
#                 'is_bullish_engulfing': ('Low', 'o', 'green'),
#                 'is_bearish_engulfing': ('High', 'x', 'purple'),
#                 'is_piercing_line': ('Low', 'D', 'darkgreen'),
#                 'is_dark_cloud_cover': ('High', 'D', 'darkred'),
#                 'is_three_white_soldiers': ('Low', 'D', 'darkgreen'),
#                 'is_three_black_crows': ('High', 'D', 'darkred'),
#                 'is_shooting_star': ('High', 'v', 'magenta'),
#                 'is_hanging_man': ('Low', 's', 'brown')
#             }
            
#             total_patterns = 0
#             st.subheader("4. 발견된 패턴 목록")
#             pattern_results = {}
            
#             # 멀티셀렉트에서 선택된 라벨을 원래 컬럼명으로 변환
#             selected_pattern_cols = [all_pattern_options[label] for label in selected_patterns]
            
#             for pattern_col_name in selected_pattern_cols:
#                 # 범례에 이니셜만 표시하도록 변경
#                 pattern_initials = pattern_mapping[pattern_col_name]['initial']
                
#                 # 적삼병과 흑삼병은 선으로 표시
#                 if pattern_col_name == 'is_three_white_soldiers':
#                     series = pd.Series(index=df.index, dtype='float64')
#                     for i in range(2, len(df_with_patterns)):
#                         if df_with_patterns.loc[df_with_patterns.index[i], pattern_col_name]:
#                             min_low = min(df.iloc[i-2:i+1]['Low'])
#                             series.iloc[i-2:i+1] = min_low * 0.99
#                             pattern_results[pattern_initials] = pattern_results.get(pattern_initials, 0) + 1
#                             total_patterns += 1
#                     if not series.dropna().empty:
#                         apds.append(mpf.make_addplot(series,
#                                                      type='line', linestyle='solid', width=5, color='red', label=pattern_initials))
                
#                 elif pattern_col_name == 'is_three_black_crows':
#                     series = pd.Series(index=df.index, dtype='float64')
#                     for i in range(2, len(df_with_patterns)):
#                         if df_with_patterns.loc[df_with_patterns.index[i], pattern_col_name]:
#                             max_high = max(df.iloc[i-2:i+1]['High'])
#                             series.iloc[i-2:i+1] = max_high * 1.01
#                             pattern_results[pattern_initials] = pattern_results.get(pattern_initials, 0) + 1
#                             total_patterns += 1
#                     if not series.dropna().empty:
#                         apds.append(mpf.make_addplot(series,
#                                                      type='line', linestyle='solid', width=5, color='blue', label=pattern_initials))
#                 else:
#                     # 마커로 표시되는 패턴들
#                     y_pos, marker, color = chart_pattern_info[pattern_col_name]
#                     candles = df_with_patterns[df_with_patterns[pattern_col_name]]
#                     if not candles.empty:
#                         pattern_data = pd.Series(index=df.index, dtype='float64')
#                         for idx in candles.index:
#                             pattern_data.loc[idx] = candles.loc[idx, y_pos]
                        
#                         apds.append(mpf.make_addplot(pattern_data,
#                                                      type='scatter',
#                                                      markersize=marker_size,
#                                                      marker=marker,
#                                                      color=color,
#                                                      label=pattern_initials))
#                         count = len(candles)
#                         pattern_results[pattern_initials] = count
#                         total_patterns += count

#             if total_patterns > 0:
#                 for initials, count in pattern_results.items():
#                     # 한글명과 함께 표시
#                     full_label = next(info['label'] for info in pattern_mapping.values() if info['initial'] == initials)
#                     st.write(f"- **{full_label} {initials}**: {count}개 발견")
#             else:
#                 st.write("선택한 기간 동안 발견된 캔들 패턴이 없습니다.")

#             # 선택된 지표들을 추가
#             indicator_apds = calculate_and_add_indicators(df_with_patterns, show_ma, show_bb, show_rsi)
#             apds.extend(indicator_apds)
            
#             st.subheader("5. 캔들 차트")
#             mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
#             s = mpf.make_mpf_style(marketcolors=mc, gridcolor='gray')
            
#             title = f'{selected_label} {selected_period} 차트'
            
#             # RSI 지표가 선택되었을 경우에만 패널 비율 조정
#             panel_ratios = (6, 1.5, 2) if show_rsi else (6, 1.5)
            
#             try:
#                 fig, axlist = mpf.plot(
#                     df_with_patterns,
#                     type='candle',
#                     style=s,
#                     title=title,
#                     ylabel='가격',
#                     volume=True,
#                     figratio=(15, 10),
#                     addplot=apds,
#                     returnfig=True,
#                     panel_ratios=panel_ratios
#                 )
#                 st.pyplot(fig)
#             except Exception as e:
#                 st.error(f"차트 시각화 중 오류가 발생했습니다: {e}")
            
#             # ---------------------------------------------------------------------------------
#             # 6. 백테스팅 결과
#             # ---------------------------------------------------------------------------------
#             st.markdown("---")
#             st.subheader("6. 백테스팅 결과")
            
#             # 백테스팅 실행
#             backtest_results, trade_returns, df_bt = run_backtest(df_with_patterns, selected_pattern_cols)

#             if not backtest_results.empty:
#                 st.write(f"총 거래 횟수: **{int(backtest_results.loc[0, 'Number of Trades'])}회**")
#                 st.write(f"최종 누적 수익률: **{backtest_results.loc[0, 'Total Return (%)']:.2f}%**")
#                 st.write(f"승률: **{backtest_results.loc[0, 'Winning Rate (%)']:.2f}%**")
                
#                 # 누적 수익률 시각화
#                 if len(trade_returns) > 0:
#                     # 누적 수익률 계산 (백분율)
#                     cumulative_returns = (1 + trade_returns / 100).cumprod() - 1
#                     cumulative_returns = cumulative_returns * 100
                    
#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     ax.plot(cumulative_returns.index + 1, cumulative_returns.values, marker='o', linestyle='-')
#                     ax.axhline(0, color='gray', linestyle='--')
#                     ax.set_title('Cumulative Return of Candle Pattern Strategy', fontsize=15)
#                     ax.set_xlabel('Number of Trades', fontsize=12)
#                     ax.set_ylabel('Cumulative Return (%)', fontsize=12)
#                     ax.grid(True)
#                     st.pyplot(fig)
#                 else:
#                     st.warning("선택한 패턴으로 백테스팅할 거래 신호가 없습니다.")
#             else:
#                 st.warning("백테스팅 결과를 생성할 수 없습니다. 선택한 패턴이 매수 또는 매도 신호를 생성하지 않았을 수 있습니다.")

#         else:
#             st.error("데이터를 가져오는 데 실패했습니다. 종목 코드나 날짜 범위를 다시 확인해 주세요.")

# # --- 캔들 패턴 설명 (st.expander와 st.markdown 사용) ---
# # 날짜 범위 선택 바로 위에 위치
# with st.expander("캔들 패턴 참고자료 📖"):
#     st.markdown("""
#     이 앱에서 분석하는 주요 캔들 패턴에 대한 간단한 설명입니다.

#     - **🔎도지(Doji)**: 시가와 종가가 거의 같은 십자형 캔들입니다. 매수자와 매도자가 서로 힘의 균형을 이루고 있다는 것을 나타내며, 추세 전환의 신호일 수 있습니다.
#     - **🔎망치형 (Hammer)**: 긴 아래 꼬리와 짧은 몸통을 가진 캔들입니다. 하락 추세에서 나타나면 바닥을 확인하고 반등할 가능성을 시사합니다.
#     - **장악형 (Engulfing)**: 현재 캔들이 이전 캔들의 몸통을 완전히 감싸는 형태입니다.
#         - **📈상승 장악형 (Bullish Engulfing)**: 큰 양봉이 이전 음봉을 감싸는 형태로, 강한 매수세와 상승 반전을 예고합니다.
#         - **📉하락 장악형 (Bearish Engulfing)**: 큰 음봉이 이전 양봉을 감싸는 형태로, 강한 매도세와 하락 반전을 예고합니다.
#     - **📈모닝 스타 (Morning Star)**: 하락 추세에서 나타나는 3개의 캔들 패턴입니다. 큰 음봉, 작은 캔들, 그리고 큰 양봉이 순서대로 나타나며, 강력한 상승 반전 신호입니다.
#     - **샛별형 (Star)**: 몸통이 이전 캔들의 몸통 위에 위치하는 캔들입니다.
#         - **📈상승 샛별형 (Bullish Star)**: 큰 음봉 이후 작은 캔들이 나타나며, 상승 반전 가능성을 시사합니다.
#         - **📉하락 샛별형 (Bearish Star)**: 큰 양봉 이후 작은 캔들이 나타나며, 하락 반전 가능성을 시사합니다.
#     - **🔎십자 샛별형 (Doji Star)**: 샛별형의 작은 캔들이 도지 형태인 경우입니다. 추세 전환의 신호로 더 강력하게 해석됩니다.
#     - **📈관통형 (Piercing Line)**: 하락 추세에서 첫 날 큰 음봉이 나타나고, 다음 날 양봉이 나타나는데, 이 양봉의 종가가 이전 날 음봉의 중간 지점을 뚫고 올라가는 형태입니다. 하락 추세가 끝날 수 있다는 긍정적인 신호로 해석됩니다.
#     - **📉흑운형 (Dark Cloud Cover)**: 상승 추세에서 첫 날 양봉이 나타나고, 다음 날 음봉이 나타나는데, 이 음봉의 종가가 이전 날 양봉의 중간 지점을 뚫고 내려오는 형태입니다. 매도세가 강해져 상승 추세가 꺾일 수 있다는 부정적인 신호입니다.
#     - **📈적삼병 (Three White Soldiers)**: 3일 연속 양봉이 나타나는 패턴입니다. 각 양봉의 종가가 이전 날의 종가보다 높게 끝나며, 강력한 상승 추세의 시작을 알리는 신호입니다.
#     - **📉흑삼병 (Three Black Crows)**: 3일 연속 음봉이 나타나는 패턴입니다. 각 음봉의 종가가 이전 날의 종가보다 낮게 끝나며, 강력한 하락 추세의 시작을 알리는 신호입니다.
#     - **📉유성형 (Shooting Star)**: 긴 위 꼬리와 짧은 몸통을 가진 캔들입니다. 상승 추세에서 나타나면 고점에서 매수세가 약해졌다는 것을 보여주며, 하락 반전 가능성을 시사합니다.
#     - **📉교수형 (Hanging Man)**: 망치형과 모양은 비슷하지만, 상승 추세에서 나타납니다. 주가가 고점에서 하락할 가능성이 있다는 경고 신호로 해석됩니다.
#     """)
