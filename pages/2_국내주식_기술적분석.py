# streamlit_test/pages/2_기술적_분석_전략.py
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os # os 모듈 추가

# ---
## Streamlit 페이지 설정
# st.set_page_config()는 반드시 파일의 가장 첫 번째 Streamlit 명령이어야 합니다.
# 모든 import 문 다음, 다른 함수 정의나 Streamlit UI 코드보다 위에 배치합니다.
st.set_page_config(layout="wide")

st.title("📌 기술적 분석 기반 전략 백테스팅")
st.markdown("골든 크로스/데드 크로스, RSI, 볼린저 밴드 등 다양한 기술적 지표를 활용한 전략의 과거 수익률을 측정합니다.")

# ---
## 기술적 지표 계산 함수
@st.cache_data # 데이터 프레임 계산 결과 캐싱 (성능 향상)
def calculate_bollinger_bands(df, window=20, num_std=2):
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['Std'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['MA'] + num_std * df['Std']
    df['Lower'] = df['MA'] - num_std * df['Std']
    return df

@st.cache_data
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # avg_loss가 0이 되는 경우를 처리 (0으로 나누는 오류 방지)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

@st.cache_data
def calculate_moving_average(df, short_window, long_window):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

# ---
## 매매 신호 생성 함수
@st.cache_data
def generate_golden_cross_signals(df):
    df['Signal_GC'] = 0
    # 골든 크로스: 단기 MA가 장기 MA를 상향 돌파 (이전에는 낮았고 현재는 높거나 같음)
    df['Buy_GC'] = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
    # 데드 크로스: 단기 MA가 장기 MA를 하향 돌파 (이전에는 높았고 현재는 낮거나 같음)
    df['Sell_GC'] = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))
    df.loc[df['Buy_GC'], 'Signal_GC'] = 1
    df.loc[df['Sell_GC'], 'Signal_GC'] = -1
    return df

@st.cache_data
def generate_rsi_signals(df, buy_threshold=30, sell_threshold=70):
    df['Signal_RSI'] = 0
    # RSI 매수: RSI가 buy_threshold를 상향 돌파할 때 (과매도 구간 탈출)
    df['Buy_RSI'] = (df['RSI'] > buy_threshold) & (df['RSI'].shift(1) <= buy_threshold)
    # RSI 매도: RSI가 sell_threshold를 하향 돌파할 때 (과매수 구간 하락)
    df['Sell_RSI'] = (df['RSI'] < sell_threshold) & (df['RSI'].shift(1) >= sell_threshold)
    df.loc[df['Buy_RSI'], 'Signal_RSI'] = 1
    df.loc[df['Sell_RSI'], 'Signal_RSI'] = -1
    return df

@st.cache_data
def generate_bollinger_signals(df):
    df['Signal_BB'] = 0
    # 볼린저 밴드 매수: 종가가 하한선 아래로 내려갔다가 다시 올라올 때 (하한선 돌파 후 회복)
    df['Buy_BB'] = (df['Close'] > df['Lower']) & (df['Close'].shift(1) <= df['Lower'].shift(1))
    # 볼린저 밴드 매도: 종가가 상한선 위로 올라갔다가 다시 내려올 때 (상한선 돌파 후 회귀)
    df['Sell_BB'] = (df['Close'] < df['Upper']) & (df['Close'].shift(1) >= df['Upper'].shift(1))
    df.loc[df['Buy_BB'], 'Signal_BB'] = 1
    df.loc[df['Sell_BB'], 'Signal_BB'] = -1
    return df

# ---
## 백테스팅 함수
def backtest(df, signal_column):
    initial_balance = 1000000
    balance = initial_balance
    holdings = 0
    transactions = []

    # 신호가 있는 데이터만 사용
    df_cleaned = df.dropna(subset=[signal_column])

    if df_cleaned.empty:
        return 0, pd.DataFrame(columns=['Date', 'Action', 'Price', 'Qty', 'Balance'])

    # 매수/매도 로직
    for i in range(1, len(df_cleaned)):
        current_date = df_cleaned.index[i]
        price = df_cleaned['Close'].iloc[i]

        # 매수 신호 (현재 주식 보유X, 잔고 충분)
        if df_cleaned[signal_column].iloc[i] == 1 and holdings == 0 and balance > 0:
            if price > 0:
                qty = balance // price
                if qty > 0:
                    holdings += qty
                    balance -= qty * price
                    transactions.append({'Date': current_date, 'Action': 'Buy', 'Price': price, 'Qty': qty, 'Balance': balance})

        # 매도 신호 (현재 주식 보유O)
        elif df_cleaned[signal_column].iloc[i] == -1 and holdings > 0:
            balance += holdings * price
            transactions.append({'Date': current_date, 'Action': 'Sell', 'Price': price, 'Qty': holdings, 'Balance': balance})
            holdings = 0

    # 최종 자산 가치 계산 (남은 현금 + 보유 중인 주식 가치)
    final_value = balance + holdings * df_cleaned['Close'].iloc[-1] if not df_cleaned.empty else initial_balance
    return_rate = (final_value - initial_balance) / initial_balance * 100
    return return_rate, pd.DataFrame(transactions)

# ---
## 종목 리스트 불러오기 함수
# company_list.csv를 로드하는 함수를 정의하고 @st.cache_data를 적용합니다.
@st.cache_data
def get_company_list():
    current_dir = os.path.dirname(__file__)
    root_dir = os.path.join(current_dir, '..')
    company_list_file_path = os.path.join(root_dir, 'company_list.csv')

    try:
        df_company = pd.read_csv(company_list_file_path)
        # 종목 코드 포맷팅이 필요하다면 아래 주석 해제
        # df_company['Code'] = df_company['Code'].astype(str).str.zfill(6)
        df_company['label'] = df_company['Name'] + ' (' + df_company['Code'] + ')'
        st.success(f"✅ 데이터 수집에 성공했습니다. 기간·분석 전략을 선택해주세요.")
        return df_company
    except FileNotFoundError:
        st.error(f"❌ 데이터 수집에 오류가 발생했습니다. 빠르게 수정하겠습니다.")
        return pd.DataFrame() # 빈 데이터프레임을 반환하여 이후 오류 방지
    except Exception as e:
        st.error(f"종목 리스트 처리 중 오류가 발생했습니다: {e}")
        return pd.DataFrame() # 빈 데이터프레임을 반환하여 이후 오류 방지

# ---
## Streamlit UI 로직
# get_company_list() 함수를 호출하여 종목 데이터를 가져옵니다.
company_df = get_company_list()

# 종목 리스트가 성공적으로 로드되었을 경우에만 UI를 그립니다.
if not company_df.empty:
    selected_label = st.selectbox("📊 분석할 종목을 선택하세요", company_df["label"].tolist())
    selected_code = company_df[company_df["label"] == selected_label]["Code"].values[0]

    # 날짜 선택
    min_date_fdr = datetime.today() - timedelta(days=365 * 10) # FDataReader는 더 긴 기간 데이터 가능
    max_date_fdr = datetime.today()
    col_date_tech1, col_date_tech2 = st.columns(2)
    with col_date_tech1:
        start_date = st.date_input("시작일", min_value=min_date_fdr, max_value=max_date_fdr, value=min_date_fdr)
    with col_date_tech2:
        end_date = st.date_input("종료일", min_value=start_date, max_value=max_date_fdr, value=max_date_fdr)

    if start_date >= end_date:
        st.error("종료 날짜는 시작 날짜보다 미래여야 합니다.")
        st.stop()

    # 전략 선택 체크박스
    st.write("### ⚙️ 백테스팅할 기술적 분석 전략을 선택하세요:")
    run_gc_backtest = st.checkbox("골든크로스/데드크로스 전략 (단기 20일, 장기 60일)", value=True)
    run_rsi_backtest = st.checkbox("RSI 전략 (매수 30, 매도 70)", value=True)
    run_bb_backtest = st.checkbox("볼린저 밴드 전략 (20일, 2표준편차)", value=True)

    if st.button("🚀 전략 백테스팅 시작"):
        # FDataReader로 주식 데이터 로드
        with st.spinner(f"🔄 {selected_label}의 데이터를 불러오는 중..."):
            # fdr.DataReader는 start와 end 인자를 datetime 객체로도 받을 수 있습니다.
            df = fdr.DataReader(selected_code, start=start_date, end=end_date)

        if df.empty or len(df) < 60: # 최소 60일 (장기 MA 기간) 데이터 필요
            st.warning("데이터가 부족합니다. 선택된 기간이나 종목을 확인해주세요. 최소 60일 이상 데이터가 필요합니다.")
        else:
            st.subheader(f"📈 {selected_label} ({selected_code}) 주가 차트")
            st.line_chart(df['Close'])
            st.metric("📊 단순 매수 후 보유 (Buy & Hold) 수익률", f"{((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100:.2f}%")

            # ---
            ### 골든크로스 전략 백테스팅
            if run_gc_backtest:
                st.markdown("---")
                st.subheader("💰 골든크로스/데드크로스 전략 백테스팅 결과")
                df_gc = calculate_moving_average(df.copy(), 20, 60)
                df_gc = generate_golden_cross_signals(df_gc)
                r_gc, log_gc = backtest(df_gc, 'Signal_GC')
                st.metric("수익률", f"{r_gc:.2f}%")

                fig_gc, ax_gc = plt.subplots(figsize=(12, 6))
                ax_gc.plot(df_gc.index, df_gc['Close'], label='Close', color='lightgray', linewidth=1)
                ax_gc.plot(df_gc.index, df_gc['Short_MA'], label='short MA (20)', color='orange', linewidth=1.5)
                ax_gc.plot(df_gc.index, df_gc['Long_MA'], label='long MA (60)', color='purple', linewidth=1.5)

                # 매수/매도 신호 시각화
                buy_signals_gc = df_gc[df_gc['Buy_GC'] == True]
                sell_signals_gc = df_gc[df_gc['Sell_GC'] == True]
                ax_gc.scatter(buy_signals_gc.index, buy_signals_gc['Close'], marker='^', color='green', s=100, label='buy(Golden Cross)', zorder=5)
                ax_gc.scatter(sell_signals_gc.index, sell_signals_gc['Close'], marker='v', color='red', s=100, label='sell(Dead Cross)', zorder=5)

                ax_gc.set_title("Golden Cross / Dead Cross Strategy")
                ax_gc.legend()
                ax_gc.grid(True)
                st.pyplot(fig_gc)

                if not log_gc.empty:
                    with st.expander("매매 기록 보기"):
                        st.dataframe(log_gc)
                else:
                    st.info("해당 기간 동안 골든크로스/데드크로스 매매 신호가 발생하지 않았습니다.")

            # ---
            ### RSI 전략 백테스팅
            if run_rsi_backtest:
                st.markdown("---")
                st.subheader("💰 RSI 전략 백테스팅 결과")
                df_rsi = calculate_rsi(df.copy())
                df_rsi = generate_rsi_signals(df_rsi)
                r_rsi, log_rsi = backtest(df_rsi, 'Signal_RSI')
                st.metric("수익률", f"{r_rsi:.2f}%")

                fig_rsi, ax_rsi = plt.subplots(figsize=(12, 6))
                ax_rsi.plot(df_rsi.index, df_rsi['Close'], label='Close', color='lightgray', linewidth=1)
                ax_rsi.plot(df_rsi.index, df_rsi['RSI'], label='RSI', color='blue', linewidth=1.5)
                ax_rsi.axhline(70, color='red', linestyle='--', label='overbought(70)')
                ax_rsi.axhline(30, color='green', linestyle='--', label='oversold(30)')

                # 매수/매도 신호 시각화
                buy_signals_rsi = df_rsi[df_rsi['Buy_RSI'] == True]
                sell_signals_rsi = df_rsi[df_rsi['Sell_RSI'] == True]
                ax_rsi.scatter(buy_signals_rsi.index, buy_signals_rsi['Close'], marker='^', color='green', s=100, label='buy(RSI)', zorder=5)
                ax_rsi.scatter(sell_signals_rsi.index, sell_signals_rsi['Close'], marker='v', color='red', s=100, label='sell(RSI)', zorder=5)

                ax_rsi.set_title("RSI Strategy")
                ax_rsi.legend()
                ax_rsi.grid(True)
                st.pyplot(fig_rsi)

                if not log_rsi.empty:
                    with st.expander("매매 기록 보기"):
                        st.dataframe(log_rsi)
                else:
                    st.info("해당 기간 동안 RSI 매매 신호가 발생하지 않았습니다.")

            # ---
            ### 볼린저 밴드 전략 백테스팅
            if run_bb_backtest:
                st.markdown("---")
                st.subheader("💰 볼린저 밴드 전략 백테스팅 결과")
                df_bb = calculate_bollinger_bands(df.copy())
                df_bb = generate_bollinger_signals(df_bb)
                r_bb, log_bb = backtest(df_bb, 'Signal_BB')
                st.metric("수익률", f"{r_bb:.2f}%")

                fig_bb, ax_bb = plt.subplots(figsize=(12, 6))
                ax_bb.plot(df_bb.index, df_bb['Close'], label='Close', color='lightgray', linewidth=1)
                ax_bb.plot(df_bb.index, df_bb['Upper'], label='upper limit', color='red', linestyle='--')
                ax_bb.plot(df_bb.index, df_bb['MA'], label='middle line', color='blue')
                ax_bb.plot(df_bb.index, df_bb['Lower'], label='lower limit', color='green', linestyle='--')

                # 매수/매도 신호 시각화
                buy_signals_bb = df_bb[df_bb['Buy_BB'] == True]
                sell_signals_bb = df_bb[df_bb['Sell_BB'] == True]
                ax_bb.scatter(buy_signals_bb.index, buy_signals_bb['Close'], marker='^', color='green', s=100, label='Buy(BB)', zorder=5)
                ax_bb.scatter(sell_signals_bb.index, sell_signals_bb['Close'], marker='v', color='red', s=100, label='Sell(BB)', zorder=5)

                ax_bb.set_title("Bollinger band strategy")
                ax_bb.legend()
                ax_bb.grid(True)
                st.pyplot(fig_bb)

                if not log_bb.empty:
                    with st.expander("매매 기록 보기"):
                        st.dataframe(log_bb)
                else:
                    st.info("해당 기간 동안 볼린저 밴드 매매 신호가 발생하지 않았습니다.")

else: # company_df가 비어있을 때 (즉, get_company_list에서 오류가 났을 때)
    st.info("종목 리스트를 불러오는 데 문제가 발생했습니다. 페이지 상단의 오류 메시지를 확인해주세요.")


st.markdown("---")
st.write("### 참고")
st.write("""
- **기술적 지표:** 주가 차트에서 패턴이나 추세를 파악하여 매매 시점을 결정하는 분석 방법입니다.
- **백테스팅 모델의 한계:** 거래 수수료, 슬리피지 등을 고려하지 않은 단순 시뮬레이션입니다.
""")


# # streamlit_test/pages/2_기술적_분석_전략.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import FinanceDataReader as fdr # fdr 라이브러리 필요
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta # datetime 모듈에서 timedelta도 import

# st.set_page_config(layout="wide")
# st.title("📌 기술적 분석 기반 전략 백테스팅")
# st.markdown("골든 크로스/데드 크로스, RSI, 볼린저 밴드 등 다양한 기술적 지표를 활용한 전략의 과거 수익률을 측정합니다.")

# # --------------------------------------------
# # 함수 정의 (이전 코드에서 가져와서 여기에 배치)
# @st.cache_data # 데이터 프레임 계산 결과 캐싱 (성능 향상)
# def calculate_bollinger_bands(df, window=20, num_std=2):
#     df['MA'] = df['Close'].rolling(window=window).mean()
#     df['Std'] = df['Close'].rolling(window=window).std()
#     df['Upper'] = df['MA'] + num_std * df['Std']
#     df['Lower'] = df['MA'] - num_std * df['Std']
#     return df

# @st.cache_data
# def calculate_rsi(df, period=14):
#     delta = df['Close'].diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#     avg_gain = gain.rolling(window=period).mean()
#     avg_loss = loss.rolling(window=period).mean()
    
#     # avg_loss가 0이 되는 경우를 처리 (0으로 나누는 오류 방지)
#     rs = avg_gain / avg_loss.replace(0, np.nan) # 0이면 NaN으로
#     df['RSI'] = 100 - (100 / (1 + rs))
#     return df

# @st.cache_data
# def calculate_moving_average(df, short_window, long_window):
#     df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
#     df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
#     return df

# @st.cache_data
# def generate_golden_cross_signals(df):
#     df['Signal_GC'] = 0
#     # 골든 크로스: 단기 MA가 장기 MA를 상향 돌파 (이전에는 낮았고 현재는 높거나 같음)
#     df['Buy_GC'] = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
#     # 데드 크로스: 단기 MA가 장기 MA를 하향 돌파 (이전에는 높았고 현재는 낮거나 같음)
#     df['Sell_GC'] = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))
#     df.loc[df['Buy_GC'], 'Signal_GC'] = 1
#     df.loc[df['Sell_GC'], 'Signal_GC'] = -1
#     return df

# @st.cache_data
# def generate_rsi_signals(df, buy_threshold=30, sell_threshold=70):
#     df['Signal_RSI'] = 0
#     # RSI 매수: RSI가 buy_threshold 이하로 내려갔다가 다시 넘어설 때 (과매도 구간 탈출)
#     # df['Buy_RSI'] = (df['RSI'] < buy_threshold) & (df['RSI'].shift(1) >= buy_threshold)
#     # 보통 RSI 매수 신호는 RSI가 30을 상향 돌파할 때로 봅니다.
#     df['Buy_RSI'] = (df['RSI'] > buy_threshold) & (df['RSI'].shift(1) <= buy_threshold)

#     # RSI 매도: RSI가 sell_threshold 이상으로 올라갔다가 다시 내려올 때 (과매수 구간 진입 후 하락)
#     # df['Sell_RSI'] = (df['RSI'] > sell_threshold) & (df['RSI'].shift(1) <= sell_threshold)
#     # 보통 RSI 매도 신호는 RSI가 70을 하향 돌파할 때로 봅니다.
#     df['Sell_RSI'] = (df['RSI'] < sell_threshold) & (df['RSI'].shift(1) >= sell_threshold)


#     df.loc[df['Buy_RSI'], 'Signal_RSI'] = 1
#     df.loc[df['Sell_RSI'], 'Signal_RSI'] = -1
#     return df

# @st.cache_data
# def generate_bollinger_signals(df):
#     df['Signal_BB'] = 0
#     # 볼린저 밴드 매수: 종가가 하한선 아래로 내려갔다가 다시 올라올 때 (과매도 구간 탈출)
#     df['Buy_BB'] = (df['Close'] > df['Lower']) & (df['Close'].shift(1) <= df['Lower'].shift(1))
#     # 볼린저 밴드 매도: 종가가 상한선 위로 올라갔다가 다시 내려올 때 (과매수 구간 진입 후 하락)
#     df['Sell_BB'] = (df['Close'] < df['Upper']) & (df['Close'].shift(1) >= df['Upper'].shift(1))
#     df.loc[df['Buy_BB'], 'Signal_BB'] = 1
#     df.loc[df['Sell_BB'], 'Signal_BB'] = -1
#     return df

# # 백테스팅 함수는 동일하게 사용
# def backtest(df, signal_column):
#     initial_balance = 1000000
#     balance = initial_balance
#     holdings = 0
#     transactions = []

#     # 첫 날은 신호 없다고 가정하고 건너뜀
#     # NaN 값 이후부터 시작
#     df_cleaned = df.dropna(subset=[signal_column])

#     if df_cleaned.empty:
#         return 0, pd.DataFrame(columns=['Date', 'Action', 'Price', 'Qty'])

#     # 매수/매도 로직
#     for i in range(1, len(df_cleaned)):
#         current_date = df_cleaned.index[i]
        
#         # 현재 보유하고 있는 종목이 없고, 매수 신호가 발생했고, 충분한 잔고가 있을 때
#         if df_cleaned[signal_column].iloc[i] == 1 and holdings == 0 and balance > 0:
#             price = df_cleaned['Close'].iloc[i]
#             if price > 0: # 가격이 0보다 커야 함
#                 qty = balance // price
#                 if qty > 0:
#                     holdings += qty
#                     balance -= qty * price
#                     transactions.append({'Date': current_date, 'Action': 'Buy', 'Price': price, 'Qty': qty})
        
#         # 현재 보유하고 있는 종목이 있고, 매도 신호가 발생했을 때
#         elif df_cleaned[signal_column].iloc[i] == -1 and holdings > 0:
#             price = df_cleaned['Close'].iloc[i]
#             balance += holdings * price
#             transactions.append({'Date': current_date, 'Action': 'Sell', 'Price': price, 'Qty': holdings})
#             holdings = 0 # 보유량 0으로 초기화

#     # 최종 자산 가치 계산 (남은 현금 + 보유 중인 주식 가치)
#     final_value = balance + holdings * df_cleaned['Close'].iloc[-1]
#     return_rate = (final_value - initial_balance) / initial_balance * 100
#     return return_rate, pd.DataFrame(transactions)

# # --------------------------------------------
# # Streamlit UI
# # --------------------------------------------

# # 기업 리스트 불러오기
# @st.cache_data
# company_df = get_company_list()

# # 파일 경로 설정 (company_list.csv용)
# current_dir = os.path.dirname(__file__)
# root_dir = os.path.join(current_dir, '..')
# company_list_file_path = os.path.join(root_dir, 'company_list.csv') # company_list.csv 파일명

# try:
#     # company_list.csv 파일 로드
#     df_company_list = pd.read_csv(company_list_file_path)
#     # df_company_list['Code'] = df_company_list['Code'].astype(str).str.zfill(6) # 필요하다면 코드 포맷팅
#     # code_name_map = df_company_list.set_index('Code')['Name'].to_dict() # 예시: 매핑 딕셔너리 생성
#     st.success(f"✅ 종목 리스트를 성공적으로 불러왔습니다. (파일: {company_list_file_path})")

# except FileNotFoundError:
#     st.error(f"❌ 종목 리스트 파일 '{company_list_file_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")
# except Exception as e:
#     st.error(f"종목 리스트 처리 중 오류가 발생했습니다: {e}")

# if not company_df.empty:
#     selected_label = st.selectbox("📊 분석할 종목을 선택하세요", company_df["label"].tolist())
#     selected_code = company_df[company_df["label"] == selected_label]["Code"].values[0]

#     # 날짜 선택
#     min_date_fdr = datetime.today() - timedelta(days=365 * 10) # FDataReader는 더 긴 기간 데이터 가능
#     max_date_fdr = datetime.today()
#     col_date_tech1, col_date_tech2 = st.columns(2)
#     with col_date_tech1:
#         start_date = st.date_input("시작일", min_value=min_date_fdr, max_value=max_date_fdr, value=min_date_fdr)
#     with col_date_tech2:
#         end_date = st.date_input("종료일", min_value=start_date, max_value=max_date_fdr, value=max_date_fdr)

#     if start_date >= end_date:
#         st.error("종료 날짜는 시작 날짜보다 미래여야 합니다.")
#         st.stop()

#     # 전략 선택 체크박스
#     st.write("### ⚙️ 백테스팅할 기술적 분석 전략을 선택하세요:")
#     run_gc_backtest = st.checkbox("골든크로스/데드크로스 전략 (단기 20일, 장기 60일)", value=True)
#     run_rsi_backtest = st.checkbox("RSI 전략 (매수 30, 매도 70)", value=True)
#     run_bb_backtest = st.checkbox("볼린저 밴드 전략 (20일, 2표준편차)", value=True)

#     if st.button("🚀 전략 백테스팅 시작"):
#         # FDataReader로 주식 데이터 로드
#         with st.spinner(f"🔄 {selected_label}의 데이터를 불러오는 중..."):
#             df = fdr.DataReader(selected_code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            
#         if df.empty or len(df) < 60: # 최소 60일 (장기 MA 기간) 데이터 필요
#             st.warning("데이터가 부족합니다. 선택된 기간이나 종목을 확인해주세요. 최소 60일 이상 데이터가 필요합니다.")
#         else:
#             st.subheader(f"📈 {selected_label} ({selected_code}) 주가 차트")
#             st.line_chart(df['Close'])
#             st.metric("📊 단순 매수 후 보유 (Buy & Hold) 수익률", f"{((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100:.2f}%")

#             # 골든크로스
#             if run_gc_backtest:
#                 st.markdown("---")
#                 st.subheader("💰 골든크로스/데드크로스 전략 백테스팅 결과")
#                 df_gc = calculate_moving_average(df.copy(), 20, 60)
#                 df_gc = generate_golden_cross_signals(df_gc)
#                 r_gc, log_gc = backtest(df_gc, 'Signal_GC')
#                 st.metric("수익률", f"{r_gc:.2f}%")
                
#                 fig_gc, ax_gc = plt.subplots(figsize=(12, 6))
#                 ax_gc.plot(df_gc.index, df_gc['Close'], label='종가', color='lightgray', linewidth=1)
#                 ax_gc.plot(df_gc.index, df_gc['Short_MA'], label='단기 MA (20일)', color='orange', linewidth=1.5)
#                 ax_gc.plot(df_gc.index, df_gc['Long_MA'], label='장기 MA (60일)', color='purple', linewidth=1.5)
                
#                 # 매수/매도 신호 시각화
#                 buy_signals_gc = df_gc[df_gc['Buy_GC'] == True]
#                 sell_signals_gc = df_gc[df_gc['Sell_GC'] == True]
#                 ax_gc.scatter(buy_signals_gc.index, buy_signals_gc['Close'], marker='^', color='green', s=100, label='매수 (골든크로스)', zorder=5)
#                 ax_gc.scatter(sell_signals_gc.index, sell_signals_gc['Close'], marker='v', color='red', s=100, label='매도 (데드크로스)', zorder=5)
                
#                 ax_gc.set_title("골든크로스/데드크로스 전략")
#                 ax_gc.legend()
#                 ax_gc.grid(True)
#                 st.pyplot(fig_gc)
                
#                 if not log_gc.empty:
#                     with st.expander("매매 기록 보기"):
#                         st.dataframe(log_gc)
#                 else:
#                     st.info("해당 기간 동안 골든크로스/데드크로스 매매 신호가 발생하지 않았습니다.")

#             # RSI
#             if run_rsi_backtest:
#                 st.markdown("---")
#                 st.subheader("💰 RSI 전략 백테스팅 결과")
#                 df_rsi = calculate_rsi(df.copy())
#                 df_rsi = generate_rsi_signals(df_rsi)
#                 r_rsi, log_rsi = backtest(df_rsi, 'Signal_RSI')
#                 st.metric("수익률", f"{r_rsi:.2f}%")
                
#                 fig_rsi, ax_rsi = plt.subplots(figsize=(12, 6))
#                 ax_rsi.plot(df_rsi.index, df_rsi['Close'], label='종가', color='lightgray', linewidth=1)
#                 ax_rsi.plot(df_rsi.index, df_rsi['RSI'], label='RSI', color='blue', linewidth=1.5)
#                 ax_rsi.axhline(70, color='red', linestyle='--', label='과매수 (70)')
#                 ax_rsi.axhline(30, color='green', linestyle='--', label='과매도 (30)')

#                 # 매수/매도 신호 시각화
#                 buy_signals_rsi = df_rsi[df_rsi['Buy_RSI'] == True]
#                 sell_signals_rsi = df_rsi[df_rsi['Sell_RSI'] == True]
#                 ax_rsi.scatter(buy_signals_rsi.index, buy_signals_rsi['Close'], marker='^', color='green', s=100, label='매수 (RSI)', zorder=5)
#                 ax_rsi.scatter(sell_signals_rsi.index, sell_signals_rsi['Close'], marker='v', color='red', s=100, label='매도 (RSI)', zorder=5)

#                 ax_rsi.set_title("RSI 전략")
#                 ax_rsi.legend()
#                 ax_rsi.grid(True)
#                 st.pyplot(fig_rsi)

#                 if not log_rsi.empty:
#                     with st.expander("매매 기록 보기"):
#                         st.dataframe(log_rsi)
#                 else:
#                     st.info("해당 기간 동안 RSI 매매 신호가 발생하지 않았습니다.")

#             # 볼린저밴드
#             if run_bb_backtest:
#                 st.markdown("---")
#                 st.subheader("💰 볼린저 밴드 전략 백테스팅 결과")
#                 df_bb = calculate_bollinger_bands(df.copy())
#                 df_bb = generate_bollinger_signals(df_bb)
#                 r_bb, log_bb = backtest(df_bb, 'Signal_BB')
#                 st.metric("수익률", f"{r_bb:.2f}%")
                
#                 fig_bb, ax_bb = plt.subplots(figsize=(12, 6))
#                 ax_bb.plot(df_bb.index, df_bb['Close'], label='종가', color='lightgray', linewidth=1)
#                 ax_bb.plot(df_bb.index, df_bb['Upper'], label='상한선', color='red', linestyle='--')
#                 ax_bb.plot(df_bb.index, df_bb['MA'], label='중간선', color='blue')
#                 ax_bb.plot(df_bb.index, df_bb['Lower'], label='하한선', color='green', linestyle='--')

#                 # 매수/매도 신호 시각화
#                 buy_signals_bb = df_bb[df_bb['Buy_BB'] == True]
#                 sell_signals_bb = df_bb[df_bb['Sell_BB'] == True]
#                 ax_bb.scatter(buy_signals_bb.index, buy_signals_bb['Close'], marker='^', color='green', s=100, label='매수 (BB)', zorder=5)
#                 ax_bb.scatter(sell_signals_bb.index, sell_signals_bb['Close'], marker='v', color='red', s=100, label='매도 (BB)', zorder=5)

#                 ax_bb.set_title("볼린저 밴드 전략")
#                 ax_bb.legend()
#                 ax_bb.grid(True)
#                 st.pyplot(fig_bb)

#                 if not log_bb.empty:
#                     with st.expander("매매 기록 보기"):
#                         st.dataframe(log_bb)
#                 else:
#                     st.info("해당 기간 동안 볼린저 밴드 매매 신호가 발생하지 않았습니다.")

# else:
#     st.info("상단에서 종목과 기간을 선택하고, 백테스팅할 전략을 선택 후 '전략 백테스팅 시작' 버튼을 눌러주세요.")

# st.markdown("---")
# st.write("### 참고")
# st.write("""

# - **백테스팅 모델의 한계:** 거래 수수료, 슬리피지 등을 고려하지 않은 단순 모델입니다.
# """)
