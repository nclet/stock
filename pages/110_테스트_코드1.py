import streamlit as st
import requests
from json.decoder import JSONDecodeError
import FinanceDataReader as fdr
import pyupbit
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import datetime
import itertools

# ---------------------------------------------------------------------------------
# 1. 설정 및 매핑
# ---------------------------------------------------------------------------------
pattern_mapping = {
    'is_hammer': {'label': '망치형 (상승)', 'initial': '[H]'},
    'is_inverted_hammer': {'label': '역망치형 (하락)', 'initial': '[IH]'},
    'is_doji': {'label': '도지형', 'initial': '[D]'},
    'is_bullish_engulfing': {'label': '상승장악형', 'initial': '[BE]'},
    'is_bearish_engulfing': {'label': '하락장악형', 'initial': '[BEE]'},
    'is_piercing_line': {'label': '관통형', 'initial': '[PL]'},
    'is_dark_cloud_cover': {'label': '흑운형', 'initial': '[DCC]'},
    'is_three_white_soldiers': {'label': '적삼병', 'initial': '[TWS]'},
    'is_three_black_crows': {'label': '흑삼병', 'initial': '[TBC]'},
    'is_shooting_star': {'label': '유성형', 'initial': '[SS]'},
    'is_hanging_man': {'label': '교수형', 'initial': '[HM]'}
}

BUY_PATTERNS = ['is_hammer', 'is_bullish_engulfing', 'is_piercing_line', 'is_three_white_soldiers', 'is_inverted_hammer']
SELL_PATTERNS = ['is_shooting_star', 'is_hanging_man', 'is_bearish_engulfing', 'is_dark_cloud_cover', 'is_three_black_crows']

# ---------------------------------------------------------------------------------
# 2. 데이터 로드 및 패턴 분석 함수 (기존 유지 및 확장)
# ---------------------------------------------------------------------------------
@st.cache_data
def get_stock_listing(market):
    try:
        df = fdr.StockListing(market)
        if 'Code' not in df.columns and 'Symbol' in df.columns:
            df.rename(columns={'Symbol': 'Code'}, inplace=True)
        df['Code'] = df['Code'].astype(str)
        df['label'] = df['Name'] + ' (' + df['Code'] + ')'
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def get_coin_listing():
    try:
        url = "https://api.upbit.com/v1/market/all"
        response = requests.get(url, params={'isDetails': 'false'})
        all_markets = response.json()
        krw_markets = [market for market in all_markets if market['market'].startswith('KRW-')]
        df_coin = pd.DataFrame(krw_markets)
        df_coin['label'] = df_coin['korean_name'] + ' (' + df_coin['market'].str.replace('KRW-', '') + ')'
        df_coin.rename(columns={'market': 'Code'}, inplace=True)
        return df_coin
    except Exception:
        return pd.DataFrame()

def get_data(ticker, start_date, end_date, market, period='일봉'):
    try:
        if market in ['한국 주식 (KRX)', '미국 증시 (NYSE/NASDAQ)']:
            data = fdr.DataReader(ticker, start_date, end_date)
            if period == '주봉':
                data = data.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            elif period == '월봉':
                data = data.resample('M').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            data.index.name = 'Date'
            return data
        elif market == '코인 (Upbit)':
            upbit_period_map = {'일봉': 'day', '주봉': 'week', '월봉': 'month'}
            days_diff = (end_date - start_date).days
            count = days_diff + 1 if period == '일봉' else int(days_diff / 7) + 1
            df = pyupbit.get_ohlcv(ticker=ticker, interval=upbit_period_map[period], count=count)
            if df is not None:
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'value']
                df.index.name = 'Date'
            return df
    except Exception:
        return None

def find_candle_patterns(df):
    for p in pattern_mapping.keys(): df[p] = False
    for i in range(len(df)):
        open_p, close_p, high_p, low_p = df.iloc[i][['Open', 'Close', 'High', 'Low']]
        body_length = abs(close_p - open_p)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p
        
        if body_length > 0 and lower_shadow > 2 * body_length and upper_shadow < body_length:
            if close_p > open_p: df.loc[df.index[i], 'is_hammer'] = True
            elif close_p < open_p: df.loc[df.index[i], 'is_hanging_man'] = True
        if body_length > 0 and upper_shadow > 2 * body_length and lower_shadow < body_length:
            if close_p > open_p: df.loc[df.index[i], 'is_inverted_hammer'] = True
            elif close_p < open_p: df.loc[df.index[i], 'is_shooting_star'] = True
        if body_length < (high_p - low_p) * 0.05: df.loc[df.index[i], 'is_doji'] = True

        if i >= 1:
            p_o, p_c, p_h, p_l = df.iloc[i-1][['Open', 'Close', 'High', 'Low']]
            p_mid = (p_o + p_c) / 2
            if (p_c < p_o and close_p > open_p and open_p < p_c and close_p > p_o): df.loc[df.index[i], 'is_bullish_engulfing'] = True
            if (p_c > p_o and close_p < open_p and open_p > p_c and close_p < p_o): df.loc[df.index[i], 'is_bearish_engulfing'] = True
            if (p_c < p_o and close_p > open_p and open_p < p_l and close_p > p_mid and close_p < p_o): df.loc[df.index[i], 'is_piercing_line'] = True
            if (p_c > p_o and close_p < open_p and open_p > p_h and close_p < p_mid and close_p > p_o): df.loc[df.index[i], 'is_dark_cloud_cover'] = True
        if i >= 2:
            p2_o, p2_c = df.iloc[i-2][['Open', 'Close']]
            p1_o, p1_c = df.iloc[i-1][['Open', 'Close']]
            if (p2_c > p2_o and p1_c > p1_o and close_p > open_p and p1_c > p2_c and close_p > p1_c): df.loc[df.index[i], 'is_three_white_soldiers'] = True
            if (p2_c < p2_o and p1_c < p1_o and close_p < open_p and p1_c < p2_c and close_p < p1_c): df.loc[df.index[i], 'is_three_black_crows'] = True
    return df

# ---------------------------------------------------------------------------------
# 3. 추가 기능: 패턴별 승률 분석 및 조합 추천
# ---------------------------------------------------------------------------------
def analyze_pattern_performance(df, pattern_cols):
    results = []
    for col in pattern_cols:
        trades = []
        sig_indices = df.index[df[col] == True].tolist()
        for idx in sig_indices:
            i = df.index.get_loc(idx)
            if i + 1 < len(df):
                buy_price = df.iloc[i]['Close']
                exit_price = df.iloc[i+1]['Open']
                trades.append((exit_price - buy_price) / buy_price * 100)
        
        if trades:
            win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
            avg_ret = sum(trades) / len(trades)
            results.append({'Pattern': pattern_mapping[col]['label'], 'WinRate': win_rate, 'AvgReturn': avg_ret, 'Count': len(trades), 'col': col})
    return pd.DataFrame(results)

def recommend_best_combinations(df, pattern_cols):
    # 2가지 패턴 조합 중 가장 승률 높은 조합 탐색
    combos = list(itertools.combinations(pattern_cols, 2))
    combo_results = []
    for c in combos:
        signal = df[list(c)].any(axis=1)
        trades = []
        sig_indices = df.index[signal == True].tolist()
        for idx in sig_indices:
            i = df.index.get_loc(idx)
            if i + 1 < len(df):
                trades.append((df.iloc[i+1]['Open'] - df.iloc[i]['Close']) / df.iloc[i]['Close'] * 100)
        if trades:
            win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
            combo_results.append({'Combo': f"{pattern_mapping[c[0]]['initial']} + {pattern_mapping[c[1]]['initial']}", 'WinRate': win_rate, 'Count': len(trades)})
    return pd.DataFrame(combo_results).sort_values('WinRate', ascending=False).head(3)

# ---------------------------------------------------------------------------------
# 4. UI 구성 (기존 레이아웃 유지하며 탭 추가)
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="주식 & 코인 패턴 분석 프로", layout="wide")
tab1, tab2 = st.tabs(["📈 차트 분석 & 백테스트", "🔍 실시간 패턴 스캐너"])

with tab1:
    st.markdown("<h1 style='text-align: center;'>캔들 패턴 분석 및 전략 리포트</h1>", unsafe_allow_html=True)
    
    # (기존 선택 옵션 부분)
    selected_market = st.radio("💰 시장 선택", ('한국 주식 (KRX)', '미국 증시 (NYSE/NASDAQ)', '코인 (Upbit)'), horizontal=True)
    df_listing = get_stock_listing('KRX') if '한국' in selected_market else get_stock_listing('NASDAQ') if '미국' in selected_market else get_coin_listing()

    if not df_listing.empty:
        selected_label = st.selectbox(f"📊 종목 선택", df_listing["label"].tolist())
        selected_code = df_listing[df_listing["label"] == selected_label]["Code"].values[0]
        
        col1, col2 = st.columns(2)
        with col1:
            selected_period = st.radio("⏳ 기간", ('일봉', '주봉', '월봉'), horizontal=True)
            start_date = st.date_input("시작", datetime.date.today() - datetime.timedelta(days=365))
            end_date = st.date_input("종료", datetime.date.today())
        with col2:
            all_pattern_options = {f"{v['label']} {v['initial']}": k for k, v in pattern_mapping.items()}
            selected_patterns = st.multiselect("📈 분석할 패턴 (복수 선택 가능)", list(all_pattern_options.keys()), default=list(all_pattern_options.keys())[:3])

        if st.button("전략 분석 실행", type="primary", use_container_width=True):
            df = get_data(selected_code, start_date, end_date, selected_market, selected_period)
            if df is not None and not df.empty:
                df_with_patterns = find_candle_patterns(df.copy())
                selected_pattern_cols = [all_pattern_options[p] for p in selected_patterns]
                
                # 차트 출력 (기존 mpf 코드 요약 적용)
                st.subheader("5. 캔들 차트 및 지표")
                fig, _ = mpf.plot(df_with_patterns, type='candle', style='charles', figratio=(15,7), returnfig=True, volume=True)
                st.pyplot(fig)

                # --- 신규 추가: 패턴별 승률 리포트 ---
                st.markdown("---")
                st.subheader("📊 패턴별 개별 성과 리포트")
                perf_df = analyze_pattern_performance(df_with_patterns, selected_pattern_cols)
                if not perf_df.empty:
                    cols = st.columns(len(perf_df))
                    for i, row in perf_df.iterrows():
                        cols[i%3].metric(row['Pattern'], f"{row['WinRate']:.1f}%", f"{row['AvgReturn']:.2f}% (Avg)")
                    st.dataframe(perf_df[['Pattern', 'WinRate', 'AvgReturn', 'Count']], use_container_width=True)
                
                # --- 신규 추가: 패턴 조합 추천 ---
                st.subheader("💡 추천 패턴 조합 (승률 TOP 3)")
                recom_df = recommend_best_combinations(df_with_patterns, selected_pattern_cols)
                if not recom_df.empty:
                    st.table(recom_df)
                else:
                    st.write("추천할 만한 충분한 조합 데이터가 없습니다.")

with tab2:
    st.subheader("🔍 실시간 마켓 패턴 스캐너")
    st.write("현재 시장에서 즉시 매수/매도 신호가 발생한 종목을 스캔합니다.")
    
    scan_market = st.selectbox("스캔할 시장 선택", ["코인 (Upbit)", "한국 주식 (KOSPI)"])
    
    if st.button("스캔 시작"):
        with st.spinner("전 종목 스캔 중..."):
            results = []
            # 샘플로 상위 15개 종목만 스캔 (속도 문제상)
            tickers = pyupbit.get_tickers(fiat="KRW")[:20] if "코인" in scan_market else ["005930", "000660", "035420", "035720", "005380"]
            
            for t in tickers:
                # 최근 10일치 데이터만 가져옴
                df_scan = get_data(t, datetime.date.today()-datetime.timedelta(days=10), datetime.date.today(), "코인 (Upbit)" if "코인" in scan_market else "한국 주식 (KRX)")
                if df_scan is not None:
                    df_scan = find_candle_patterns(df_scan)
                    last_row = df_scan.iloc[-1]
                    found = [pattern_mapping[p]['label'] for p in pattern_mapping.keys() if last_row[p]]
                    if found:
                        results.append({"종목/티커": t, "발견된 패턴": ", ".join(found), "현재가": last_row['Close']})
            
            if results:
                st.success(f"{len(results)}개의 종목에서 패턴이 발견되었습니다!")
                st.table(pd.DataFrame(results))
            else:
                st.info("현재 발견된 패턴이 없습니다.")


# --- 캔들 패턴 설명 (st.expander와 st.markdown 사용) ---
# 날짜 범위 선택 바로 위에 위치
with st.expander("캔들 패턴 참고자료 📖"):
    st.markdown("""
    이 앱에서 분석하는 주요 캔들 패턴에 대한 간단한 설명입니다.

    - **🔎도지(Doji)**: 시가와 종가가 거의 같은 십자형 캔들입니다. 매수자와 매도자가 서로 힘의 균형을 이루고 있다는 것을 나타내며, 추세 전환의 신호일 수 있습니다.
    - **🔎망치형 (Hammer)**: 긴 아래 꼬리와 짧은 몸통을 가진 캔들입니다. 하락 추세에서 나타나면 바닥을 확인하고 반등할 가능성을 시사합니다.
    - **장악형 (Engulfing)**: 현재 캔들이 이전 캔들의 몸통을 완전히 감싸는 형태입니다.
        - **📈상승 장악형 (Bullish Engulfing)**: 큰 양봉이 이전 음봉을 감싸는 형태로, 강한 매수세와 상승 반전을 예고합니다.
        - **📉하락 장악형 (Bearish Engulfing)**: 큰 음봉이 이전 양봉을 감싸는 형태로, 강한 매도세와 하락 반전을 예고합니다.
    - **📈모닝 스타 (Morning Star)**: 하락 추세에서 나타나는 3개의 캔들 패턴입니다. 큰 음봉, 작은 캔들, 그리고 큰 양봉이 순서대로 나타나며, 강력한 상승 반전 신호입니다.
    - **샛별형 (Star)**: 몸통이 이전 캔들의 몸통 위에 위치하는 캔들입니다.
        - **📈상승 샛별형 (Bullish Star)**: 큰 음봉 이후 작은 캔들이 나타나며, 상승 반전 가능성을 시사합니다.
        - **📉하락 샛별형 (Bearish Star)**: 큰 양봉 이후 작은 캔들이 나타나며, 하락 반전 가능성을 시사합니다.
    - **🔎십자 샛별형 (Doji Star)**: 샛별형의 작은 캔들이 도지 형태인 경우입니다. 추세 전환의 신호로 더 강력하게 해석됩니다.
    - **📈관통형 (Piercing Line)**: 하락 추세에서 첫 날 큰 음봉이 나타나고, 다음 날 양봉이 나타나는데, 이 양봉의 종가가 이전 날 음봉의 중간 지점을 뚫고 올라가는 형태입니다. 하락 추세가 끝날 수 있다는 긍정적인 신호로 해석됩니다.
    - **📉흑운형 (Dark Cloud Cover)**: 상승 추세에서 첫 날 양봉이 나타나고, 다음 날 음봉이 나타나는데, 이 음봉의 종가가 이전 날 양봉의 중간 지점을 뚫고 내려오는 형태입니다. 매도세가 강해져 상승 추세가 꺾일 수 있다는 부정적인 신호입니다.
    - **📈적삼병 (Three White Soldiers)**: 3일 연속 양봉이 나타나는 패턴입니다. 각 양봉의 종가가 이전 날의 종가보다 높게 끝나며, 강력한 상승 추세의 시작을 알리는 신호입니다.
    - **📉흑삼병 (Three Black Crows)**: 3일 연속 음봉이 나타나는 패턴입니다. 각 음봉의 종가가 이전 날의 종가보다 낮게 끝나며, 강력한 하락 추세의 시작을 알리는 신호입니다.
    - **📉유성형 (Shooting Star)**: 긴 위 꼬리와 짧은 몸통을 가진 캔들입니다. 상승 추세에서 나타나면 고점에서 매수세가 약해졌다는 것을 보여주며, 하락 반전 가능성을 시사합니다.
    - **📉교수형 (Hanging Man)**: 망치형과 모양은 비슷하지만, 상승 추세에서 나타납니다. 주가가 고점에서 하락할 가능성이 있다는 경고 신호로 해석됩니다.
    """)
