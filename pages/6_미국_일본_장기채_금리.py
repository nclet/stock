import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf
from datetime import datetime, timedelta
import traceback

# --- 설정 ---
st.set_page_config(page_title="미국-일본 10년물 국채 금리 및 시장 분석", layout="wide")

# FRED API 키를 st.secrets에서 불러옵니다.
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    st.error("🚨 FRED API 키('FRED_API_KEY')가 Streamlit Secrets에 설정되어 있지 않습니다.")
    st.info("Streamlit Cloud 대시보드의 'Settings' -> 'Secrets' 메뉴에서 'FRED_API_KEY'를 설정해주세요.")
    st.stop() # API 키 없으면 앱 실행 중지

fred = Fred(api_key=FRED_API_KEY)

# --- 데이터 불러오기 함수 ---
@st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
def load_all_data(start_date, end_date):
    data = {}
    errors = []

    # 1. 미국 10년물 국채 금리 (일별)
    st.info("🔄 미국 10년물 국채 금리 데이터를 불러오는 중...")
    try:
        us_10y = fred.get_series('GS10', start_date, end_date)
        if us_10y is None or us_10y.empty:
            errors.append("❌ 미국 10년물 금리 데이터 로드 실패: 'GS10'. 기간을 조정해 보세요.")
        else:
            data['US_10Y'] = us_10y.rename("US_10Y")
    except Exception as e:
        errors.append(f"❌ 미국 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")


    # 2. 일본 10년물 국채 금리 (월별)
    st.info("🔄 일본 10년물 국채 금리 데이터를 불러오는 중... (FRED: 월별 데이터)")
    try:
        # 'IRLTLT01JPM156N': OECD Long-Term Interest Rate: 10-Year Government Rate for Japan, Monthly
        jgb_10y = fred.get_series('IRLTLT01JPM156N', start_date, end_date) 
        if jgb_10y is None or jgb_10y.empty:
            errors.append("❌ 일본 10년물 금리 데이터 로드 실패: 'IRLTLT01JPM156N'. 기간을 조정해 보세요.")
            st.info("참고: FRED에서 제공하는 일본 10년물 국채 금리 데이터는 월별입니다.")
        else:
            data['JP_10Y'] = jgb_10y.rename("JP_10Y")
    except Exception as e:
        errors.append(f"❌ 일본 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

    # 3. S&P 500 지수 (일별) - 'Adj Close' 대신 'Close' 사용
    st.info("🔄 S&P 500 지수 데이터를 불러오는 중...")
    sp500_ticker = "^GSPC" # S&P 500 티커
    try:
        # start/end date를 yfinance가 선호하는 문자열 형식으로 변환
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        sp500_data = yf.download(sp500_ticker, start=start_date_str, end=end_date_str)["Close"]
        
        if sp500_data.empty:
            errors.append(f"❌ S&P 500 지수 데이터 로드 실패: '{sp500_ticker}' (데이터 없음). 티커 또는 기간을 확인하세요.")
        else:
            data['SP500'] = sp500_data.rename("SP500")
    except Exception as e:
        errors.append(f"❌ S&P 500 지수 데이터 로드 중 오류 발생: {e}. YFinance 문제일 수 있습니다. Traceback: {traceback.format_exc()}")


    if errors:
        for err in errors:
            st.error(err)
        st.warning("일부 데이터 로드에 실패했습니다. 그래프가 올바르게 표시되지 않을 수 있습니다.")
        return pd.DataFrame() # 오류가 있다면 빈 DataFrame 반환

    # 모든 데이터프레임을 하나의 DataFrame으로 합치기
    df = pd.DataFrame()
    for key, series in data.items():
        if not series.empty:
            df = pd.concat([df, series.resample('D').mean()], axis=1)
    
    # 인덱스를 datetime 형식으로 통일
    df.index = pd.to_datetime(df.index)
    
    # 10년물 스프레드 계산
    df["Spread_10Y"] = df["US_10Y"] - df["JP_10Y"]
    
    required_cols = ['US_10Y', 'JP_10Y', 'Spread_10Y', 'SP500']
    df = df[required_cols].dropna(subset=['US_10Y', 'JP_10Y', 'SP500', 'Spread_10Y'], how='any')

    if df.empty:
        st.warning("선택된 기간에 유효한 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
    return df

# --- 날짜 선택 ---
st.sidebar.title("📅 기간 선택")
start_date = st.sidebar.date_input("시작일", datetime.today() - timedelta(days=365 * 5))
end_date = st.sidebar.date_input("종료일", datetime.today())

# --- 데이터 불러오기 ---
with st.spinner("📊 필요한 모든 데이터를 불러오는 중... (FRED 및 YFinance 사용)"):
    df = load_all_data(start_date, end_date)

# --- 시각화 ---
if not df.empty:
    st.title("🇺🇸 미국-일본 10년물 국채 금리 및 시장 분석")

    # 첫 번째 섹션: 10년물 금리 비교
    st.subheader("1. 미국 vs 일본 10년물 국채 금리")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    df["US_10Y"].plot(ax=ax1, label="미국 10Y", color="blue", linewidth=1.5)
    df["JP_10Y"].plot(ax=ax1, label="일본 10Y (월별)", color="red", linewidth=1.5)
    ax1.set_ylabel("금리 (%)")
    ax1.set_title("미국 vs 일본 10년물 국채 금리 추이")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig1)

    # 두 번째 섹션: 10년물 금리 스프레드와 S&P 500 지수
    st.subheader("2. 미국-일본 10년물 금리 스프레드와 S&P 500 지수")
    fig3, ax3_primary = plt.subplots(figsize=(12, 7))

    color_spread = 'tab:green'
    ax3_primary.set_xlabel('날짜')
    ax3_primary.set_ylabel('금리 스프레드 (%)', color=color_spread)
    ax3_primary.plot(df.index, df["Spread_10Y"], label="미국-일본 10Y 금리 스프레드", color=color_spread, linewidth=2)
    ax3_primary.tick_params(axis='y', labelcolor=color_spread)
    ax3_primary.axhline(0, color="gray", linestyle="--", alpha=0.7) # 0% 스프레드 라인
    ax3_primary.legend(loc='upper left')
    
    # S&P 500을 위한 보조 y축
    ax3_secondary = ax3_primary.twinx()
    color_sp500 = 'tab:purple'
    ax3_secondary.set_ylabel('S&P 500 지수', color=color_sp500)
    ax3_secondary.plot(df.index, df["SP500"], label="S&P 500 지수", color=color_sp500, linestyle='--', linewidth=1.5)
    ax3_secondary.tick_params(axis='y', labelcolor=color_sp500)
    ax3_secondary.legend(loc='upper right')

    ax3_primary.set_title("미국-일본 10년물 금리 스프레드와 S&P 500 지수")
    ax3_primary.grid(True, linestyle='--', alpha=0.7)
    fig3.tight_layout() # 레이아웃 자동 조정
    st.pyplot(fig3)

else:
    st.warning("데이터를 불러오지 못했거나 선택된 기간에 유효한 데이터가 없습니다. 날짜 범위를 조정해 보세요.")

# --- 해석 도움말 ---
with st.expander("📖 분석 해석 가이드"):
    st.markdown("""
    #### 📈 금리 스프레드와 엔캐리 트레이드, 미국 증시

    - **엔캐리 트레이드(Yen Carry Trade)**: 일본의 낮은 금리(낮은 대출 비용)를 활용하여 엔화를 빌린 후, 이 자금으로 미국 등 금리가 높은 국가의 자산(주식, 채권)에 투자하여 금리 차이(스프레드)만큼 수익을 추구하는 전략입니다.

    - **금리 차이 확대 (스프레드 상승):**
        - 미국 금리 > 일본 금리 (금리 차이 확대)
        - 엔캐리 트레이드 유지 또는 활성화 → 엔화 매도, 달러/미국 자산 매수 → 미국 증시 **긍정적** 영향
        - 그래프에서 **초록색 선(스프레드)이 상승**하는 시기.

    - **금리 차이 축소 (스프레드 하락):**
        - 일본 금리 상승 또는 미국 금리 하락 (금리 차이 축소)
        - 엔캐리 트레이드의 수익성이 줄어들거나 손실 위험 → 엔화를 되갚기 위해 미국 자산 매도 → 엔화 매수 → 미국 증시 **조정 또는 하락 압력**
        - 그래프에서 **초록색 선(스프레드)이 하락**하는 시기. 특히 0% 또는 그 이하로 근접하면 엔캐리 트레이드의 청산(unwind)이 가속화될 수 있다는 신호로 해석되기도 합니다.

    #### 📊 그래프 해석 포인트

    - **10년물 금리 그래프:** 미국과 일본의 10년물 국채 금리 변화 추이를 한눈에 파악할 수 있습니다. 일본 금리가 미국 금리보다 훨씬 낮은 수준을 유지하는 것이 일반적입니다.
    - **스프레드 & S&P 500 그래프:**
        - **초록색 선 (금리 스프레드)**: 미국과 일본 10년물 국채 금리 차이를 나타냅니다.
        - **보라색 점선 (S&P 500 지수)**: 미국 주식 시장의 대표 지수입니다.
        - **관찰 포인트**: 초록색 선(스프레드)이 급격히 하락하는 시기에 보라색 점선(S&P 500)도 함께 하락하는 경향이 있는지 주시하세요. 이는 엔캐리 트레이드 청산이 미국 증시에 미치는 영향을 시사할 수 있습니다.

    ---
    **⚠️ 데이터 빈도 참고사항:**
    - 미국 10년물 국채 금리 및 S&P 500 지수는 **일별(Daily) 데이터**입니다.
    - 일본 10년물 국채 금리 데이터는 FRED에서 **월별(Monthly) 기준**으로 제공됩니다.
    - 따라서 그래프 상에서 일본 금리 데이터는 해당 월의 첫 영업일에만 업데이트되는 것처럼 보일 수 있으며, 금리 스프레드 역시 월별 데이터가 존재하는 날짜에만 계산됩니다.
    """)
