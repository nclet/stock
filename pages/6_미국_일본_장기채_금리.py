import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
# yfinance는 더 이상 사용하지 않으므로 제거합니다.
from datetime import datetime, timedelta
import traceback # 오류 스택 추적을 위해 임포트

# --- 설정 ---
st.set_page_config(page_title="미국-일본 10년물 금리차 대시보드", layout="wide")

# FRED API 키를 st.secrets에서 불러옵니다.
# secrets.toml 파일에 FRED_API_KEY = "YOUR_KEY" 형태로 저장되어 있어야 합니다.
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    st.error("🚨 FRED API 키('FRED_API_KEY')가 Streamlit Secrets에 설정되어 있지 않습니다.")
    st.info("Streamlit Cloud 대시보드의 'Settings' -> 'Secrets' 메뉴에서 'FRED_API_KEY'를 설정해주세요.")
    st.stop() # API 키 없으면 앱 실행 중지

fred = Fred(api_key=FRED_API_KEY)

# --- 데이터 불러오기 함수 ---
@st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
def load_yield_data(start_date, end_date):
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


    # 2. 일본 10년물 국채 금리 (월별) - FRED에서 가져오도록 수정
    st.info("🔄 일본 10년물 국채 금리 데이터를 불러오는 중... (FRED: 월별 데이터)")
    try:
        # 'IRLTLT01JPM156N': OECD Long-Term Interest Rate: 10-Year Government Bonds for Japan, Monthly
        jgb_10y = fred.get_series('IRLTLT01JPM156N', start_date, end_date) 
        if jgb_10y is None or jgb_10y.empty:
            errors.append("❌ 일본 10년물 금리 데이터 로드 실패: 'IRLTLT01JPM156N'. 기간을 조정해 보세요.")
            st.info("참고: FRED에서 제공하는 일본 10년물 국채 금리 데이터는 월별입니다.")
        else:
            data['JP_10Y'] = jgb_10y.rename("JP_10Y")
    except Exception as e:
        errors.append(f"❌ 일본 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")
    
    # S&P 500 지수 데이터 수집 부분은 완전히 제거되었습니다.

    if errors:
        for err in errors:
            st.error(err)
        st.warning("일부 데이터 로드에 실패했습니다. 그래프가 올바르게 표시되지 않을 수 있습니다.")
        return pd.DataFrame() # 오류가 있다면 빈 DataFrame 반환

    # 모든 데이터프레임을 하나의 DataFrame으로 합치기
    df = pd.DataFrame()
    for key, series in data.items():
        if not series.empty:
            df = pd.concat([df, series.resample('D').mean()], axis=1) # 월별 데이터는 해당 월의 모든 일자에 해당 월의 값 적용
    
    # 인덱스를 datetime 형식으로 통일
    df.index = pd.to_datetime(df.index)
    
    # 10년물 스프레드 계산
    df["Spread"] = df["US_10Y"] - df["JP_10Y"]
    
    # 최종적으로 필요한 컬럼만 남기고 NaN 값 제거
    df = df.dropna(subset=['US_10Y', 'JP_10Y', 'Spread'], how='any')

    if df.empty:
        st.warning("선택된 기간에 유효한 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
    return df

# --- 날짜 선택 ---
st.sidebar.title("📅 기간 선택")
start_date = st.sidebar.date_input("시작일", datetime.today() - timedelta(days=365 * 5)) # 기본 기간을 5년으로 늘림
end_date = st.sidebar.date_input("종료일", datetime.today())

# --- 데이터 불러오기 ---
with st.spinner("📊 데이터를 불러오는 중... (FRED 사용)"): # YFinance 사용 문구 제거
    df = load_yield_data(start_date, end_date)

# --- 시각화 ---
if not df.empty:
    st.title("미국·일본 10년 국채 금리 및 스프레드")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("미국 vs 일본 10년물 금리")
        fig1, ax1 = plt.subplots(figsize=(10, 6)) # figsize 추가
        df["US_10Y"].plot(ax=ax1, label="AMERICA 10Y", color="blue", linewidth=1.5)
        df["JP_10Y"].plot(ax=ax1, label="JAPAN 10Y", color="red", linewidth=1.5) # 월별임을 명시
        ax1.set_ylabel("Interest rate(%)")
        ax1.set_title("U.S, Japan 10-year bond spread") # 제목 추가
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7) # 그리드 추가
        st.pyplot(fig1)

    with col2:
        st.subheader("🇺🇸-🇯🇵 금리 스프레드")
        fig2, ax2 = plt.subplots(figsize=(10, 6)) # figsize 추가
        df["Spread"].plot(ax=ax2, color="green", linewidth=2)
        ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax2.set_ylabel("Interest rate(%)")
        ax2.set_title("U.S.-Japan 10-year interest rate spread") # 제목 추가
        ax2.grid(True, linestyle='--', alpha=0.7) # 그리드 추가
        st.pyplot(fig2)
    

else:
    st.warning("데이터를 불러오지 못했거나 선택된 기간에 유효한 데이터가 없습니다. 날짜 범위를 조정해 보세요.")

# --- 해석 도움말 ---
with st.expander("📖 금리 스프레드 해석 가이드"):
    st.markdown("""
    - **엔캐리 트레이드(Yen Carry Trade)**: 일본의 낮은 금리(낮은 대출 비용)를 활용하여 엔화를 빌린 후, 이 자금으로 미국 등 금리가 높은 국가의 자산(주식, 채권)에 투자하여 금리 차이(스프레드)만큼 수익을 추구하는 전략입니다.

    - **금리 차이 확대 (스프레드 상승):**
        - 미국 금리 > 일본 금리 (금리 차이 확대)
        - 엔캐리 트레이드 유지 또는 활성화 → 엔화 매도, 달러/미국 자산 매수 → 미국 증시 **긍정적** 영향
        - 그래프에서 **초록색 선(스프레드)이 상승**하는 시기.

    - **금리 차이 축소 (스프레드 하락):**
        - 일본 금리 상승 또는 미국 금리 하락 (금리 차이 축소)
        - 엔캐리 트레이드의 수익성이 줄어들거나 손실 위험 → 엔화를 되갚기 위해 미국 자산 매도 → 엔화 매수 → 미국 증시 **조정 또는 하락 압력**
        - 그래프에서 **초록색 선(스프레드)이 하락**하는 시기. 특히 0% 또는 그 이하로 근접하면 엔캐리 트레이드의 청산(unwind)이 가속화될 수 있다는 신호로 해석되기도 합니다.
    - ※요약:
        - **금리 차이 확대(↑)**: 일본 금리는 여전히 낮고, 미국 금리는 높음 → 엔캐리 트레이드 유지 → 미국 증시 **안정적**
        - **금리 차이 축소(↓)**: 일본 금리 상승 또는 미국 금리 하락 → 캐리 트레이드 축소 → 미국 증시 **조정 가능성 증가**
        - 특히 **스프레드가 1% 이하로 줄어들면** 리스크 자산 회피 신호로 볼 수 있음

    ---
    **⚠️ 데이터 빈도 참고사항:**
    - 미국 10년물 국채 금리는 **일별(Daily) 데이터**입니다.
    - 일본 10년물 국채 금리 데이터는 FRED에서 **월별(Monthly) 기준**으로 제공됩니다.
    - 따라서 그래프 상에서 일본 금리 데이터는 해당 월의 첫 영업일에만 업데이트되는 것처럼 보일 수 있으며, 금리 스프레드 역시 월별 데이터가 존재하는 날짜에만 계산됩니다.
    """)
