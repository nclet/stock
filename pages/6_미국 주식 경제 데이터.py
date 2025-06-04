import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
# yfinance는 더 이상 사용하지 않으므로 제거합니다.
from datetime import datetime, timedelta
import traceback # 오류 스택 추적을 위해 임포트

# --- 설정 ---
st.set_page_config(page_title="미국-일본 10년물 금리차 및 경제 지표 대시보드", layout="wide")

# FRED API 키를 st.secrets에서 불러옵니다.
# secrets.toml 파일에 FRED_API_KEY = "YOUR_KEY" 형태로 저장되어 있어야 합니다.
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    st.error("🚨 FRED API 키('FRED_API_KEY')가 Streamlit Secrets에 설정되어 있지 않습니다.")
    st.info("Streamlit Cloud 대시보드의 'Settings' -> 'Secrets' 메뉴에서 'FRED_API_KEY'를 설정해주세요.")
    st.stop() # API 키 없으면 앱 실행 중지

fred = Fred(api_key=FRED_API_KEY)

# --- 데이터 불러오기 함수 (기존 금리 스프레드 데이터) ---
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
        st.warning("선택된 기간에 유효한 금리 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 금리 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
    return df

# --- 새로운 데이터 불러오기 함수 (CPI, 고용 지표) ---
@st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
def load_economic_indicators(start_date, end_date):
    econ_data = {}
    econ_errors = []

    # 1. 소비자물가지수 (CPIAUCSL) - 월별
    st.info("🔄 소비자물가지수(CPI) 데이터를 불러오는 중... (FRED: 월별 데이터)")
    try:
        cpi = fred.get_series('CPIAUCSL', start_date, end_date)
        if cpi is None or cpi.empty:
            econ_errors.append("❌ 소비자물가지수(CPI) 데이터 로드 실패: 'CPIAUCSL'. 기간을 조정해 보세요.")
        else:
            econ_data['CPI'] = cpi.rename("CPI")
    except Exception as e:
        econ_errors.append(f"❌ 소비자물가지수(CPI) 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

    # 2. 실업률 (UNRATE) - 월별
    st.info("🔄 실업률 데이터를 불러오는 중... (FRED: 월별 데이터)")
    try:
        unemployment_rate = fred.get_series('UNRATE', start_date, end_date)
        if unemployment_rate is None or unemployment_rate.empty:
            econ_errors.append("❌ 실업률 데이터 로드 실패: 'UNRATE'. 기간을 조정해 보세요.")
        else:
            econ_data['Unemployment_Rate'] = unemployment_rate.rename("Unemployment_Rate")
    except Exception as e:
        econ_errors.append(f"❌ 실업률 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

    # 3. 비농업 고용자 수 (PAYEMS) - 월별
    st.info("🔄 비농업 고용자 수 데이터를 불러오는 중... (FRED: 월별 데이터)")
    try:
        nonfarm_payrolls = fred.get_series('PAYEMS', start_date, end_date)
        if nonfarm_payrolls is None or nonfarm_payrolls.empty:
            econ_errors.append("❌ 비농업 고용자 수 데이터 로드 실패: 'PAYEMS'. 기간을 조정해 보세요.")
        else:
            econ_data['Nonfarm_Payrolls'] = nonfarm_payrolls.rename("Nonfarm_Payrolls")
    except Exception as e:
        econ_errors.append(f"❌ 비농업 고용자 수 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

    if econ_errors:
        for err in econ_errors:
            st.error(err)
        st.warning("일부 경제 지표 데이터 로드에 실패했습니다. 해당 그래프가 올바르게 표시되지 않을 수 있습니다.")
        return pd.DataFrame() # 오류가 있다면 빈 DataFrame 반환

    # 모든 데이터를 하나의 DataFrame으로 합치기
    econ_df = pd.DataFrame()
    for key, series in econ_data.items():
        if not series.empty:
            econ_df = pd.concat([econ_df, series], axis=1)

    # 인덱스를 datetime 형식으로 통일
    econ_df.index = pd.to_datetime(econ_df.index)
    
    # NaN 값 제거
    econ_df = econ_df.dropna(how='any')

    if econ_df.empty:
        st.warning("선택된 기간에 유효한 경제 지표 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 경제 지표 데이터 로드 완료! ({econ_df.index.min().date()} ~ {econ_df.index.max().date()})")
    return econ_df

# --- 날짜 선택 ---
st.sidebar.title("📅 기간 선택")
start_date_bond = st.sidebar.date_input("금리 데이터 시작일", datetime.today() - timedelta(days=365 * 5), key='bond_start') # 기본 기간을 5년으로 늘림
end_date_bond = st.sidebar.date_input("금리 데이터 종료일", datetime.today(), key='bond_end')

st.sidebar.markdown("---") # 구분선 추가

start_date_econ = st.sidebar.date_input("경제 지표 시작일", datetime.today() - timedelta(days=365 * 10), key='econ_start') # 기본 기간을 10년으로 설정
end_date_econ = st.sidebar.date_input("경제 지표 종료일", datetime.today(), key='econ_end')


# --- 데이터 불러오기 ---
with st.spinner("📊 금리 데이터를 불러오는 중..."):
    df_bond = load_yield_data(start_date_bond, end_date_bond)

with st.spinner("📊 경제 지표 데이터를 불러오는 중..."):
    df_econ = load_economic_indicators(start_date_econ, end_date_econ)


# --- 시각화 ---
if not df_bond.empty:
    st.title("🇺🇸 미국·🇯🇵 일본 10년 국채 금리 및 스프레드")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("미국 vs 일본 10년물 금리")
        fig1, ax1 = plt.subplots(figsize=(10, 6)) # figsize 추가
        df_bond["US_10Y"].plot(ax=ax1, label="AMERICA 10Y", color="blue", linewidth=1.5)
        df_bond["JP_10Y"].plot(ax=ax1, label="JAPAN 10Y", color="red", linewidth=1.5) # 월별임을 명시
        ax1.set_ylabel("Interest rate(%)")
        ax1.set_title("U.S, Japan 10-year bond spread") # 제목 추가
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7) # 그리드 추가
        st.pyplot(fig1)

    with col2:
        st.subheader("🇺🇸-🇯🇵 금리 스프레드")
        fig2, ax2 = plt.subplots(figsize=(10, 6)) # figsize 추가
        df_bond["Spread"].plot(ax=ax2, color="green", linewidth=2)
        ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax2.set_ylabel("Interest rate(%)")
        ax2.set_title("U.S.-Japan 10-year interest rate spread") # 제목 추가
        ax2.grid(True, linestyle='--', alpha=0.7) # 그리드 추가
        st.pyplot(fig2)
    
else:
    st.warning("금리 데이터를 불러오지 못했거나 선택된 기간에 유효한 데이터가 없습니다. 날짜 범위를 조정해 보세요.")

# --- 금리 스프레드 해석 도움말 ---
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


# 경제 지표 분석

if not df_econ.empty:
    st.title("📈 미국 주요 경제 지표 (물가 & 고용) 추이")
    st.markdown("경제 활동의 건전성과 연준의 통화 정책 방향성을 엿볼 수 있는 핵심 지표들입니다.")

    # 1. 소비자물가지수 (CPI)
    st.subheader("1. 소비자물가지수 (CPI) 추이")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    df_econ["CPI"].plot(ax=ax3, color="orange", linewidth=2)
    ax3.set_ylabel("지수 (1982-84=100)")
    ax3.set_title("미국 소비자물가지수 (CPI, SA)")
    ax3.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig3)
    st.info("CPI는 소비자들이 구매하는 상품과 서비스의 평균 가격 변동을 측정합니다. 높은 CPI는 인플레이션 압력을 시사하며, 이는 연준의 금리 인상 가능성을 높여 주식 시장에 부정적일 수 있습니다.")

    # 2. 실업률
    st.subheader("2. 실업률 추이")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    df_econ["Unemployment_Rate"].plot(ax=ax4, color="purple", linewidth=2)
    ax4.set_ylabel("실업률 (%)")
    ax4.set_title("미국 실업률")
    ax4.grid(True, linestyle='--', alpha=0.7)
    # 실업률이 특정 수준 이하일 때 (예: 4% 이하) 경고 표시
    if df_econ["Unemployment_Rate"].min() < 4.0:
        ax4.axhspan(0, 4.0, color='red', alpha=0.1, label='낮은 실업률 (인플레이션 압력)')
        ax4.legend()
    st.pyplot(fig4)
    st.info("실업률은 경제 활동의 강도를 나타내는 핵심 지표입니다. 낮은 실업률은 경제가 건강하다는 신호이지만, 너무 낮으면 임금 상승과 인플레이션 압력으로 이어질 수 있습니다.")

    # 3. 비농업 고용자 수
    st.subheader("3. 비농업 고용자 수 추이 (월별 변화)")
    # 월별 변화량을 계산하여 시각화 (더 의미 있는 데이터)
    df_econ['Nonfarm_Payrolls_MoM_Change'] = df_econ['Nonfarm_Payrolls'].diff()
    
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    df_econ['Nonfarm_Payrolls_MoM_Change'].plot(ax=ax5, color="blue", linewidth=2)
    ax5.set_ylabel("월별 변화 (천 명)")
    ax5.set_title("미국 비농업 고용자 수 월별 변화")
    ax5.axhline(0, color="gray", linestyle="--", alpha=0.7) # 0선 표시
    ax5.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig5)
    st.info("비농업 고용자 수는 비농업 부문의 월별 고용 변화를 보여줍니다. 이 지표의 강세는 경제 성장과 소비 증가를 시사하지만, 예상치를 크게 상회하는 증가는 연준의 긴축 우려를 높일 수도 있습니다.")


else:
    st.warning("경제 지표 데이터를 불러오지 못했거나 선택된 기간에 유효한 데이터가 없습니다. 날짜 범위를 조정해 보세요.")

# --- 경제 지표 해석 도움말 ---
with st.expander("📖 경제 지표와 주식 시장 해석 가이드"):
    st.markdown("""
    #### 📈 경제 지표와 주식 시장 관계

    - **소비자물가지수 (CPI):**
        - **CPI 상승:** 인플레이션 압력 증가. 연준의 금리 인상 가능성↑. 기업 비용↑. 주식 시장에 **부정적** 영향.
        - **CPI 하락:** 인플레이션 압력 완화. 연준의 금리 인하 또는 동결 기대감↑. 기업 비용↓. 주식 시장에 **긍정적** 영향.
        - **연준 목표치 (2%)**와 비교하여 현재 인플레이션 수준을 파악하는 것이 중요합니다.

    - **실업률:**
        - **실업률 하락 (고용 증가):** 경제 활동 활발. 소비 증가 기대. 기업 이익 증가 가능성↑. 주식 시장에 **긍정적**.
        - **실업률 상승 (고용 감소):** 경기 둔화/침체 우려. 소비 위축. 기업 실적 악화 가능성↓. 주식 시장에 **부정적**.
        - **너무 낮은 실업률:** 과열된 고용 시장은 임금 상승을 유발하고 인플레이션 압력을 높여 연준의 긴축을 유도할 수 있습니다.

    - **비농업 고용자 수:**
        - **비농업 고용자 수 증가:** 경제 성장과 고용 시장의 강세를 나타냄. 주식 시장에 **긍정적**이지만, 너무 빠른 증가는 인플레이션 우려를 낳을 수 있음.
        - **비농업 고용자 수 감소:** 경기 둔화 또는 침체 신호. 주식 시장에 **부정적**.

    #### 📊 그래프 해석 포인트

    - **각 지표의 추세:** 과거 대비 현재 지표의 수준이 어떤지, 상승/하락 추세가 이어지는지 확인합니다.
    - **전월 대비 변화:** 특히 고용 지표의 경우, 절대적인 수치보다 전월 대비 변화량(MoM Change)이 시장의 기대치와 얼마나 다른지가 중요합니다.
    - **연준의 정책 방향:** 이들 지표는 연방준비제도(Fed)의 통화 정책 결정에 핵심적인 영향을 미칩니다. 금리 인상/인하 기대감과 지표의 변화를 함께 고려하여 시장 반응을 예측합니다.
    """)


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from fredapi import Fred
# # yfinance는 더 이상 사용하지 않으므로 제거합니다.
# from datetime import datetime, timedelta
# import traceback # 오류 스택 추적을 위해 임포트

# # --- 설정 ---
# st.set_page_config(page_title="미국-일본 10년물 금리차 대시보드", layout="wide")

# # FRED API 키를 st.secrets에서 불러옵니다.
# # secrets.toml 파일에 FRED_API_KEY = "YOUR_KEY" 형태로 저장되어 있어야 합니다.
# try:
#     FRED_API_KEY = st.secrets["FRED_API_KEY"]
# except KeyError:
#     st.error("🚨 FRED API 키('FRED_API_KEY')가 Streamlit Secrets에 설정되어 있지 않습니다.")
#     st.info("Streamlit Cloud 대시보드의 'Settings' -> 'Secrets' 메뉴에서 'FRED_API_KEY'를 설정해주세요.")
#     st.stop() # API 키 없으면 앱 실행 중지

# fred = Fred(api_key=FRED_API_KEY)

# # --- 데이터 불러오기 함수 ---
# @st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
# def load_yield_data(start_date, end_date):
#     data = {}
#     errors = []

#     # 1. 미국 10년물 국채 금리 (일별)
#     st.info("🔄 미국 10년물 국채 금리 데이터를 불러오는 중...")
#     try:
#         us_10y = fred.get_series('GS10', start_date, end_date)
#         if us_10y is None or us_10y.empty:
#             errors.append("❌ 미국 10년물 금리 데이터 로드 실패: 'GS10'. 기간을 조정해 보세요.")
#         else:
#             data['US_10Y'] = us_10y.rename("US_10Y")
#     except Exception as e:
#         errors.append(f"❌ 미국 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")


#     # 2. 일본 10년물 국채 금리 (월별) - FRED에서 가져오도록 수정
#     st.info("🔄 일본 10년물 국채 금리 데이터를 불러오는 중... (FRED: 월별 데이터)")
#     try:
#         # 'IRLTLT01JPM156N': OECD Long-Term Interest Rate: 10-Year Government Bonds for Japan, Monthly
#         jgb_10y = fred.get_series('IRLTLT01JPM156N', start_date, end_date) 
#         if jgb_10y is None or jgb_10y.empty:
#             errors.append("❌ 일본 10년물 금리 데이터 로드 실패: 'IRLTLT01JPM156N'. 기간을 조정해 보세요.")
#             st.info("참고: FRED에서 제공하는 일본 10년물 국채 금리 데이터는 월별입니다.")
#         else:
#             data['JP_10Y'] = jgb_10y.rename("JP_10Y")
#     except Exception as e:
#         errors.append(f"❌ 일본 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")
    
#     # S&P 500 지수 데이터 수집 부분은 완전히 제거되었습니다.

#     if errors:
#         for err in errors:
#             st.error(err)
#         st.warning("일부 데이터 로드에 실패했습니다. 그래프가 올바르게 표시되지 않을 수 있습니다.")
#         return pd.DataFrame() # 오류가 있다면 빈 DataFrame 반환

#     # 모든 데이터프레임을 하나의 DataFrame으로 합치기
#     df = pd.DataFrame()
#     for key, series in data.items():
#         if not series.empty:
#             df = pd.concat([df, series.resample('D').mean()], axis=1) # 월별 데이터는 해당 월의 모든 일자에 해당 월의 값 적용
    
#     # 인덱스를 datetime 형식으로 통일
#     df.index = pd.to_datetime(df.index)
    
#     # 10년물 스프레드 계산
#     df["Spread"] = df["US_10Y"] - df["JP_10Y"]
    
#     # 최종적으로 필요한 컬럼만 남기고 NaN 값 제거
#     df = df.dropna(subset=['US_10Y', 'JP_10Y', 'Spread'], how='any')

#     if df.empty:
#         st.warning("선택된 기간에 유효한 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
#         return pd.DataFrame()

#     st.success(f"✅ 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
#     return df

# # --- 날짜 선택 ---
# st.sidebar.title("📅 기간 선택")
# start_date = st.sidebar.date_input("시작일", datetime.today() - timedelta(days=365 * 5)) # 기본 기간을 5년으로 늘림
# end_date = st.sidebar.date_input("종료일", datetime.today())

# # --- 데이터 불러오기 ---
# with st.spinner("📊 데이터를 불러오는 중... (FRED 사용)"): # YFinance 사용 문구 제거
#     df = load_yield_data(start_date, end_date)

# # --- 시각화 ---
# if not df.empty:
#     st.title("미국·일본 10년 국채 금리 및 스프레드")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("미국 vs 일본 10년물 금리")
#         fig1, ax1 = plt.subplots(figsize=(10, 6)) # figsize 추가
#         df["US_10Y"].plot(ax=ax1, label="AMERICA 10Y", color="blue", linewidth=1.5)
#         df["JP_10Y"].plot(ax=ax1, label="JAPAN 10Y", color="red", linewidth=1.5) # 월별임을 명시
#         ax1.set_ylabel("Interest rate(%)")
#         ax1.set_title("U.S, Japan 10-year bond spread") # 제목 추가
#         ax1.legend()
#         ax1.grid(True, linestyle='--', alpha=0.7) # 그리드 추가
#         st.pyplot(fig1)

#     with col2:
#         st.subheader("🇺🇸-🇯🇵 금리 스프레드")
#         fig2, ax2 = plt.subplots(figsize=(10, 6)) # figsize 추가
#         df["Spread"].plot(ax=ax2, color="green", linewidth=2)
#         ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
#         ax2.set_ylabel("Interest rate(%)")
#         ax2.set_title("U.S.-Japan 10-year interest rate spread") # 제목 추가
#         ax2.grid(True, linestyle='--', alpha=0.7) # 그리드 추가
#         st.pyplot(fig2)
    

# else:
#     st.warning("데이터를 불러오지 못했거나 선택된 기간에 유효한 데이터가 없습니다. 날짜 범위를 조정해 보세요.")

# # --- 해석 도움말 ---
# with st.expander("📖 금리 스프레드 해석 가이드"):
#     st.markdown("""
#     - **엔캐리 트레이드(Yen Carry Trade)**: 일본의 낮은 금리(낮은 대출 비용)를 활용하여 엔화를 빌린 후, 이 자금으로 미국 등 금리가 높은 국가의 자산(주식, 채권)에 투자하여 금리 차이(스프레드)만큼 수익을 추구하는 전략입니다.

#     - **금리 차이 확대 (스프레드 상승):**
#         - 미국 금리 > 일본 금리 (금리 차이 확대)
#         - 엔캐리 트레이드 유지 또는 활성화 → 엔화 매도, 달러/미국 자산 매수 → 미국 증시 **긍정적** 영향
#         - 그래프에서 **초록색 선(스프레드)이 상승**하는 시기.

#     - **금리 차이 축소 (스프레드 하락):**
#         - 일본 금리 상승 또는 미국 금리 하락 (금리 차이 축소)
#         - 엔캐리 트레이드의 수익성이 줄어들거나 손실 위험 → 엔화를 되갚기 위해 미국 자산 매도 → 엔화 매수 → 미국 증시 **조정 또는 하락 압력**
#         - 그래프에서 **초록색 선(스프레드)이 하락**하는 시기. 특히 0% 또는 그 이하로 근접하면 엔캐리 트레이드의 청산(unwind)이 가속화될 수 있다는 신호로 해석되기도 합니다.
#     - ※요약:
#         - **금리 차이 확대(↑)**: 일본 금리는 여전히 낮고, 미국 금리는 높음 → 엔캐리 트레이드 유지 → 미국 증시 **안정적**
#         - **금리 차이 축소(↓)**: 일본 금리 상승 또는 미국 금리 하락 → 캐리 트레이드 축소 → 미국 증시 **조정 가능성 증가**
#         - 특히 **스프레드가 1% 이하로 줄어들면** 리스크 자산 회피 신호로 볼 수 있음

#     ---
#     **⚠️ 데이터 빈도 참고사항:**
#     - 미국 10년물 국채 금리는 **일별(Daily) 데이터**입니다.
#     - 일본 10년물 국채 금리 데이터는 FRED에서 **월별(Monthly) 기준**으로 제공됩니다.
#     - 따라서 그래프 상에서 일본 금리 데이터는 해당 월의 첫 영업일에만 업데이트되는 것처럼 보일 수 있으며, 금리 스프레드 역시 월별 데이터가 존재하는 날짜에만 계산됩니다.
#     """)
