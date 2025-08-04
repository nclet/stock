import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime, timedelta
import traceback # 오류 스택 추적을 위해 임포트
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import urllib.error # HTTPError를 위해 임포트

# --- 설정 ---
st.set_page_config(page_title="미국-일본 10년물 금리차 및 경제 지표 대시보드", layout="wide")

# FRED API 키를 st.secrets에서 불러옵니다.
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    st.error("🚨 FRED API 키('FRED_API_KEY')가 Streamlit Secrets에 설정되어 있지 않습니다.")
    st.info("Streamlit Cloud 대시보드의 'Settings' -> 'Secrets' 메뉴에서 'FRED_API_KEY'를 설정해주세요.")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

# --- 재시도 데코레이터 설정 ---
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((urllib.error.HTTPError, ConnectionResetError)),
    reraise=True
)
def fetch_fred_series_with_retry(series_id, start_date, end_date):
    """
    FRED API에서 데이터를 가져오는 함수에 재시도 로직을 추가합니다.
    """
    return fred.get_series(series_id, start_date, end_date)


# --- 데이터 불러오기 함수 (기존 금리 스프레드 데이터) ---
@st.cache_data(ttl=3600)
def load_yield_data(start_date, end_date):
    data = {}
    errors = []

    try:
        us_10y = fetch_fred_series_with_retry('GS10', start_date, end_date)
        if us_10y is None or us_10y.empty:
            errors.append("❌ 미국 10년물 금리 데이터 로드 실패: 'GS10'. 기간을 조정해 보세요.")
        else:
            data['US_10Y'] = us_10y.rename("US_10Y")
    except Exception as e:
        errors.append(f"❌ 미국 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

    try:
        jgb_10y = fetch_fred_series_with_retry('IRLTLT01JPM156N', start_date, end_date)
        if jgb_10y is None or jgb_10y.empty:
            errors.append("❌ 일본 10년물 금리 데이터 로드 실패: 'IRLTLT01JPM156N'. 기간을 조정해 보세요.")
            st.info("참고: FRED에서 제공하는 일본 10년물 국채 금리 데이터는 월별입니다.")
        else:
            data['JP_10Y'] = jgb_10y.rename("JP_10Y")
    except Exception as e:
        errors.append(f"❌ 일본 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

    if errors:
        for err in errors:
            st.error(err)
        st.warning("일부 데이터 로드에 실패했습니다. 그래프가 올바르게 표시되지 않을 수 있습니다.")
        return pd.DataFrame()

    df = pd.DataFrame()
    for key, series in data.items():
        if not series.empty:
            df = pd.concat([df, series], axis=1)

    df.index = pd.to_datetime(df.index)
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D'))
    df['JP_10Y'] = df['JP_10Y'].ffill()

    df["Spread"] = df["US_10Y"] - df["JP_10Y"]
    df = df.dropna(subset=['US_10Y', 'JP_10Y', 'Spread'], how='any')

    if df.empty:
        st.warning("선택된 기간에 유효한 금리 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 금리 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
    return df

# --- 새로운 데이터 불러오기 함수 (CPI, 고용 지표) ---
@st.cache_data(ttl=3600)
def load_economic_indicators(start_date, end_date):
    econ_data = {}
    econ_errors = []

    try:
        cpi = fetch_fred_series_with_retry('CPIAUCSL', start_date, end_date)
        if cpi is None or cpi.empty:
            econ_errors.append("❌ 소비자물가지수(CPI) 데이터 로드 실패: 'CPIAUCSL'. 기간을 조정해 보세요.")
        else:
            econ_data['CPI'] = cpi.rename("CPI")
    except Exception as e:
        econ_errors.append(f"❌ 소비자물가지수(CPI) 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

    try:
        unemployment_rate = fetch_fred_series_with_retry('UNRATE', start_date, end_date)
        if unemployment_rate is None or unemployment_rate.empty:
            econ_errors.append("❌ 실업률 데이터 로드 실패: 'UNRATE'. 기간을 조정해 보세요.")
        else:
            econ_data['Unemployment_Rate'] = unemployment_rate.rename("Unemployment_Rate")
    except Exception as e:
        econ_errors.append(f"❌ 실업률 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

    try:
        nonfarm_payrolls = fetch_fred_series_with_retry('PAYEMS', start_date, end_date)
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
        return pd.DataFrame()

    econ_df = pd.DataFrame()
    for key, series in econ_data.items():
        if not series.empty:
            econ_df = pd.concat([econ_df, series], axis=1)

    econ_df.index = pd.to_datetime(econ_df.index)
    econ_df = econ_df.dropna(how='any')

    if econ_df.empty:
        st.warning("선택된 기간에 유효한 경제 지표 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 경제 지표 데이터 로드 완료! ({econ_df.index.min().date()} ~ {econ_df.index.max().date()})")
    return econ_df


# --- 날짜 선택 (본문으로 이동) ---
st.header("📅 데이터 기간 설정")

# 국채 금리 기간 설정
st.subheader("국채 금리 기간")
col1_date, col2_date = st.columns(2)
with col1_date:
    start_date_bond = st.date_input("시작일", datetime.today() - timedelta(days=365 * 5), key='bond_start')
with col2_date:
    end_date_bond = st.date_input("종료일", datetime.today(), key='bond_end')

st.markdown("---")

# CPI, 실업률, 고용률 기간 설정
st.subheader("CPI, 실업률, 고용률 기간")
col3_date, col4_date = st.columns(2)
with col3_date:
    start_date_econ = st.date_input("시작일", datetime.today() - timedelta(days=365 * 10), key='econ_start')
with col4_date:
    end_date_econ = st.date_input("종료일", datetime.today(), key='econ_end')


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
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        df_bond["US_10Y"].plot(ax=ax1, label="AMERICA 10Y", color="blue", linewidth=1.5)
        df_bond["JP_10Y"].plot(ax=ax1, label="JAPAN 10Y", color="red", linewidth=1.5)
        ax1.set_ylabel("Interest rate(%)")
        ax1.set_title("U.S, Japan 10-year bond spread")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)

    with col2:
        st.subheader("🇺🇸-🇯🇵 금리 스프레드")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df_bond["Spread"].plot(ax=ax2, color="green", linewidth=2)
        ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax2.set_ylabel("Interest rate(%)")
        ax2.set_title("U.S.-Japan 10-year interest rate spread")
        ax2.grid(True, linestyle='--', alpha=0.7)
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
    ax3.set_ylabel("Index (1982-84=100)")
    ax3.set_title("U.S. Consumer Price Index (CPI)")
    ax3.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig3)
    st.info("CPI는 소비자들이 구매하는 상품과 서비스의 평균 가격 변동을 측정합니다. 높은 CPI는 인플레이션 압력을 시사하며, 이는 연준의 금리 인상 가능성을 높여 주식 시장에 부정적일 수 있습니다.")

    # 2. 실업률
    st.subheader("2. 실업률 추이")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    df_econ["Unemployment_Rate"].plot(ax=ax4, color="purple", linewidth=2)
    ax4.set_ylabel("unemployment rate (%)")
    ax4.set_title("U.S unemployment rate")
    ax4.grid(True, linestyle='--', alpha=0.7)
    if df_econ["Unemployment_Rate"].min() < 4.0:
        ax4.axhspan(0, 4.0, color='red', alpha=0.1, label='Low unemployment (inflationary pressure)')
        ax4.legend()
    st.pyplot(fig4)
    st.info("실업률은 경제 활동의 강도를 나타내는 핵심 지표입니다. 낮은 실업률은 경제가 건강하다는 신호이지만, 너무 낮으면 임금 상승과 인플레이션 압력으로 이어질 수 있습니다.")

    # 3. 비농업 고용자 수
    st.subheader("3. 비농업 고용자 수 추이 (월별 변화)")
    df_econ['Nonfarm_Payrolls_MoM_Change'] = df_econ['Nonfarm_Payrolls'].diff()

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    df_econ['Nonfarm_Payrolls_MoM_Change'].plot(ax=ax5, color="blue", linewidth=2)
    ax5.set_ylabel("Monthly Changes (Thousands)")
    ax5.set_title("Monthly Changes in the Number of Nonfarm Employees in the U.S")
    ax5.axhline(0, color="gray", linestyle="--", alpha=0.7)
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
# from datetime import datetime, timedelta
# import traceback # 오류 스택 추적을 위해 임포트
# from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
# import urllib.error # HTTPError를 위해 임포트

# # --- 설정 ---
# st.set_page_config(page_title="미국-일본 10년물 금리차 및 경제 지표 대시보드", layout="wide")

# # FRED API 키를 st.secrets에서 불러옵니다.
# try:
#     FRED_API_KEY = st.secrets["FRED_API_KEY"]
# except KeyError:
#     st.error("🚨 FRED API 키('FRED_API_KEY')가 Streamlit Secrets에 설정되어 있지 않습니다.")
#     st.info("Streamlit Cloud 대시보드의 'Settings' -> 'Secrets' 메뉴에서 'FRED_API_KEY'를 설정해주세요.")
#     st.stop()

# fred = Fred(api_key=FRED_API_KEY)

# # --- 재시도 데코레이터 설정 ---
# # HTTP 403 Forbidden 오류 또는 ConnectionResetError 발생 시 재시도
# # 처음 1초 대기 후, 매 재시도마다 대기 시간이 기하급수적으로 증가 (최대 10초)
# # 최대 3번까지 재시도
# @retry(
#     wait=wait_exponential(multiplier=1, min=1, max=10),
#     stop=stop_after_attempt(3),
#     retry=retry_if_exception_type((urllib.error.HTTPError, ConnectionResetError)),
#     reraise=True # 마지막 시도까지 실패하면 예외를 다시 발생시킴
# )
# def fetch_fred_series_with_retry(series_id, start_date, end_date):
#     """
#     FRED API에서 데이터를 가져오는 함수에 재시도 로직을 추가합니다.
#     """
#     return fred.get_series(series_id, start_date, end_date)


# # --- 데이터 불러오기 함수 (기존 금리 스프레드 데이터) ---
# @st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
# def load_yield_data(start_date, end_date):
#     data = {}
#     errors = []

#     # 1. 미국 10년물 국채 금리 (일별)
#     try:
#         us_10y = fetch_fred_series_with_retry('GS10', start_date, end_date)
#         if us_10y is None or us_10y.empty:
#             errors.append("❌ 미국 10년물 금리 데이터 로드 실패: 'GS10'. 기간을 조정해 보세요.")
#         else:
#             data['US_10Y'] = us_10y.rename("US_10Y")
#     except Exception as e:
#         errors.append(f"❌ 미국 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")


#     # 2. 일본 10년물 국채 금리 (월별) - FRED에서 가져오도록 수정
#     try:
#         jgb_10y = fetch_fred_series_with_retry('IRLTLT01JPM156N', start_date, end_date)
#         if jgb_10y is None or jgb_10y.empty:
#             errors.append("❌ 일본 10년물 금리 데이터 로드 실패: 'IRLTLT01JPM156N'. 기간을 조정해 보세요.")
#             st.info("참고: FRED에서 제공하는 일본 10년물 국채 금리 데이터는 월별입니다.")
#         else:
#             data['JP_10Y'] = jgb_10y.rename("JP_10Y")
#     except Exception as e:
#         errors.append(f"❌ 일본 10년물 금리 데이터 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

#     if errors:
#         for err in errors:
#             st.error(err)
#         st.warning("일부 데이터 로드에 실패했습니다. 그래프가 올바르게 표시되지 않을 수 있습니다.")
#         return pd.DataFrame()

#     df = pd.DataFrame()
#     for key, series in data.items():
#         if not series.empty:
#             # 월별 데이터는 해당 월의 모든 일자에 해당 월의 값 적용 (resample('D').ffill() 또는 .asfreq('D').ffill())
#             # 여기서는 .resample('D').mean() 대신 .asfreq('D', method='ffill')이 더 적합할 수 있습니다.
#             # 하지만 이미 concat 이후 .dropna를 사용하고 있으므로, 일별 데이터와 월별 데이터를 합치는 과정에서
#             # 월별 데이터의 경우 해당 월의 첫 날에만 값이 있고 나머지는 NaN이 됩니다.
#             # 이후 .dropna에서 이 NaN들이 제거되므로, 월별 데이터가 일별 데이터와 병합될 때
#             # 실제 존재하는 날짜에만 값이 남게 됩니다. 이 부분은 사용 목적에 따라 적절히 조절해야 합니다.
#             # 일단 기존 로직을 유지하고, 필요에 따라 보간법을 고려할 수 있습니다.
#             df = pd.concat([df, series], axis=1) # resample 대신 바로 concat 후 날짜 인덱스 처리


#     df.index = pd.to_datetime(df.index)
#     df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D'))
#     df['JP_10Y'] = df['JP_10Y'].ffill() # 월별 데이터를 일별 데이터에 맞춰 채움

#     df["Spread"] = df["US_10Y"] - df["JP_10Y"]
#     df = df.dropna(subset=['US_10Y', 'JP_10Y', 'Spread'], how='any')

#     if df.empty:
#         st.warning("선택된 기간에 유효한 금리 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
#         return pd.DataFrame()

#     st.success(f"✅ 금리 데이터 로드 완료! ({df.index.min().date()} ~ {df.index.max().date()})")
#     return df

# # --- 새로운 데이터 불러오기 함수 (CPI, 고용 지표) ---
# @st.cache_data(ttl=3600) # 데이터를 1시간(3600초) 동안 캐시
# def load_economic_indicators(start_date, end_date):
#     econ_data = {}
#     econ_errors = []

#     # 1. 소비자물가지수 (CPIAUCSL) - 월별
#     try:
#         cpi = fetch_fred_series_with_retry('CPIAUCSL', start_date, end_date)
#         if cpi is None or cpi.empty:
#             econ_errors.append("❌ 소비자물가지수(CPI) 데이터 로드 실패: 'CPIAUCSL'. 기간을 조정해 보세요.")
#         else:
#             econ_data['CPI'] = cpi.rename("CPI")
#     except Exception as e:
#         econ_errors.append(f"❌ 소비자물가지수(CPI) 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

#     # 2. 실업률 (UNRATE) - 월별
#     try:
#         unemployment_rate = fetch_fred_series_with_retry('UNRATE', start_date, end_date)
#         if unemployment_rate is None or unemployment_rate.empty:
#             econ_errors.append("❌ 실업률 데이터 로드 실패: 'UNRATE'. 기간을 조정해 보세요.")
#         else:
#             econ_data['Unemployment_Rate'] = unemployment_rate.rename("Unemployment_Rate")
#     except Exception as e:
#         econ_errors.append(f"❌ 실업률 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

#     # 3. 비농업 고용자 수 (PAYEMS) - 월별
#     try:
#         nonfarm_payrolls = fetch_fred_series_with_retry('PAYEMS', start_date, end_date)
#         if nonfarm_payrolls is None or nonfarm_payrolls.empty:
#             econ_errors.append("❌ 비농업 고용자 수 데이터 로드 실패: 'PAYEMS'. 기간을 조정해 보세요.")
#         else:
#             econ_data['Nonfarm_Payrolls'] = nonfarm_payrolls.rename("Nonfarm_Payrolls")
#     except Exception as e:
#         econ_errors.append(f"❌ 비농업 고용자 수 로드 중 오류 발생: {e}. Traceback: {traceback.format_exc()}")

#     if econ_errors:
#         for err in econ_errors:
#             st.error(err)
#         st.warning("일부 경제 지표 데이터 로드에 실패했습니다. 해당 그래프가 올바르게 표시되지 않을 수 있습니다.")
#         return pd.DataFrame()

#     econ_df = pd.DataFrame()
#     for key, series in econ_data.items():
#         if not series.empty:
#             econ_df = pd.concat([econ_df, series], axis=1)

#     econ_df.index = pd.to_datetime(econ_df.index)
#     econ_df = econ_df.dropna(how='any')

#     if econ_df.empty:
#         st.warning("선택된 기간에 유효한 경제 지표 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
#         return pd.DataFrame()

#     st.success(f"✅ 경제 지표 데이터 로드 완료! ({econ_df.index.min().date()} ~ {econ_df.index.max().date()})")
#     return econ_df

# # --- 날짜 선택 ---
# st.sidebar.title("📅 국채 금리 설정")
# start_date_bond = st.sidebar.date_input("금리 데이터 시작일", datetime.today() - timedelta(days=365 * 5), key='bond_start')
# end_date_bond = st.sidebar.date_input("금리 데이터 종료일", datetime.today(), key='bond_end')

# st.sidebar.markdown("---")

# st.sidebar.title("📅 CPI, 실업률, 고용률 기간 설정")
# start_date_econ = st.sidebar.date_input("경제 지표 시작일", datetime.today() - timedelta(days=365 * 10), key='econ_start')
# end_date_econ = st.sidebar.date_input("경제 지표 종료일", datetime.today(), key='econ_end')


# # --- 데이터 불러오기 ---
# with st.spinner("📊 금리 데이터를 불러오는 중..."):
#     df_bond = load_yield_data(start_date_bond, end_date_bond)

# with st.spinner("📊 경제 지표 데이터를 불러오는 중..."):
#     df_econ = load_economic_indicators(start_date_econ, end_date_econ)


# # --- 시각화 ---
# if not df_bond.empty:
#     st.title("🇺🇸 미국·🇯🇵 일본 10년 국채 금리 및 스프레드")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("미국 vs 일본 10년물 금리")
#         fig1, ax1 = plt.subplots(figsize=(10, 6))
#         df_bond["US_10Y"].plot(ax=ax1, label="AMERICA 10Y", color="blue", linewidth=1.5)
#         df_bond["JP_10Y"].plot(ax=ax1, label="JAPAN 10Y", color="red", linewidth=1.5)
#         ax1.set_ylabel("Interest rate(%)")
#         ax1.set_title("U.S, Japan 10-year bond spread")
#         ax1.legend()
#         ax1.grid(True, linestyle='--', alpha=0.7)
#         st.pyplot(fig1)

#     with col2:
#         st.subheader("🇺🇸-🇯🇵 금리 스프레드")
#         fig2, ax2 = plt.subplots(figsize=(10, 6))
#         df_bond["Spread"].plot(ax=ax2, color="green", linewidth=2)
#         ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
#         ax2.set_ylabel("Interest rate(%)")
#         ax2.set_title("U.S.-Japan 10-year interest rate spread")
#         ax2.grid(True, linestyle='--', alpha=0.7)
#         st.pyplot(fig2)

# else:
#     st.warning("금리 데이터를 불러오지 못했거나 선택된 기간에 유효한 데이터가 없습니다. 날짜 범위를 조정해 보세요.")

# # --- 금리 스프레드 해석 도움말 ---
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


# # 경제 지표 분석

# if not df_econ.empty:
#     st.title("📈 미국 주요 경제 지표 (물가 & 고용) 추이")
#     st.markdown("경제 활동의 건전성과 연준의 통화 정책 방향성을 엿볼 수 있는 핵심 지표들입니다.")

#     # 1. 소비자물가지수 (CPI)
#     st.subheader("1. 소비자물가지수 (CPI) 추이")
#     fig3, ax3 = plt.subplots(figsize=(12, 6))
#     df_econ["CPI"].plot(ax=ax3, color="orange", linewidth=2)
#     ax3.set_ylabel("Index (1982-84=100)")
#     ax3.set_title("U.S. Consumer Price Index (CPI)")
#     ax3.grid(True, linestyle='--', alpha=0.7)
#     st.pyplot(fig3)
#     st.info("CPI는 소비자들이 구매하는 상품과 서비스의 평균 가격 변동을 측정합니다. 높은 CPI는 인플레이션 압력을 시사하며, 이는 연준의 금리 인상 가능성을 높여 주식 시장에 부정적일 수 있습니다.")

#     # 2. 실업률
#     st.subheader("2. 실업률 추이")
#     fig4, ax4 = plt.subplots(figsize=(12, 6))
#     df_econ["Unemployment_Rate"].plot(ax=ax4, color="purple", linewidth=2)
#     ax4.set_ylabel("unemployment rate (%)")
#     ax4.set_title("U.S unemployment rate")
#     ax4.grid(True, linestyle='--', alpha=0.7)
#     if df_econ["Unemployment_Rate"].min() < 4.0:
#         ax4.axhspan(0, 4.0, color='red', alpha=0.1, label='Low unemployment (inflationary pressure)')
#         ax4.legend()
#     st.pyplot(fig4)
#     st.info("실업률은 경제 활동의 강도를 나타내는 핵심 지표입니다. 낮은 실업률은 경제가 건강하다는 신호이지만, 너무 낮으면 임금 상승과 인플레이션 압력으로 이어질 수 있습니다.")

#     # 3. 비농업 고용자 수
#     st.subheader("3. 비농업 고용자 수 추이 (월별 변화)")
#     df_econ['Nonfarm_Payrolls_MoM_Change'] = df_econ['Nonfarm_Payrolls'].diff()

#     fig5, ax5 = plt.subplots(figsize=(12, 6))
#     df_econ['Nonfarm_Payrolls_MoM_Change'].plot(ax=ax5, color="blue", linewidth=2)
#     ax5.set_ylabel("Monthly Changes (Thousands)")
#     ax5.set_title("Monthly Changes in the Number of Nonfarm Employees in the U.S")
#     ax5.axhline(0, color="gray", linestyle="--", alpha=0.7)
#     ax5.grid(True, linestyle='--', alpha=0.7)
#     st.pyplot(fig5)
#     st.info("비농업 고용자 수는 비농업 부문의 월별 고용 변화를 보여줍니다. 이 지표의 강세는 경제 성장과 소비 증가를 시사하지만, 예상치를 크게 상회하는 증가는 연준의 긴축 우려를 높일 수도 있습니다.")

# else:
#     st.warning("경제 지표 데이터를 불러오지 못했거나 선택된 기간에 유효한 데이터가 없습니다. 날짜 범위를 조정해 보세요.")

# # --- 경제 지표 해석 도움말 ---
# with st.expander("📖 경제 지표와 주식 시장 해석 가이드"):
#     st.markdown("""
#     #### 📈 경제 지표와 주식 시장 관계

#     - **소비자물가지수 (CPI):**
#         - **CPI 상승:** 인플레이션 압력 증가. 연준의 금리 인상 가능성↑. 기업 비용↑. 주식 시장에 **부정적** 영향.
#         - **CPI 하락:** 인플레이션 압력 완화. 연준의 금리 인하 또는 동결 기대감↑. 기업 비용↓. 주식 시장에 **긍정적** 영향.
#         - **연준 목표치 (2%)**와 비교하여 현재 인플레이션 수준을 파악하는 것이 중요합니다.

#     - **실업률:**
#         - **실업률 하락 (고용 증가):** 경제 활동 활발. 소비 증가 기대. 기업 이익 증가 가능성↑. 주식 시장에 **긍정적**.
#         - **실업률 상승 (고용 감소):** 경기 둔화/침체 우려. 소비 위축. 기업 실적 악화 가능성↓. 주식 시장에 **부정적**.
#         - **너무 낮은 실업률:** 과열된 고용 시장은 임금 상승을 유발하고 인플레이션 압력을 높여 연준의 긴축을 유도할 수 있습니다.

#     - **비농업 고용자 수:**
#         - **비농업 고용자 수 증가:** 경제 성장과 고용 시장의 강세를 나타냄. 주식 시장에 **긍정적**이지만, 너무 빠른 증가는 인플레이션 우려를 낳을 수 있음.
#         - **비농업 고용자 수 감소:** 경기 둔화 또는 침체 신호. 주식 시장에 **부정적**.

#     #### 📊 그래프 해석 포인트

#     - **각 지표의 추세:** 과거 대비 현재 지표의 수준이 어떤지, 상승/하락 추세가 이어지는지 확인합니다.
#     - **전월 대비 변화:** 특히 고용 지표의 경우, 절대적인 수치보다 전월 대비 변화량(MoM Change)이 시장의 기대치와 얼마나 다른지가 중요합니다.
#     - **연준의 정책 방향:** 이들 지표는 연방준비제도(Fed)의 통화 정책 결정에 핵심적인 영향을 미칩니다. 금리 인상/인하 기대감과 지표의 변화를 함께 고려하여 시장 반응을 예측합니다.
#     """)

