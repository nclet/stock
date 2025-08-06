import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred # FRED API를 위한 라이브러리
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import urllib.error # HTTPError를 위해 임포트
import plotly.graph_objects as go
from plotly.subplots import make_subplots # make_subplots 임포트 추가
import time # 시간 지연을 위해 추가

# --- 페이지 설정 ---
st.set_page_config(page_title="미국 거시 경제 지표 시각화", layout="wide")
st.title("📊 미국 거시 경제 지표 추이 시각화 (테스트용)")

st.markdown("""
FRED API를 통해 미국 소비자물가지수(CPI)와 미국 국채 장단기 금리 스프레드 데이터를 가져와
시간에 따른 추이를 시각화하여 보여줍니다.
이 앱은 데이터 로드 및 시각화 테스트를 위해 암호화폐 관련 기능을 제외했습니다.
""")

# ------------------------
# ✨ 한글 폰트 설정 (matplotlib용 - 필요시 주석 해제)
# ------------------------
# Streamlit Cloud 환경에서 기본 폰트가 한글을 지원하지 않을 경우,
# 차트의 한글 텍스트가 깨져 보일 수 있습니다.
# 이 경우, Streamlit 앱 배포 환경에 한글 폰트를 설치하거나
# Plotly 등 다른 시각화 라이브러리를 고려할 수 있습니다.
plt.rc('axes', unicode_minus=False) # 마이너스 폰트 깨짐 방지 (일반적인 설정이므로 유지)

# ------------------------
# ✨ FRED API 설정
# ------------------------
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except KeyError:
    st.warning("🚨 FRED API 키('FRED_API_KEY')가 Streamlit Secrets에 설정되어 있지 않습니다. 거시 경제 지표는 로드되지 않습니다.")
    fred = None # FRED API 키가 없으면 fred 객체를 None으로 설정

# --- 재시도 데코레이터 설정 (FRED API용) ---
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((urllib.error.HTTPError, ConnectionResetError, ValueError)), # ValueError 추가
    reraise=True
)
def fetch_fred_series_with_retry(series_id, start_date, end_date):
    """
    FRED API에서 데이터를 가져오는 함수에 재시도 로직을 추가합니다.
    데이터가 없거나 비어있으면 ValueError를 발생시켜 재시도를 유도합니다.
    """
    if fred:
        series = fred.get_series(series_id, start_date, end_date)
        if series is None or series.empty:
            # 명시적으로 ValueError를 발생시켜 tenacity 재시도 유도
            raise ValueError(f"FRED series '{series_id}' returned no data for the period {start_date} to {end_date}.")
        return series
    return None # FRED 객체가 없을 경우

# ------------------------
# ✨ FRED 거시 경제 데이터 로드 함수
# ------------------------
@st.cache_data(ttl=3600)
def load_fred_macro_data(start_date, end_date):
    """
    FRED API에서 CPI, 미국 10년물 국채 금리, 미국 2년물 국채 금리 데이터를 가져와
    장단기 금리 스프레드를 계산합니다.
    """
    econ_data = {}
    econ_errors = []

    if not fred: # FRED API 키가 없으면 함수 종료
        return pd.DataFrame()

    st.info("🔄 FRED 거시 경제 데이터 수집 중...")

    # 1. 소비자물가지수 (CPIAUCSL) - 월별
    try:
        cpi = fetch_fred_series_with_retry('CPIAUCSL', start_date, end_date)
        econ_data['CPI'] = cpi.rename("CPI")
        st.info(f"✅ CPI 데이터 로드: {cpi.index.min().date()} ~ {cpi.index.max().date()}")
    except Exception as e:
        econ_errors.append(f"❌ 소비자물가지수(CPI) 로드 중 오류 발생: {e}")

    # 2. 미국 10년물 국채 금리 (GS10) - 일별
    try:
        us_10y = fetch_fred_series_with_retry('GS10', start_date, end_date)
        econ_data['US_10Y_Yield'] = us_10y.rename("US_10Y_Yield")
        st.info(f"✅ 미국 10년물 국채 금리 로드: {us_10y.index.min().date()} ~ {us_10y.index.max().date()}")
    except Exception as e:
        econ_errors.append(f"❌ 미국 10년물 국채 금리 로드 중 오류 발생: {e}")

    # 3. 미국 2년물 국채 금리 (GS2) - 일별
    try:
        us_2y = fetch_fred_series_with_retry('GS2', start_date, end_date)
        econ_data['US_2Y_Yield'] = us_2y.rename("US_2Y_Yield")
        st.info(f"✅ 미국 2년물 국채 금리 로드: {us_2y.index.min().date()} ~ {us_2y.index.max().date()}")
    except Exception as e:
        econ_errors.append(f"❌ 미국 2년물 국채 금리 로드 중 오류 발생: {e}")

    if econ_errors:
        for err in econ_errors:
            st.error(err)
        st.warning("일부 거시 경제 지표 데이터 로드에 실패했습니다. 해당 그래프가 올바르게 표시되지 않을 수 있습니다.")
        return pd.DataFrame()

    econ_df = pd.DataFrame()
    for key, series in econ_data.items():
        if not series.empty:
            econ_df = pd.concat([econ_df, series], axis=1)

    econ_df.index = pd.to_datetime(econ_df.index)
    # 월별 데이터를 일별 데이터로 채우기 (CPI)
    econ_df = econ_df.resample('D').ffill()
    econ_df = econ_df.dropna(how='all') # 모든 컬럼이 NaN인 행 제거

    # 장단기 금리 스프레드 계산
    if 'US_10Y_Yield' in econ_df.columns and 'US_2Y_Yield' in econ_df.columns:
        econ_df['US_Yield_Spread'] = econ_df['US_10Y_Yield'] - econ_df['US_2Y_Yield']
    else:
        st.warning("미국 국채 장단기 금리 스프레드를 계산할 수 없습니다. 10년물 또는 2년물 금리 데이터가 부족합니다.")
        econ_df['US_Yield_Spread'] = np.nan

    if econ_df.empty:
        st.warning("선택된 기간에 유효한 거시 경제 지표 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 거시 경제 지표 데이터 로드 완료! ({econ_df.index.min().date()} ~ {econ_df.index.max().date()})")
    return econ_df


# ------------------------
# ✨ 날짜 선택 UI
# ------------------------
st.header("데이터 기간 설정")

# 기본 시작 날짜를 1950년 1월 1일로 설정하여 FRED 데이터의 최대 가용 기간을 활용
default_end_date = datetime.today()
default_start_date = datetime(1950, 1, 1) # FRED 데이터의 시작점을 고려하여 매우 오래전으로 설정

start_date = st.date_input("데이터 시작 날짜", default_start_date)
end_date = st.date_input("데이터 종료 날짜", default_end_date)

if start_date >= end_date:
    st.error("❌ 종료 날짜는 시작 날짜보다 미래여야 합니다.")
    st.stop()

# ------------------------
# ✨ 시각화 실행 버튼
# ------------------------
if st.button("🚀 FRED 데이터 로드 및 시각화 실행"):
    with st.spinner("📊 FRED 거시 경제 데이터를 불러오는 중..."):
        df_fred = load_fred_macro_data(start_date, end_date)

    if not df_fred.empty:
        st.subheader("📊 미국 거시 경제 지표 추이")
        
        # --- 1. 소비자물가지수 (CPI) 추이 ---
        st.markdown("#### 📈 소비자물가지수 (CPI) 추이")
        if 'CPI' in df_fred.columns and not df_fred['CPI'].dropna().empty:
            fig_cpi = go.Figure()
            fig_cpi.add_trace(go.Scatter(x=df_fred.index, y=df_fred['CPI'],
                                         mode='lines', name='소비자물가지수 (CPI)', line=dict(color='orange')))
            fig_cpi.update_layout(height=400, title_text="소비자물가지수 (CPI)",
                                  xaxis_rangeslider_visible=True) # 범위 슬라이더 추가
            fig_cpi.update_xaxes(showgrid=True, tickangle=45)
            st.plotly_chart(fig_cpi, use_container_width=True)
        else:
            st.warning("CPI 데이터가 부족하여 시각화할 수 없습니다. 날짜 범위를 확인하거나 FRED API 키를 점검하세요.")
        
        st.markdown("---") # 구분선 추가

        # --- 2. 미국 국채 장단기 금리 스프레드 추이 ---
        st.markdown("#### 📈 미국 국채 장단기 금리 스프레드 (10년물 - 2년물) 추이")
        if 'US_Yield_Spread' in df_fred.columns and not df_fred['US_Yield_Spread'].dropna().empty:
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(x=df_fred.index, y=df_fred['US_Yield_Spread'],
                                            mode='lines', name='미국 국채 장단기 금리 스프레드', line=dict(color='green')))
            fig_spread.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="장단기 금리 역전 (0%)", annotation_position="top right")
            fig_spread.update_layout(height=400, title_text="미국 국채 장단기 금리 스프레드",
                                     xaxis_rangeslider_visible=True) # 범위 슬라이더 추가
            fig_spread.update_xaxes(showgrid=True, tickangle=45)
            st.plotly_chart(fig_spread, use_container_width=True)
        else:
            st.warning("미국 국채 장단기 금리 스프레드 데이터가 부족하여 시각화할 수 없습니다. 날짜 범위를 확인하거나 FRED API 키를 점검하세요.")

    else:
        st.error("FRED 거시 경제 데이터를 로드할 수 없어 시각화를 진행할 수 없습니다. FRED API 키를 확인하거나 날짜 범위를 조정해 보세요.")

    st.markdown("---")
    st.write("### 📝 참고 사항")
    st.write("""
    - **FRED API 키**: `.streamlit/secrets.toml` 파일에 `FRED_API_KEY = "YOUR_FRED_API_KEY"` 형식으로 FRED API 키를 설정해야 합니다.
    - **데이터 기간**: 이 앱은 FRED 데이터의 최대 가용 기간을 활용하기 위해 기본 시작 날짜를 1950년으로 설정했습니다. 원하는 기간으로 조정하여 데이터를 확인해 보세요.
    - **장단기 금리 역전**: 10년물 국채 금리에서 2년물 국채 금리를 뺀 값이 0보다 작아지면 (음수가 되면) '장단기 금리 역전'이라고 하며, 이는 종종 경기 침체의 전조로 해석되기도 합니다.
    """)
