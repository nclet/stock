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
st.set_page_config(page_title="암호화폐 vs. 미국 거시 지표", layout="wide")
st.title("📈 암호화폐 가격과 미국 거시 경제 지표 비교")

st.markdown("""
Upbit API를 통해 암호화폐 가격 데이터를 가져오고, FRED API를 통해 미국 소비자물가지수(CPI)와
미국 10년물 국채 금리 데이터를 가져와 함께 시각화하여 비교합니다.
데이터는 2020년 1월 1일을 기준으로 수집됩니다.
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

# ------------------------
# ✨ 암호화폐 종목 선택 UI
# ------------------------
st.header("데이터 및 시각화 설정")

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

# 날짜 설정 (기본 2020년 1월 1일)
default_end_date = datetime.today()
default_start_date = datetime(2020, 1, 1) # 기본 시작 날짜를 2020년 1월 1일로 설정
start_date = st.date_input("데이터 시작 날짜", default_start_date)
end_date = st.date_input("데이터 종료 날짜", default_end_date)

if start_date >= end_date:
    st.error("❌ 종료 날짜는 시작 날짜보다 미래여야 합니다.")
    st.stop()

# ------------------------
# ✨ Upbit API 함수 (캔들 데이터 로드)
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
            'to': (current_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
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
            
            current_date = temp_df['timestamp'].min().date() - timedelta(days=1)
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
            st.error(f"❌ Upbit API 응답 파싱 오류: {e}")
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
    
    st.success(f"✅ **{company_name}** 데이터 로드 완료! ({df_final.index.min().date()} ~ {df_final.index.max().date()})")
    return df_final

# ------------------------
# ✨ FRED 거시 경제 데이터 로드 함수
# ------------------------
@st.cache_data(ttl=3600)
def load_fred_macro_data(start_date, end_date):
    """
    FRED API에서 CPI, 미국 10년물 국채 금리 데이터를 가져옵니다.
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

    if econ_df.empty:
        st.warning("선택된 기간에 유효한 거시 경제 지표 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 거시 경제 지표 데이터 로드 완료! ({econ_df.index.min().date()} ~ {econ_df.index.max().date()})")
    return econ_df


# ------------------------
# ✨ 시각화 실행 버튼
# ------------------------
if st.button("🚀 데이터 로드 및 시각화 실행"):
    with st.spinner("데이터 로드 중..."):
        # 암호화폐 데이터 로드
        df_crypto = load_crypto_data(symbol, start_date, end_date)
        
        if df_crypto.empty:
            st.error("암호화폐 데이터 로드에 실패하여 시각화를 진행할 수 없습니다.")
            st.stop()

        # FRED 거시 경제 데이터 로드
        df_fred = load_fred_macro_data(start_date, end_date)

        if df_fred.empty:
            st.error("FRED 거시 경제 데이터를 로드할 수 없어 시각화를 진행할 수 없습니다. FRED API 키를 확인하거나 날짜 범위를 조정해 보세요.")
            st.stop()

    st.subheader("📊 암호화폐 가격과 미국 거시 경제 지표 비교")
    
    # --- 1. 암호화폐 가격 vs. 소비자물가지수 (CPI) ---
    st.markdown("#### 📈 암호화폐 가격 vs. 소비자물가지수 (CPI)")
    df_cpi_combined = pd.merge(df_crypto[['close']], df_fred[['CPI']], 
                               left_index=True, right_index=True, how='inner')
    df_cpi_combined = df_cpi_combined.dropna()

    if df_cpi_combined.empty:
        st.warning("선택된 기간에 암호화폐 가격과 CPI 데이터를 모두 포함하는 데이터가 충분하지 않습니다. 날짜 범위를 조정해 보세요.")
    else:
        st.success(f"✅ 암호화폐 가격-CPI 결합 데이터 로드 완료! ({df_cpi_combined.index.min().date()} ~ {df_cpi_combined.index.max().date()})")
        fig_cpi = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.1,
                                row_heights=[0.7, 0.3]) # 가격 차트를 더 크게

        fig_cpi.add_trace(go.Scatter(x=df_cpi_combined.index, y=df_cpi_combined['close'],
                                     mode='lines', name=f'{company_name} 가격', line=dict(color='blue')), row=1, col=1)
        fig_cpi.update_yaxes(title_text=f"{company_name} 가격", row=1, col=1)

        fig_cpi.add_trace(go.Scatter(x=df_cpi_combined.index, y=df_cpi_combined['CPI'],
                                     mode='lines', name='소비자물가지수 (CPI)', line=dict(color='orange')), row=2, col=1)
        fig_cpi.update_yaxes(title_text="CPI", row=2, col=1)

        fig_cpi.update_layout(height=600, title_text=f"{company_name} 가격과 CPI 비교",
                              xaxis_rangeslider_visible=False)
        fig_cpi.update_xaxes(showgrid=True, tickangle=45)
        st.plotly_chart(fig_cpi, use_container_width=True)
    
    st.markdown("---") # 구분선 추가

    # --- 2. 암호화폐 가격 vs. 미국 10년물 국채 금리 ---
    st.markdown("#### 📈 암호화폐 가격 vs. 미국 10년물 국채 금리")
    df_us10y_combined = pd.merge(df_crypto[['close']], df_fred[['US_10Y_Yield']], 
                                 left_index=True, right_index=True, how='inner')
    df_us10y_combined = df_us10y_combined.dropna()

    if df_us10y_combined.empty:
        st.warning("선택된 기간에 암호화폐 가격과 미국 10년물 국채 금리 데이터를 모두 포함하는 데이터가 충분하지 않습니다. 날짜 범위를 조정해 보세요.")
    else:
        st.success(f"✅ 암호화폐 가격-미국 10년물 국채 금리 결합 데이터 로드 완료! ({df_us10y_combined.index.min().date()} ~ {df_us10y_combined.index.max().date()})")
        fig_us10y = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.1,
                                  row_heights=[0.7, 0.3]) # 가격 차트를 더 크게

        fig_us10y.add_trace(go.Scatter(x=df_us10y_combined.index, y=df_us10y_combined['close'],
                                      mode='lines', name=f'{company_name} 가격', line=dict(color='blue')), row=1, col=1)
        fig_us10y.update_yaxes(title_text=f"{company_name} 가격", row=1, col=1)

        fig_us10y.add_trace(go.Scatter(x=df_us10y_combined.index, y=df_us10y_combined['US_10Y_Yield'],
                                      mode='lines', name='미국 10년물 국채 금리', line=dict(color='green')), row=2, col=1)
        fig_us10y.update_yaxes(title_text="미국 10년물 금리 (%)", row=2, col=1)

        fig_us10y.update_layout(height=600, title_text=f"{company_name} 가격과 미국 10년물 국채 금리 비교",
                                xaxis_rangeslider_visible=False)
        fig_us10y.update_xaxes(showgrid=True, tickangle=45)
        st.plotly_chart(fig_us10y, use_container_width=True)

    st.markdown("---")
    st.write("### 📝 참고 사항")
    st.write("""
    - **FRED API 키**: `.streamlit/secrets.toml` 파일에 `FRED_API_KEY = "YOUR_FRED_API_KEY"` 형식으로 FRED API 키를 설정해야 합니다.
    - **데이터 기간**: 이 앱은 기본적으로 2020년 1월 1일부터 현재까지의 데이터를 사용합니다. 원하는 기간으로 조정하여 데이터를 확인해 보세요.
    - **데이터 병합**: 암호화폐 데이터와 FRED 데이터는 `inner join` 방식으로 병합되므로, 두 데이터셋 모두에 존재하는 공통 날짜에 대해서만 시각화됩니다.
    """)
