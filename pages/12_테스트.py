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
st.set_page_config(page_title="암호화폐 vs. CPI 영향 분석", layout="wide")
st.title("📈 암호화폐 가격과 미국 CPI 발표 영향 분석")

st.markdown("""
Upbit API를 통해 암호화폐 가격 데이터를 가져오고, FRED API를 통해 미국 소비자물가지수(CPI) 데이터를 가져와
지난 7년간 CPI 발표 시점의 암호화폐 가격 움직임을 시각화하여 분석합니다.
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
    except ValueError as e: # JSONDecodeError 대신 ValueError로 변경 (requests.json()에서 발생할 수 있음)
        st.error(f"❌ Upbit API 응답 파싱 오류: {e}")
        st.stop()
        return {}
    except Exception as e: # 기타 예외 처리
        st.error(f"❌ Upbit API 목록 로드 중 알 수 없는 오류 발생: {e}")
        st.stop()
        return {}


crypto_list = get_upbit_markets()
company_names = list(crypto_list.keys())

# ------------------------
# ✨ 암호화폐 종목 선택 UI
# ------------------------
st.header("데이터 및 분석 설정")

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

# 날짜 설정 (기본 7년치 데이터)
default_end_date = datetime.today()
default_start_date = default_end_date - timedelta(days=365 * 7) # 기본 7년치 데이터
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
    # Upbit API는 한 번에 최대 200개 캔들만 제공하므로, 반복적으로 요청해야 합니다.
    # 요청 횟수를 제한하여 무한 루프 방지 및 API 호출 제한 준수
    max_requests = (end_date - start_date).days // 200 + 2 # 대략적인 필요한 요청 횟수 + 여유분
    
    st.info(f"� 업비트에서 **{symbol}** 데이터를 수집하고 있습니다...")
    progress_bar = st.progress(0)
    
    for i in range(max_requests):
        params = {
            'market': symbol,
            'to': (current_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
            'count': 200
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
            data = response.json()
            
            if not data: # 더 이상 데이터가 없으면 중단
                break
                
            temp_df = pd.DataFrame(data)
            temp_df['timestamp'] = pd.to_datetime(temp_df['candle_date_time_kst'])
            temp_df = temp_df.rename(columns={'opening_price': 'open', 'high_price': 'high', 'low_price': 'low', 'trade_price': 'close', 'candle_acc_trade_volume': 'volume'})
            df_list.append(temp_df)
            
            # 다음 요청을 위한 날짜 업데이트
            current_date = temp_df['timestamp'].min().date() - timedelta(days=1)
            
            # 진행률 업데이트
            progress_percentage = (end_date - current_date).days / (end_date - start_date).days
            progress_bar.progress(min(1.0, progress_percentage))
            
            time.sleep(0.15) # API 호출 제한을 준수하기 위한 지연
        
        except requests.exceptions.RequestException as e:
            st.error(f"Upbit API 요청 실패: {e}")
            progress_bar.empty()
            return pd.DataFrame()
        except ValueError as e: # JSONDecodeError 대신 ValueError로 변경 (requests.json()에서 발생할 수 있음)
            st.error(f"Upbit API 응답 파싱 오류: {e}")
            progress_bar.empty()
            return pd.DataFrame()
        except Exception as e: # 기타 예외 처리
            st.error(f"Upbit 데이터 로드 중 알 수 없는 오류 발생: {e}")
            progress_bar.empty()
            return pd.DataFrame()

    progress_bar.empty()

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
# ✨ FRED CPI 데이터 로드 함수
# ------------------------
@st.cache_data(ttl=3600)
def load_fred_cpi_data(start_date, end_date):
    """
    FRED API에서 소비자물가지수 (CPIAUCSL) 데이터를 가져옵니다.
    """
    if not fred: # FRED API 키가 없으면 함수 종료
        return pd.DataFrame()

    st.info("🔄 FRED CPI 데이터 수집 중...")

    # 소비자물가지수 (CPIAUCSL) - 월별
    try:
        cpi_series = fetch_fred_series_with_retry('CPIAUCSL', start_date, end_date)
        if cpi_series is not None and not cpi_series.empty:
            cpi_series = cpi_series.rename("CPI")
            st.success(f"✅ CPI 데이터 로드 완료! ({cpi_series.index.min().date()} ~ {cpi_series.index.max().date()})")
            return cpi_series
        else:
            st.warning("선택된 기간에 CPI 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ 소비자물가지수(CPI) 로드 중 오류 발생: {e}")
        return pd.DataFrame()


# ------------------------
# ✨ CPI 발표 시점 암호화폐 가격 영향 분석 함수
# ------------------------
def analyze_cpi_impact(df_crypto, cpi_series, window_days=7):
    """
    CPI 발표 날짜를 기준으로 암호화폐 가격의 변화를 분석합니다.
    """
    st.info(f"🔄 CPI 발표 시점 암호화폐 가격 변화 분석 중 (발표 후 {window_days}일 기준)...")
    
    analysis_results = []
    
    # CPI 시리즈의 인덱스가 CPI 발표 날짜로 간주합니다.
    # CPI는 월별 데이터이므로, 각 월의 첫 날짜가 데이터 포인트가 됩니다.
    for cpi_date in cpi_series.index:
        # CPI 발표일의 암호화폐 가격
        crypto_price_on_cpi_date = df_crypto['close'].asof(cpi_date)
        
        # CPI 발표일로부터 window_days 후의 암호화폐 가격
        future_date = cpi_date + timedelta(days=window_days)
        crypto_price_future = df_crypto['close'].asof(future_date)
        
        if pd.notna(crypto_price_on_cpi_date) and pd.notna(crypto_price_future):
            price_change_percent = ((crypto_price_future - crypto_price_on_cpi_date) / crypto_price_on_cpi_date) * 100
            
            analysis_results.append({
                'CPI 발표일': cpi_date.strftime('%Y-%m-%d'),
                'CPI 값': cpi_series.loc[cpi_date],
                f'{company_name} 가격 (발표일)': f"{crypto_price_on_cpi_date:,.2f}",
                f'{company_name} 가격 ({window_days}일 후)': f"{crypto_price_future:,.2f}",
                f'가격 변화율 ({window_days}일, %)': f"{price_change_percent:,.2f}%"
            })
            
    if not analysis_results:
        st.warning("⚠️ CPI 발표일과 암호화폐 가격 데이터가 겹치는 유효한 분석 기간이 부족합니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    df_analysis = pd.DataFrame(analysis_results)
    
    st.success("✅ CPI 발표 시점 암호화폐 가격 변화 분석 완료!")
    return df_analysis


# ------------------------
# ✨ 시각화 실행 버튼
# ------------------------
if st.button("🚀 데이터 로드 및 분석 실행"):
    with st.spinner("데이터 로드 중..."):
        # 암호화폐 데이터 로드
        df_crypto = load_crypto_data(symbol, start_date, end_date)
        
        if df_crypto.empty:
            st.error("암호화폐 데이터 로드에 실패하여 분석을 진행할 수 없습니다.")
            st.stop()

        # FRED CPI 데이터 로드
        cpi_series = load_fred_cpi_data(start_date, end_date)

        if cpi_series.empty:
            st.error("FRED CPI 데이터를 로드할 수 없어 분석을 진행할 수 없습니다. FRED API 키를 확인하거나 날짜 범위를 조정해 보세요.")
            st.stop()

    # CPI 영향 분석 실행
    df_cpi_impact = analyze_cpi_impact(df_crypto, cpi_series, window_days=7)

    if not df_cpi_impact.empty:
        st.subheader("📊 CPI 발표 시점 암호화폐 가격 변화 (표)")
        st.dataframe(df_cpi_impact)

        st.subheader(f"📈 {company_name} 가격 추이 및 CPI 발표 시점")
        fig_price_cpi = go.Figure()

        # 암호화폐 가격 라인
        fig_price_cpi.add_trace(go.Scatter(x=df_crypto.index, y=df_crypto['close'],
                                           mode='lines', name=f'{company_name} 가격', line=dict(color='blue')))
        
        # CPI 발표 시점 세로선 추가
        for cpi_date in cpi_series.index:
            if cpi_date in df_crypto.index: # 암호화폐 데이터에 해당 날짜가 있는 경우에만 표시
                fig_price_cpi.add_vline(x=cpi_date, line_width=1, line_dash="dot", line_color="red",
                                        annotation_text=f"CPI({cpi_date.strftime('%Y-%m')})",
                                        annotation_position="top right",
                                        annotation_font_size=10,
                                        annotation_font_color="red")

        fig_price_cpi.update_layout(height=600, title_text=f"{company_name} 가격 추이와 CPI 발표 시점",
                                    xaxis_rangeslider_visible=True) # 범위 슬라이더 추가
        fig_price_cpi.update_xaxes(showgrid=True, tickangle=45)
        st.plotly_chart(fig_price_cpi, use_container_width=True)

    st.markdown("---")
    st.write("### 📝 참고 사항")
    st.write("""
    - **FRED API 키**: `.streamlit/secrets.toml` 파일에 `FRED_API_KEY = "YOUR_FRED_API_KEY"` 형식으로 FRED API 키를 설정해야 합니다.
    - **데이터 기간**: 이 앱은 기본적으로 지난 7년간의 데이터를 사용합니다. 원하는 기간으로 조정하여 데이터를 확인해 보세요.
    - **CPI 발표일**: FRED에서 제공하는 월별 CPI 데이터의 인덱스 날짜를 CPI 발표일로 간주하고 분석을 수행합니다. 실제 발표일과는 약간의 차이가 있을 수 있습니다.
    - **가격 변화율**: CPI 발표일의 종가와 발표일로부터 7일 후의 종가를 기준으로 계산됩니다. 주말이나 공휴일로 인해 7일 후의 정확한 데이터가 없는 경우, 가장 가까운 유효한 날짜의 데이터가 사용됩니다 (`asof` 메서드).
    """)
