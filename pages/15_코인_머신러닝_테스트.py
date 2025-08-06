import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm # 한글 폰트 관련 모듈 제거 (요청에 따라)
from fredapi import Fred # FRED API를 위한 라이브러리
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import urllib.error # HTTPError를 위해 임포트
import plotly.graph_objects as go
from plotly.subplots import make_subplots # make_subplots 임포트 추가
import time # 시간 지연을 위해 추가

# --- 페이지 설정 ---
st.set_page_config(page_title="암호화폐 vs. 미국 국채 스프레드", layout="wide")
st.title("📈 암호화폐 가격 vs. 미국 국채 장단기 금리 스프레드")

st.markdown("""
Upbit API를 통해 암호화폐 가격 데이터를 가져오고, FRED API를 통해 미국 국채 장단기 금리 데이터를 가져와
지난 5년간의 추이를 함께 시각화하여 비교합니다.
""")

# ------------------------
# ✨ 한글 폰트 설정 (제거됨)
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

# 날짜 설정 (기본 5년치 데이터)
default_end_date = datetime.today()
default_start_date = default_end_date - timedelta(days=365 * 5) # 기본 5년치 데이터
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
            st.error(f"Upbit API 응답 파싱 오류: {e}")
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
# ✨ FRED 데이터 로드 함수 (미국 국채 장단기 금리 스프레드)
# ------------------------
@st.cache_data(ttl=3600)
def load_us_treasury_spread(start_date, end_date):
    """
    FRED API에서 미국 10년물 국채 금리 (GS10)와 2년물 국채 금리 (GS2)를 가져와
    장단기 금리 스프레드를 계산합니다.
    """
    if not fred: # FRED API 키가 없으면 함수 종료
        return pd.DataFrame()

    st.info("🔄 FRED에서 미국 국채 금리 데이터 수집 중...")
    
    # 미국 10년물 국채 금리 (일별)
    try:
        gs10 = fetch_fred_series_with_retry('GS10', start_date, end_date)
        gs10 = gs10.rename("US_10Y_Yield")
        st.info(f"✅ 미국 10년물 국채 금리 로드: {gs10.index.min().date()} ~ {gs10.index.max().date()}")
    except Exception as e:
        st.error(f"❌ 미국 10년물 국채 금리 로드 중 오류 발생: {e}")
        return pd.DataFrame()

    # 미국 2년물 국채 금리 (일별)
    try:
        gs2 = fetch_fred_series_with_retry('GS2', start_date, end_date)
        gs2 = gs2.rename("US_2Y_Yield")
        st.info(f"✅ 미국 2년물 국채 금리 로드: {gs2.index.min().date()} ~ {gs2.index.max().date()}")
    except Exception as e:
        st.error(f"❌ 미국 2년물 국채 금리 로드 중 오류 발생: {e}")
        return pd.DataFrame()

    # 두 시리즈를 병합하고 스프레드 계산
    df_treasury = pd.merge(gs10, gs2, left_index=True, right_index=True, how='inner')
    df_treasury['US_Yield_Spread'] = df_treasury['US_10Y_Yield'] - df_treasury['US_2Y_Yield']
    
    # 결측치 제거 (FRED 데이터는 기본적으로 결측치가 적지만, 혹시 모를 경우 대비)
    df_treasury = df_treasury.dropna()

    if df_treasury.empty:
        st.warning("선택된 기간에 유효한 미국 국채 금리 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    st.success(f"✅ 미국 국채 장단기 금리 스프레드 데이터 로드 완료! ({df_treasury.index.min().date()} ~ {df_treasury.index.max().date()})")
    return df_treasury


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

        # 미국 국채 스프레드 데이터 로드
        df_treasury_spread = load_us_treasury_spread(start_date, end_date)

        if df_treasury_spread.empty:
            st.error("미국 국채 장단기 금리 스프레드 데이터 로드에 실패하여 시각화를 진행할 수 없습니다.")
            st.stop()

    st.subheader("📊 암호화폐 가격과 미국 국채 장단기 금리 스프레드 비교")
    
    # 두 데이터셋의 공통 날짜 범위 찾기
    common_dates_index = df_crypto.index.intersection(df_treasury_spread.index)
    
    if common_dates_index.empty:
        st.warning("선택된 기간에 암호화폐 가격과 미국 국채 장단기 금리 스프레드를 모두 포함하는 공통 데이터가 충분하지 않습니다. 날짜 범위를 조정해 보세요.")
    else:
        # 공통 날짜 범위로 데이터 필터링
        df_crypto_filtered = df_crypto.loc[common_dates_index]
        df_treasury_spread_filtered = df_treasury_spread.loc[common_dates_index]

        st.success(f"✅ 최종 시각화 데이터 기간: {common_dates_index.min().date()} ~ {common_dates_index.max().date()}")

        # 2개의 서브플롯: 암호화폐 가격, 미국 국채 장단기 금리 스프레드
        fig_comparison = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.1,
                                       row_heights=[0.7, 0.3]) # 가격 차트를 더 크게

        # 1행: 암호화폐 가격
        fig_comparison.add_trace(go.Scatter(x=df_crypto_filtered.index, y=df_crypto_filtered['close'],
                                            mode='lines', name=f'{company_name} 가격', line=dict(color='blue')), row=1, col=1)
        fig_comparison.update_yaxes(title_text=f"{company_name} 가격", row=1, col=1)

        # 2행: 미국 국채 장단기 금리 스프레드
        fig_comparison.add_trace(go.Scatter(x=df_treasury_spread_filtered.index, y=df_treasury_spread_filtered['US_Yield_Spread'],
                                            mode='lines', name='미국 국채 장단기 금리 스프레드', line=dict(color='green')), row=2, col=1)
        fig_comparison.update_yaxes(title_text="금리 스프레드 (%)", row=2, col=1)
        fig_comparison.add_hline(y=0, line_dash="dot", line_color="red", row=2, col=1) # 0선 추가 (장단기 금리 역전 표시)

        fig_comparison.update_layout(height=700, title_text=f"{company_name} 가격과 미국 국채 장단기 금리 스프레드 추이",
                                     xaxis_rangeslider_visible=False)
        fig_comparison.update_xaxes(showgrid=True, tickangle=45)
        st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("---")
    st.write("### 📝 참고 사항")
    st.write("""
    - **장단기 금리 스프레드**: 10년물 국채 금리에서 2년물 국채 금리를 뺀 값입니다. 이 값이 0보다 작아지면 (음수가 되면) '장단기 금리 역전'이라고 하며, 이는 종종 경기 침체의 전조로 해석되기도 합니다.
    - **데이터 기간**: 암호화폐 데이터와 FRED 데이터의 실제 시작일이 다를 수 있습니다. 시각화는 두 데이터 모두 존재하는 가장 긴 공통 기간에 대해서만 이루어집니다.
    - **FRED API 키**: `.streamlit/secrets.toml` 파일에 `FRED_API_KEY = "YOUR_FRED_API_KEY"` 형식으로 FRED API 키를 설정해야 합니다.
    """)



# import streamlit as st
# import pandas as pd
# import requests
# from datetime import datetime, timedelta
# import numpy as np
# import matplotlib.pyplot as plt
# # import matplotlib.font_manager as fm # 한글 폰트 관련 모듈 제거
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from json.decoder import JSONDecodeError
# import time

# # --- 페이지 설정 ---
# st.set_page_config(page_title="암호화폐 LSTM 가격 예측", layout="wide")
# st.title("📈 암호화폐 LSTM 가격 예측 및 시각화")

# st.markdown("""
# Upbit API를 통해 암호화폐 가격 데이터를 가져와 LSTM 딥러닝 모델로
# 미래 가격을 예측하고 시각화합니다.
# """)

# # ------------------------
# # ✨ 한글 폰트 설정 (제거됨)
# # ------------------------
# # 기존 한글 폰트 설정 함수 및 호출 코드 제거됨.
# # Streamlit Cloud 환경에서 기본 폰트가 한글을 지원하지 않을 경우,
# # 차트의 한글 텍스트가 깨져 보일 수 있습니다.
# # 이 경우, Streamlit 앱 배포 환경에 한글 폰트를 설치하거나
# # Plotly 등 다른 시각화 라이브러리를 고려할 수 있습니다.
# plt.rc('axes', unicode_minus=False) # 마이너스 폰트 깨짐 방지 (일반적인 설정이므로 유지)


# # ------------------------
# # ✨ 암호화폐 종목 목록 로드 (Upbit API)
# # ------------------------
# @st.cache_data
# def get_upbit_markets():
#     """
#     Upbit API에서 원화(KRW) 마켓에 있는 모든 암호화폐 목록을 가져옵니다.
#     """
#     url = "https://api.upbit.com/v1/market/all"
#     try:
#         response = requests.get(url, params={'isDetails': 'false'})
#         response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
#         markets = response.json()
        
#         # KRW 마켓만 필터링하고 코인 이름으로 매핑
#         krw_markets = {market['korean_name']: market['market'] for market in markets if market['market'].startswith('KRW-')}
        
#         if not krw_markets:
#             st.error("❌ Upbit API에서 원화 마켓 목록을 가져오지 못했습니다.")
#             st.info("Upbit API 서버 상태를 확인하거나 잠시 후 다시 시도해주세요.")
#             st.stop()
        
#         return krw_markets
    
#     except requests.exceptions.RequestException as e:
#         st.error(f"❌ Upbit API 연결 오류: {e}")
#         st.info("인터넷 연결 상태를 확인하거나 Upbit 서버에 문제가 있을 수 있습니다.")
#         st.stop()
#         return {}
#     except JSONDecodeError as e:
#         st.error(f"❌ Upbit API 응답 파싱 오류: {e}")
#         st.stop()
#         return {}

# crypto_list = get_upbit_markets()
# company_names = list(crypto_list.keys())

# # ------------------------
# # ✨ 암호화폐 종목 선택 UI
# # ------------------------
# st.header("데이터 및 모델 설정")

# default_crypto = "비트코인"
# if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
#     st.session_state.selected_company = default_crypto if default_crypto in company_names else company_names[0]

# company_name = st.selectbox(
#     "✅ 분석할 암호화폐 선택",
#     company_names,
#     index=company_names.index(st.session_state.selected_company),
#     key="selected_company"
# )
# symbol = crypto_list.get(st.session_state.selected_company)

# # 날짜 설정 (최소 1년치 데이터 권장)
# default_end_date = datetime.today()
# default_start_date = default_end_date - timedelta(days=365 * 3) # 기본 3년치 데이터
# start_date = st.date_input("데이터 시작 날짜", default_start_date)
# end_date = st.date_input("데이터 종료 날짜", default_end_date)

# if start_date >= end_date:
#     st.error("❌ 종료 날짜는 시작 날짜보다 미래여야 합니다.")
#     st.stop()

# # ------------------------
# # ✨ Upbit API 함수 (캔들 데이터 로드)
# # ------------------------
# @st.cache_data(ttl=3600)
# def load_crypto_data(symbol, start_date, end_date):
#     """
#     Upbit API를 통해 일별 캔들 데이터를 가져와 DataFrame으로 반환합니다.
#     """
#     base_url = "https://api.upbit.com/v1/candles/days"
#     df_list = []
#     current_date = end_date
#     max_requests = 20 # 200일씩 20번 요청 (총 4000일, 약 10년치)
#     requests_count = 0
    
#     st.info(f"🔄 업비트에서 **{symbol}** 데이터를 수집하고 있습니다...")
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     while current_date >= start_date and requests_count < max_requests:
#         params = {
#             'market': symbol,
#             'to': (current_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
#             'count': 200
#         }
        
#         try:
#             response = requests.get(base_url, params=params)
#             response.raise_for_status()
#             data = response.json()
            
#             if not data:
#                 break
                
#             temp_df = pd.DataFrame(data)
#             temp_df['timestamp'] = pd.to_datetime(temp_df['candle_date_time_kst'])
#             temp_df = temp_df.rename(columns={'opening_price': 'open', 'high_price': 'high', 'low_price': 'low', 'trade_price': 'close', 'candle_acc_trade_volume': 'volume'})
#             df_list.append(temp_df)
            
#             current_date = temp_df['timestamp'].min().date() - timedelta(days=1)
#             requests_count += 1
            
#             progress_percentage = (end_date - current_date).days / (end_date - start_date).days
#             progress_bar.progress(min(1.0, progress_percentage))
#             status_text.text(f"데이터 수집 중: {current_date} 부터...")
#             time.sleep(0.15)
        
#         except requests.exceptions.RequestException as e:
#             st.error(f"Upbit API 요청 실패: {e}")
#             progress_bar.empty()
#             status_text.empty()
#             return pd.DataFrame()
#         except JSONDecodeError as e:
#             st.error(f"Upbit API 응답 파싱 오류: {e}")
#             progress_bar.empty()
#             status_text.empty()
#             return pd.DataFrame()

#     progress_bar.empty()
#     status_text.empty()

#     if not df_list:
#         st.warning("⚠️ 지정된 기간 동안 데이터를 가져오지 못했습니다. 날짜 범위를 확인하세요.")
#         return pd.DataFrame()

#     df_final = pd.concat(df_list, ignore_index=True)
#     df_final = df_final.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)
#     df_final = df_final[(df_final['timestamp'].dt.date >= start_date) & (df_final['timestamp'].dt.date <= end_date)].reset_index(drop=True)
#     df_final.set_index('timestamp', inplace=True)
    
#     st.success(f"✅ **{company_name}** 데이터 로드 완료! ({df_final.index.min().date()} ~ {df_final.index.max().date()})")
#     return df_final

# # ------------------------
# # ✨ LSTM 모델 관련 설정 및 함수
# # ------------------------
# st.subheader("LSTM 모델 파라미터")
# look_back = st.slider("과거 데이터 사용 기간 (look_back)", 10, 60, 30)
# epochs = st.slider("학습 에포크 (epochs)", 10, 100, 50)
# batch_size = st.slider("배치 크기 (batch_size)", 16, 128, 32)
# train_test_split_ratio = st.slider("학습/테스트 데이터 분할 비율 (%)", 70, 95, 80) / 100.0

# def create_sequences(data, look_back):
#     """LSTM 모델을 위한 시퀀스 데이터셋을 생성합니다."""
#     X, Y = [], []
#     for i in range(len(data) - look_back):
#         X.append(data[i:(i + look_back), 0])
#         Y.append(data[i + look_back, 0])
#     return np.array(X), np.array(Y)

# # ------------------------
# # ✨ 예측 실행 버튼
# # ------------------------
# if st.button("🚀 LSTM 모델 학습 및 예측 실행"):
#     with st.spinner("데이터 로드 및 전처리 중..."):
#         df = load_crypto_data(symbol, start_date, end_date)
        
#         if df.empty:
#             st.error("데이터 로드에 실패하여 예측을 진행할 수 없습니다.")
#             st.stop()

#         # 'close' 가격만 사용
#         data = df['close'].values.reshape(-1, 1)

#         # 데이터 정규화
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         scaled_data = scaler.fit_transform(data)

#         # 학습/테스트 데이터 분할
#         train_size = int(len(scaled_data) * train_test_split_ratio)
#         train_data = scaled_data[0:train_size, :]
#         test_data = scaled_data[train_size:len(scaled_data), :]

#         # 시퀀스 생성
#         X_train, y_train = create_sequences(train_data, look_back)
#         X_test, y_test = create_sequences(test_data, look_back)

#         # LSTM 입력 형태에 맞게 reshape (samples, time_steps, features)
#         X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     with st.spinner("LSTM 모델 학습 중..."):
#         # LSTM 모델 구축
#         model = Sequential()
#         model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
#         model.add(Dropout(0.2))
#         model.add(LSTM(units=50, return_sequences=False))
#         model.add(Dropout(0.2))
#         model.add(Dense(units=1)) # 출력 레이어는 1 (예측 가격)

#         model.compile(optimizer='adam', loss='mean_squared_error')

#         # 조기 종료 (Early Stopping) 콜백 설정
#         early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#         # 모델 학습
#         history = model.fit(X_train, y_train, 
#                             epochs=epochs, 
#                             batch_size=batch_size, 
#                             validation_split=0.1, # 학습 데이터의 10%를 검증에 사용
#                             callbacks=[early_stopping],
#                             verbose=0) # Streamlit에서는 verbose를 0으로 설정하여 출력 줄임
        
#         st.success("✅ LSTM 모델 학습 완료!")

#     with st.spinner("가격 예측 중..."):
#         # 예측 수행
#         train_predict = model.predict(X_train)
#         test_predict = model.predict(X_test)

#         # 예측 값 역정규화
#         train_predict = scaler.inverse_transform(train_predict)
#         y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
        
#         test_predict = scaler.inverse_transform(test_predict)
#         y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

#         # 예측 결과를 위한 데이터프레임 생성
#         train_predict_plot = np.empty_like(data)
#         train_predict_plot[:, :] = np.nan
#         train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

#         test_predict_plot = np.empty_like(data)
#         test_predict_plot[:, :] = np.nan
#         test_predict_plot[len(train_predict) + (look_back * 2):len(data), :] = test_predict

#         # 날짜 인덱스 매핑
#         dates = df.index
#         df_results = pd.DataFrame(index=dates)
#         df_results['실제 가격'] = data
#         df_results['학습 예측'] = train_predict_plot
#         df_results['테스트 예측'] = test_predict_plot
        
#         st.success("✅ 가격 예측 완료!")

#     # ------------------------
#     # ✨ 시각화
#     # ------------------------
#     st.subheader("📊 실제 가격 vs. LSTM 예측 가격")
#     fig, ax = plt.subplots(figsize=(14, 7))
#     ax.plot(df_results.index, df_results['실제 가격'], label='실제 가격', color='blue')
#     ax.plot(df_results.index, df_results['학습 예측'], label='학습 예측', color='green', linestyle='--')
#     ax.plot(df_results.index, df_results['테스트 예측'], label='테스트 예측', color='red', linestyle='--')
    
#     ax.set_title(f"{company_name} 가격 예측 (LSTM)")
#     ax.set_xlabel("날짜")
#     ax.set_ylabel("가격")
#     ax.legend()
#     ax.grid(True, linestyle='--', alpha=0.7)
#     plt.xticks(rotation=45)
#     st.pyplot(fig)

#     st.markdown("---")
#     st.write("### 📝 참고")
#     st.write("""
#     - **LSTM 모델**: 과거 데이터를 기반으로 미래 값을 예측하는 딥러닝 모델입니다. `look_back` 기간 동안의 데이터를 사용하여 다음 날의 가격을 예측합니다.
#     - **데이터 정규화**: 모델 학습 효율을 높이기 위해 데이터를 0과 1 사이로 스케일링합니다. 예측 후 다시 원래 스케일로 역변환합니다.
#     - **학습/테스트 분할**: 전체 데이터 중 일부를 학습에 사용하고, 나머지는 모델이 얼마나 잘 예측하는지 평가하는 데 사용합니다.
#     - **예측의 한계**: 암호화폐 시장은 변동성이 매우 크고 다양한 외부 요인에 의해 영향을 받으므로, 딥러닝 모델도 완벽하게 예측하기는 어렵습니다. 이 앱은 예측 모델의 개념을 보여주는 예시이며, 실제 투자에 활용하기에는 추가적인 연구와 검증이 필요합니다.
#     """)
