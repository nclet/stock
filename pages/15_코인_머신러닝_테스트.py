import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm # 한글 폰트 관련 모듈 제거 (요청에 따라)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from json.decoder import JSONDecodeError
import time
from fredapi import Fred # FRED API를 위한 라이브러리
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import urllib.error # HTTPError를 위해 임포트
import plotly.graph_objects as go
from plotly.subplots import make_subplots # make_subplots 임포트 추가

# --- 페이지 설정 ---
st.set_page_config(page_title="암호화폐 예측 및 지표 분석", layout="wide")
st.title("📈 암호화폐 LSTM 예측 및 다양한 지표 분석")

st.markdown("""
Upbit API를 통해 암호화폐 가격 데이터를 가져와 LSTM 딥러닝 모델로 미래 가격을 예측하고,
다양한 기술적 지표, 온체인 데이터(설명), 거시 경제 지표를 함께 시각화하여 분석합니다.
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
    # 이 부분이 이전 오류의 원인이었습니다. 정확히 수정되었습니다.
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
st.header("데이터 및 모델 설정")

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

# 날짜 설정 (최소 1년치 데이터 권장)
default_end_date = datetime.today()
default_start_date = default_end_date - timedelta(days=365 * 3) # 기본 3년치 데이터
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
# ✨ LSTM 모델 관련 설정 및 함수
# ------------------------
st.subheader("LSTM 모델 파라미터")
look_back = st.slider("과거 데이터 사용 기간 (look_back)", 10, 60, 30)
epochs = st.slider("학습 에포크 (epochs)", 10, 100, 50)
batch_size = st.slider("배치 크기 (batch_size)", 16, 128, 32)
train_test_split_ratio = st.slider("학습/테스트 데이터 분할 비율 (%)", 70, 95, 80) / 100.0

def create_sequences(data, look_back):
    """LSTM 모델을 위한 시퀀스 데이터셋을 생성합니다."""
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# ------------------------
# ✨ 기술적 지표 계산 함수
# ------------------------
def calculate_technical_indicators(df):
    """
    DataFrame에 모멘텀, RSI, MACD, OBV를 추가합니다.
    """
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()

    # 모멘텀 (14일)
    df['Momentum'] = df['close'].pct_change(14) * 100

    # RSI (14일)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = np.where(loss == 0, np.inf, gain / loss)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # OBV (On-Balance Volume)
    obv_values = np.zeros(len(df))
    if len(df) > 0:
        obv_values[0] = df['volume'].iloc[0]
    for k in range(1, len(df)):
        if df['close'].iloc[k] > df['close'].iloc[k-1]:
            obv_values[k] = obv_values[k-1] + df['volume'].iloc[k]
        elif df['close'].iloc[k] < df['close'].iloc[k-1]:
            obv_values[k] = obv_values[k-1] - df['volume'].iloc[k]
        else:
            obv_values[k] = obv_values[k-1]
    df['OBV'] = obv_values
    
    return df

# ------------------------
# ✨ FRED 데이터 로드 함수
# ------------------------
@st.cache_data(ttl=3600)
def load_fred_indicators(start_date, end_date):
    """
    FRED API에서 CPI와 미국 10년물 국채 금리 데이터를 가져옵니다.
    """
    econ_data = {}
    econ_errors = []

    if not fred: # FRED API 키가 없으면 함수 종료
        return pd.DataFrame()

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
# ✨ 예측 및 시각화 실행 버튼
# ------------------------
if st.button("🚀 LSTM 모델 학습 및 지표 시각화 실행"):
    with st.spinner("데이터 로드 및 전처리 중..."):
        df = load_crypto_data(symbol, start_date, end_date)
        
        if df.empty:
            st.error("데이터 로드에 실패하여 예측을 진행할 수 없습니다.")
            st.stop()

        # 기술적 지표 계산
        df_with_indicators = calculate_technical_indicators(df.copy())
        
        # 'close' 가격만 사용 (LSTM 예측용)
        data = df['close'].values.reshape(-1, 1)

        # 데이터 정규화
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # 학습/테스트 데이터 분할
        train_size = int(len(scaled_data) * train_test_split_ratio)
        train_data = scaled_data[0:train_size, :]
        test_data = scaled_data[train_size:len(scaled_data), :]

        # 시퀀스 생성
        X_train, y_train = create_sequences(train_data, look_back)
        X_test, y_test = create_sequences(test_data, look_back)

        # LSTM 입력 형태에 맞게 reshape (samples, time_steps, features)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    with st.spinner("LSTM 모델 학습 중..."):
        # LSTM 모델 구축
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1)) # 출력 레이어는 1 (예측 가격)

        model.compile(optimizer='adam', loss='mean_squared_error')

        # 조기 종료 (Early Stopping) 콜백 설정
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # 모델 학습
        history = model.fit(X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_split=0.1, 
                            callbacks=[early_stopping],
                            verbose=0)
        
        st.success("✅ LSTM 모델 학습 완료!")

    with st.spinner("가격 예측 중..."):
        # 예측 수행
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # 예측 값 역정규화
        train_predict = scaler.inverse_transform(train_predict)
        y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
        
        test_predict = scaler.inverse_transform(test_predict)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # 예측 결과를 위한 데이터프레임 생성
        train_predict_plot = np.empty_like(data)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

        test_predict_plot = np.empty_like(data)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(train_predict) + (look_back * 2):len(data), :] = test_predict

        # 날짜 인덱스 매핑
        dates = df.index
        df_results = pd.DataFrame(index=dates)
        df_results['실제 가격'] = data
        df_results['학습 예측'] = train_predict_plot
        df_results['테스트 예측'] = test_predict_plot
        
        st.success("✅ 가격 예측 완료!")

    # ------------------------
    # ✨ 1. LSTM 예측 및 기술적 지표 시각화
    # ------------------------
    st.subheader("📊 LSTM 예측 및 기술적 지표 시각화")
    # 5개의 서브플롯: 가격(LSTM), 모멘텀, RSI, MACD, OBV
    fig_tech = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                             vertical_spacing=0.05,
                             row_width=[0.4, 0.15, 0.15, 0.15, 0.15]) # 비율 조정

    # 1행: 실제 가격 및 LSTM 예측
    fig_tech.add_trace(go.Scatter(x=df_results.index, y=df_results['실제 가격'], 
                                  mode='lines', name='실제 가격', line=dict(color='blue')), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_results.index, y=df_results['학습 예측'], 
                                  mode='lines', name='학습 예측', line=dict(color='green', dash='dot')), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_results.index, y=df_results['테스트 예측'], 
                                  mode='lines', name='테스트 예측', line=dict(color='red', dash='dot')), row=1, col=1)
    fig_tech.update_yaxes(title_text="가격", row=1, col=1)


    # 2행: 모멘텀
    fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Momentum'], 
                                  mode='lines', name='모멘텀', line=dict(color='purple')), row=2, col=1)
    fig_tech.update_yaxes(title_text="모멘텀", row=2, col=1)
    fig_tech.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1) # 0선 추가

    # 3행: RSI
    fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['RSI'], 
                                  mode='lines', name='RSI', line=dict(color='orange')), row=3, col=1)
    fig_tech.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig_tech.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig_tech.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    # 4행: MACD
    fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD'], 
                                  mode='lines', name='MACD', line=dict(color='blue')), row=4, col=1)
    fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD_Signal'], 
                                  mode='lines', name='Signal', line=dict(color='red', dash='dot')), row=4, col=1)
    # MACD 히스토그램 (바 차트)
    colors_macd_hist = ['rgba(0,255,0,0.5)' if val >= 0 else 'rgba(255,0,0,0.5)' for val in df_with_indicators['MACD_Hist']]
    fig_tech.add_trace(go.Bar(x=df_with_indicators.index, y=df_with_indicators['MACD_Hist'], 
                               name='MACD Hist', marker_color=colors_macd_hist), row=4, col=1)
    fig_tech.update_yaxes(title_text="MACD", row=4, col=1)
    fig_tech.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)

    # 5행: OBV
    fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['OBV'], 
                                  mode='lines', name='OBV', line=dict(color='darkgreen')), row=5, col=1)
    fig_tech.update_yaxes(title_text="OBV", row=5, col=1)

    fig_tech.update_layout(height=1000, title_text=f"{company_name} 가격 예측 및 기술적 지표",
                            xaxis_rangeslider_visible=False)
    fig_tech.update_xaxes(showgrid=True, tickangle=45)
    st.plotly_chart(fig_tech, use_container_width=True)


    # ------------------------
    # ✨ 2. 온체인 데이터 시각화 (설명)
    # ------------------------
    st.subheader("🔗 온체인 데이터 분석 (개념 설명)")
    st.warning("⚠️ **Upbit API는 온체인 데이터를 직접 제공하지 않습니다.**")
    st.info("""
    온체인 데이터는 블록체인 네트워크 상의 실제 활동(예: 거래량, 활성 주소, 고래 움직임)을 보여주므로, 시장 심리와 추세를 파악하는 데 매우 중요합니다.
    하지만 Upbit과 같은 중앙화된 거래소의 API는 주로 거래소 내부의 가격 및 주문 정보만을 제공합니다.
    
    온체인 데이터를 활용하려면 **Glassnode, CryptoQuant, CoinMetrics**와 같은 전문 온체인 데이터 분석 플랫폼의 API를 사용해야 합니다.
    이러한 서비스는 대부분 유료 구독 모델을 제공합니다.
    
    **주요 온체인 지표 (예시):**
    - **고래 움직임 (Whale Movements):** 대규모 자금의 이동은 시장에 큰 영향을 미칠 수 있습니다.
    - **거래소 입출금량 (Exchange Inflows/Outflows):** 거래소로 코인이 유입되면 매도 압력, 유출되면 매수 압력으로 해석될 수 있습니다.
    - **해시레이트 (Hash Rate):** 비트코인 등 PoW 코인의 채굴 난이도 및 네트워크 보안성을 나타냅니다.
    - **미실현 손익 (Unrealized Profit/Loss):** 현재 가격 기준으로 코인 보유자들이 얼마나 이익 또는 손실을 보고 있는지 추정합니다.
    
    이러한 데이터를 가져올 수 있다면, 가격 데이터와 결합하여 모델의 예측 정확도를 더욱 높일 수 있습니다.
    """)

    # ------------------------
    # ✨ 3. 거시 경제 지표 시각화 (FRED 데이터) - 분리된 차트
    # ------------------------
    st.subheader("🌍 거시 경제 지표와 암호화폐 가격 비교")
    
    # FRED 데이터 로드 (기본 시작 날짜를 10년 전으로 설정)
    default_fred_start_date = datetime.today() - timedelta(days=365 * 10) 
    df_fred = load_fred_indicators(default_fred_start_date, end_date)

    if not df_fred.empty:
        # --- 3-1. 암호화폐 가격 vs. 소비자물가지수 (CPI) ---
        st.markdown("#### 📈 암호화폐 가격 vs. 소비자물가지수 (CPI)")
        df_cpi_combined = pd.merge(df_results[['실제 가격']], df_fred[['CPI']], 
                                   left_index=True, right_index=True, how='inner')
        df_cpi_combined = df_cpi_combined.dropna()

        if df_cpi_combined.empty:
            st.warning("선택된 기간에 암호화폐 가격과 CPI 데이터를 모두 포함하는 데이터가 충분하지 않습니다. 날짜 범위를 조정해 보세요.")
        else:
            st.success(f"✅ 암호화폐 가격-CPI 결합 데이터 로드 완료! ({df_cpi_combined.index.min().date()} ~ {df_cpi_combined.index.max().date()})")
            fig_cpi = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.1,
                                    row_width=[0.5, 0.5])

            fig_cpi.add_trace(go.Scatter(x=df_cpi_combined.index, y=df_cpi_combined['실제 가격'],
                                         mode='lines', name=f'{company_name} 실제 가격', line=dict(color='blue')), row=1, col=1)
            fig_cpi.update_yaxes(title_text=f"{company_name} 가격", row=1, col=1)

            fig_cpi.add_trace(go.Scatter(x=df_cpi_combined.index, y=df_cpi_combined['CPI'],
                                         mode='lines', name='소비자물가지수 (CPI)', line=dict(color='orange')), row=2, col=1)
            fig_cpi.update_yaxes(title_text="CPI", row=2, col=1)

            fig_cpi.update_layout(height=600, title_text=f"{company_name} 가격과 CPI 비교",
                                  xaxis_rangeslider_visible=False)
            fig_cpi.update_xaxes(showgrid=True, tickangle=45)
            st.plotly_chart(fig_cpi, use_container_width=True)
        
        st.markdown("---") # 구분선 추가

        # --- 3-2. 암호화폐 가격 vs. 미국 10년물 국채 금리 ---
        st.markdown("#### 📈 암호화폐 가격 vs. 미국 10년물 국채 금리")
        df_us10y_combined = pd.merge(df_results[['실제 가격']], df_fred[['US_10Y_Yield']], 
                                     left_index=True, right_index=True, how='inner')
        df_us10y_combined = df_us10y_combined.dropna()

        if df_us10y_combined.empty:
            st.warning("선택된 기간에 암호화폐 가격과 미국 10년물 국채 금리 데이터를 모두 포함하는 데이터가 충분하지 않습니다. 날짜 범위를 조정해 보세요.")
        else:
            st.success(f"✅ 암호화폐 가격-미국 10년물 국채 금리 결합 데이터 로드 완료! ({df_us10y_combined.index.min().date()} ~ {df_us10y_combined.index.max().date()})")
            fig_us10y = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      vertical_spacing=0.1,
                                      row_width=[0.5, 0.5])

            fig_us10y.add_trace(go.Scatter(x=df_us10y_combined.index, y=df_us10y_combined['실제 가격'],
                                          mode='lines', name=f'{company_name} 실제 가격', line=dict(color='blue')), row=1, col=1)
            fig_us10y.update_yaxes(title_text=f"{company_name} 가격", row=1, col=1)

            fig_us10y.add_trace(go.Scatter(x=df_us10y_combined.index, y=df_us10y_combined['US_10Y_Yield'],
                                          mode='lines', name='미국 10년물 국채 금리', line=dict(color='green')), row=2, col=1)
            fig_us10y.update_yaxes(title_text="미국 10년물 금리 (%)", row=2, col=1)

            fig_us10y.update_layout(height=600, title_text=f"{company_name} 가격과 미국 10년물 국채 금리 비교",
                                    xaxis_rangeslider_visible=False)
            fig_us10y.update_xaxes(showgrid=True, tickangle=45)
            st.plotly_chart(fig_us10y, use_container_width=True)

    else:
        st.warning("거시 경제 지표를 로드할 수 없어 시각화를 건너킵니다. FRED API 키를 확인하거나 날짜 범위를 조정해 보세요.")

    st.markdown("---")
    st.write("### 📝 추가 참고 사항")
    st.write("""
    - **데이터 통합**: 다양한 유형의 데이터를 모델에 통합할 때는 각 데이터의 빈도(일별, 월별 등)를 맞추는 것이 중요합니다. (예: 월별 데이터를 일별로 `ffill` 하는 방식)
    - **피처 엔지니어링**: 단순히 원시 데이터를 사용하는 것을 넘어, 각 지표의 변화율, 이동평균선과의 관계, 특정 임계값 돌파 여부 등 새로운 피처를 생성하여 모델의 학습 능력을 향상시킬 수 있습니다.
    - **모델 복잡도**: 더 많은 피처를 사용할수록 모델의 복잡도가 증가하며, 과적합(Overfitting) 위험이 커질 수 있습니다. 적절한 정규화(Regularization) 기법(예: Dropout)과 검증을 통해 이를 관리해야 합니다.
    - **해석의 어려움**: 다양한 팩터를 포함할수록 모델의 '블랙박스' 특성이 강해져 예측 결과의 원인을 해석하기 어려워질 수 있습니다.
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
