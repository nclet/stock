import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
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
st.title("📈 암호화폐 LSTM 예측 및 경제 데이터 분석")

st.markdown("""
코인을 LSTM 딥러닝 모델로 미래 가격을 예측하고,
미국 소비자물가지수(CPI) 및 미국 국채 장단기 금리 스프레드 발표/변화 시점의 암호화폐 가격 움직임을 분석하고 시각화합니다.
""")
st.write("에포크와 데이터 기간이 커지면 분석 속도가 늦어질 수 있습니다.")


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
    # Upbit API는 한 번에 최대 200개 캔들만 제공하므로, 반복적으로 요청해야 합니다.
    # 요청 횟수를 제한하여 무한 루프 방지 및 API 호출 제한 준수
    max_requests = (end_date - start_date).days // 200 + 2 # 대략적인 필요한 요청 횟수 + 여유분
    
    st.info(f"🔄 업비트에서 **{symbol}** 데이터를 수집하고 있습니다.")
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
# ✨ FRED 거시 경제 데이터 로드 함수 (CPI 및 국채 금리)
# ------------------------
@st.cache_data(ttl=3600)
def load_fred_macro_data(start_date, end_date):
    """
    FRED API에서 소비자물가지수 (CPIAUCSL), 미국 10년물 국채 금리 (GS10),
    미국 2년물 국채 금리 (GS2) 데이터를 가져와 장단기 금리 스프레드를 계산합니다.
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
                'CPI 값': f"{cpi_series.loc[cpi_date]:.2f}",
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
# ✨ 장단기 금리 스프레드 특정 시점 암호화폐 가격 영향 분석 함수
# ------------------------
def analyze_yield_spread_impact(df_crypto, df_fred_macro, window_days=7):
    """
    미국 국채 장단기 금리 스프레드 데이터 포인트를 기준으로 암호화폐 가격의 변화를 분석합니다.
    """
    st.info(f"🔄 장단기 금리 스프레드 시점 암호화폐 가격 변화 분석 중 (기준일 후 {window_days}일 기준)...")
    
    analysis_results = []
    
    # 장단기 금리 스프레드 데이터가 유효한 날짜만 추출
    spread_dates = df_fred_macro['US_Yield_Spread'].dropna().index
    
    # 너무 많은 데이터 포인트로 인한 성능 저하를 막기 위해 월별로 샘플링
    # 예: 각 월의 첫 번째 유효한 스프레드 날짜만 사용
    sampled_spread_dates = []
    last_month = None
    for date in spread_dates.sort_values():
        if last_month is None or date.month != last_month:
            sampled_spread_dates.append(date)
            last_month = date.month

    for current_date in sampled_spread_dates:
        # 현재 날짜의 암호화폐 가격
        crypto_price_on_date = df_crypto['close'].asof(current_date)
        
        # 현재 날짜의 장단기 금리 스프레드 값
        yield_spread_value = df_fred_macro.loc[current_date, 'US_Yield_Spread']
        
        # 현재 날짜로부터 window_days 후의 암호화폐 가격
        future_date = current_date + timedelta(days=window_days)
        crypto_price_future = df_crypto['close'].asof(future_date)
        
        if pd.notna(crypto_price_on_date) and pd.notna(crypto_price_future) and pd.notna(yield_spread_value):
            price_change_percent = ((crypto_price_future - crypto_price_on_date) / crypto_price_on_date) * 100
            
            analysis_results.append({
                '기준일': current_date.strftime('%Y-%m-%d'),
                '장단기 금리 스프레드': f"{yield_spread_value:.2f}",
                f'{company_name} 가격 (기준일)': f"{crypto_price_on_date:,.2f}",
                f'{company_name} 가격 ({window_days}일 후)': f"{crypto_price_future:,.2f}",
                f'가격 변화율 ({window_days}일, %)': f"{price_change_percent:,.2f}%"
            })
            
    if not analysis_results:
        st.warning("⚠️ 장단기 금리 스프레드 데이터와 암호화폐 가격 데이터가 겹치는 유효한 분석 기간이 부족합니다. 날짜 범위를 조정해 보세요.")
        return pd.DataFrame()

    df_analysis = pd.DataFrame(analysis_results)
    
    st.success("✅ 장단기 금리 스프레드 시점 암호화폐 가격 변화 분석 완료!")
    return df_analysis


# ------------------------
# ✨ 예측 및 시각화 실행 버튼
# ------------------------
if st.button("🚀 LSTM 모델 학습 및 지표 시각화 실행"):
    with st.spinner("데이터 로드 및 전처리 중..."):
        df = load_crypto_data(symbol, start_date, end_date)
        
        if df.empty:
            st.error("암호화폐 데이터 로드에 실패하여 예측을 진행할 수 없습니다.")
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
    st.subheader("온체인 데이터 분석 (업데이트 중)")
    st.warning("⚠️ *업비트에서 온체인 데이터를 직접 제공하지 않기에 추후 다른 API를 통해 도입준비중입니다.**")
    st.info("""
    참고 : 온체인 데이터는 블록체인 네트워크 상의 실제 활동(예: 거래량, 활성 주소, 고래 움직임)을 보여주므로, 시장 심리와 추세를 파악하는 데 매우 중요합니다.
    
    온체인 데이터는 **Glassnode, CryptoQuant, CoinMetrics**와 같은 전문 온체인 데이터 분석 플랫폼에서 제공중인데,대부분 유료 구독 모델이기에 잔고 상황에 따라 업데이트가 달라질 수 있습니다. 
    
    -준비중인 서비스-
    **주요 온체인 지표 (예시):**
    - **고래 움직임 (Whale Movements):** 대규모 자금의 이동은 시장에 큰 영향을 미칠 수 있습니다.
    - **거래소 입출금량 (Exchange Inflows/Outflows):** 거래소로 코인이 유입되면 매도 압력, 유출되면 매수 압력으로 해석될 수 있습니다.
    - **해시레이트 (Hash Rate):** 비트코인 등 PoW 코인의 채굴 난이도 및 네트워크 보안성을 나타냅니다.
    - **미실현 손익 (Unrealized Profit/Loss):** 현재 가격 기준으로 코인 보유자들이 얼마나 이익 또는 손실을 보고 있는지 추정합니다.
        """)

    # ------------------------
    # ✨ 3. 거시 경제 지표 분석 및 시각화 (FRED 데이터)
    # ------------------------
    st.subheader("🌍 거시 경제 지표와 암호화폐 가격 영향 분석")
    
    # FRED 거시 경제 데이터 로드 (CPI, 10년물, 2년물)
    with st.spinner("📊 FRED 거시 경제 데이터를 불러오는 중..."):
        df_fred_macro = load_fred_macro_data(start_date, end_date)

    if df_fred_macro.empty:
        st.error("FRED 거시 경제 데이터를 로드할 수 없어 거시 경제 지표 분석을 진행할 수 없습니다. FRED API 키를 확인하거나 날짜 범위를 조정해 보세요.")
        st.stop()

    # --- CPI 발표 시점 분석 및 시각화 ---
    st.markdown("### 📈 CPI 발표 시점 암호화폐 가격 영향 분석")
    df_cpi_impact = analyze_cpi_impact(df, df_fred_macro['CPI'].dropna(), window_days=7) # df_results 대신 df 사용

    if not df_cpi_impact.empty:
        st.markdown("#### CPI 발표 시점 암호화폐 가격 변화 (표)")
        st.dataframe(df_cpi_impact)

        st.markdown(f"#### {company_name} 가격 추이 및 CPI 발표 시점")
        fig_price_cpi = go.Figure()

        # 암호화폐 가격 라인
        fig_price_cpi.add_trace(go.Scatter(x=df.index, y=df['close'], # df_results 대신 df 사용
                                           mode='lines', name=f'{company_name} 가격', line=dict(color='blue')))
        
        # CPI 발표 시점 세로선 추가
        for cpi_date in df_fred_macro['CPI'].dropna().index:
            if cpi_date in df.index: # 암호화폐 데이터에 해당 날짜가 있는 경우에만 표시 (df_results 대신 df 사용)
                fig_price_cpi.add_vline(x=cpi_date, line_width=1, line_dash="dot", line_color="red",
                                        annotation_text=f"CPI({cpi_date.strftime('%Y-%m')})",
                                        annotation_position="top right",
                                        annotation_font_size=10,
                                        annotation_font_color="red")

        fig_price_cpi.update_layout(height=600, title_text=f"{company_name} 가격 추이와 CPI 발표 시점",
                                    xaxis_rangeslider_visible=True) # 범위 슬라이더 추가
        fig_price_cpi.update_xaxes(showgrid=True, tickangle=45)
        st.plotly_chart(fig_price_cpi, use_container_width=True)
    else:
        st.warning("CPI 분석을 위한 데이터가 부족합니다. 날짜 범위를 확인하거나 FRED API 키를 점검하세요.")
    
    st.markdown("---") # 구분선 추가

    # --- 장단기 금리 스프레드 특정 시점 분석 및 시각화 ---
    st.markdown("### 📈 미국 국채 장단기 금리 스프레드 추이 및 암호화폐 가격 영향 분석")
    df_yield_spread_impact = analyze_yield_spread_impact(df, df_fred_macro, window_days=7) # df_results 대신 df 사용

    if not df_yield_spread_impact.empty:
        st.markdown("#### 장단기 금리 스프레드 시점 암호화폐 가격 변화 (표)")
        st.dataframe(df_yield_spread_impact)

        st.markdown(f"#### {company_name} 가격 추이 및 장단기 금리 스프레드 변화 시점")
        fig_price_spread = go.Figure()

        # 암호화폐 가격 라인
        fig_price_spread.add_trace(go.Scatter(x=df.index, y=df['close'], # df_results 대신 df 사용
                                              mode='lines', name=f'{company_name} 가격', line=dict(color='blue')))
        
        # 장단기 금리 스프레드 데이터 포인트 세로선 추가
        # 여기서는 스프레드 값이 존재하는 모든 날짜를 기준으로 합니다.
        for spread_date in df_fred_macro['US_Yield_Spread'].dropna().index:
            if spread_date in df.index: # 암호화폐 데이터에 해당 날짜가 있는 경우에만 표시 (df_results 대신 df 사용)
                fig_price_spread.add_vline(x=spread_date, line_width=1, line_dash="dot", line_color="purple",
                                            annotation_text=f"스프레드({spread_date.strftime('%Y-%m-%d')})",
                                            annotation_position="top right",
                                            annotation_font_size=10,
                                            annotation_font_color="purple")

        fig_price_spread.update_layout(height=600, title_text=f"{company_name} 가격 추이와 장단기 금리 스프레드 변화 시점",
                                       xaxis_rangeslider_visible=True) # 범위 슬라이더 추가
        fig_price_spread.update_xaxes(showgrid=True, tickangle=45)
        st.plotly_chart(fig_price_spread, use_container_width=True)
    else:
        st.warning("장단기 금리 스프레드 분석을 위한 데이터가 부족합니다. 날짜 범위를 확인하거나 FRED API 키를 점검하세요.")

    st.markdown("---")
    st.write("### 📝 추가 참고 사항")
    st.write("""
    - **데이터 기간**: 기본적으로 지난 5년간의 데이터를 사용합니다. 원하는 기간으로 조정하여 데이터를 확인해 보세요.
    - **CPI/스프레드 기준일**: FRED에서 제공하는 데이터의 인덱스 날짜를 기준으로 분석을 수행합니다. 실제 발표일/변화 시점과는 약간의 차이가 있을 수 있습니다.
    - **가격 변화율**: 기준일의 종가와 기준일로부터 7일 후의 종가를 기준으로 계산됩니다. 주말이나 공휴일로 인해 7일 후의 정확한 데이터가 없는 경우, 가장 가까운 유효한 날짜의 데이터가 사용됩니다.
    - **장단기 금리 스프레드**: 10년물 국채 금리에서 2년물 국채 금리를 뺀 값입니다. 이 값이 0보다 작아지면 (음수가 되면) '장단기 금리 역전'이라고 하며, 이는 종종 경기 침체의 전조로 해석되기도 합니다.
    """)

#####################샘플
# import streamlit as st
# import pandas as pd
# import requests
# from datetime import datetime, timedelta
# import numpy as np
# import matplotlib.pyplot as plt
# # import matplotlib.font_manager as fm # 한글 폰트 관련 모듈 제거 (요청에 따라)
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from json.decoder import JSONDecodeError
# import time
# from fredapi import Fred # FRED API를 위한 라이브러리
# from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
# import urllib.error # HTTPError를 위해 임포트
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots # make_subplots 임포트 추가

# # --- 페이지 설정 ---
# st.set_page_config(page_title="암호화폐 예측 및 지표 분석", layout="wide")
# st.title("📈 암호화폐 LSTM 예측 및 다양한 지표 분석")

# st.markdown("""
# Upbit API를 통해 암호화폐 가격 데이터를 가져와 LSTM 딥러닝 모델로 미래 가격을 예측하고,
# 다양한 기술적 지표, 온체인 데이터(설명), 거시 경제 지표를 함께 시각화하여 분석합니다.
# """)

# # ------------------------
# # ✨ 한글 폰트 설정 (제거됨)
# # ------------------------
# # Streamlit Cloud 환경에서 기본 폰트가 한글을 지원하지 않을 경우,
# # 차트의 한글 텍스트가 깨져 보일 수 있습니다.
# # 이 경우, Streamlit 앱 배포 환경에 한글 폰트를 설치하거나
# # Plotly 등 다른 시각화 라이브러리를 고려할 수 있습니다.
# plt.rc('axes', unicode_minus=False) # 마이너스 폰트 깨짐 방지 (일반적인 설정이므로 유지)

# # ------------------------
# # ✨ FRED API 설정
# # ------------------------
# try:
#     # 이 부분이 이전 오류의 원인이었습니다. 정확히 수정되었습니다.
#     FRED_API_KEY = st.secrets["FRED_API_KEY"]
#     fred = Fred(api_key=FRED_API_KEY)
# except KeyError:
#     st.warning("🚨 FRED API 키('FRED_API_KEY')가 Streamlit Secrets에 설정되어 있지 않습니다. 거시 경제 지표는 로드되지 않습니다.")
#     fred = None # FRED API 키가 없으면 fred 객체를 None으로 설정

# # --- 재시도 데코레이터 설정 (FRED API용) ---
# @retry(
#     wait=wait_exponential(multiplier=1, min=1, max=10),
#     stop=stop_after_attempt(3),
#     retry=retry_if_exception_type((urllib.error.HTTPError, ConnectionResetError, ValueError)), # ValueError 추가
#     reraise=True
# )
# def fetch_fred_series_with_retry(series_id, start_date, end_date):
#     """
#     FRED API에서 데이터를 가져오는 함수에 재시도 로직을 추가합니다.
#     데이터가 없거나 비어있으면 ValueError를 발생시켜 재시도를 유도합니다.
#     """
#     if fred:
#         series = fred.get_series(series_id, start_date, end_date)
#         if series is None or series.empty:
#             # 명시적으로 ValueError를 발생시켜 tenacity 재시도 유도
#             raise ValueError(f"FRED series '{series_id}' returned no data for the period {start_date} to {end_date}.")
#         return series
#     return None # FRED 객체가 없을 경우

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
# # ✨ 기술적 지표 계산 함수
# # ------------------------
# def calculate_technical_indicators(df):
#     """
#     DataFrame에 모멘텀, RSI, MACD, OBV를 추가합니다.
#     """
#     df['MA20'] = df['close'].rolling(window=20).mean()
#     df['MA60'] = df['close'].rolling(window=60).mean()

#     # 모멘텀 (14일)
#     df['Momentum'] = df['close'].pct_change(14) * 100

#     # RSI (14일)
#     delta = df['close'].diff(1)
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = np.where(loss == 0, np.inf, gain / loss)
#     df['RSI'] = 100 - (100 / (1 + rs))

#     # MACD (12, 26, 9)
#     exp1 = df['close'].ewm(span=12, adjust=False).mean()
#     exp2 = df['close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = exp1 - exp2
#     df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
#     df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

#     # OBV (On-Balance Volume)
#     obv_values = np.zeros(len(df))
#     if len(df) > 0:
#         obv_values[0] = df['volume'].iloc[0]
#     for k in range(1, len(df)):
#         if df['close'].iloc[k] > df['close'].iloc[k-1]:
#             obv_values[k] = obv_values[k-1] + df['volume'].iloc[k]
#         elif df['close'].iloc[k] < df['close'].iloc[k-1]:
#             obv_values[k] = obv_values[k-1] - df['volume'].iloc[k]
#         else:
#             obv_values[k] = obv_values[k-1]
#     df['OBV'] = obv_values
    
#     return df

# # ------------------------
# # ✨ FRED 데이터 로드 함수
# # ------------------------
# @st.cache_data(ttl=3600)
# def load_fred_indicators(start_date, end_date):
#     """
#     FRED API에서 CPI와 미국 10년물 국채 금리 데이터를 가져옵니다.
#     """
#     econ_data = {}
#     econ_errors = []

#     if not fred: # FRED API 키가 없으면 함수 종료
#         return pd.DataFrame()

#     # 1. 소비자물가지수 (CPIAUCSL) - 월별
#     try:
#         cpi = fetch_fred_series_with_retry('CPIAUCSL', start_date, end_date)
#         econ_data['CPI'] = cpi.rename("CPI")
#         st.info(f"✅ CPI 데이터 로드: {cpi.index.min().date()} ~ {cpi.index.max().date()}")
#     except Exception as e:
#         econ_errors.append(f"❌ 소비자물가지수(CPI) 로드 중 오류 발생: {e}")

#     # 2. 미국 10년물 국채 금리 (GS10) - 일별
#     try:
#         us_10y = fetch_fred_series_with_retry('GS10', start_date, end_date)
#         econ_data['US_10Y_Yield'] = us_10y.rename("US_10Y_Yield")
#         st.info(f"✅ 미국 10년물 국채 금리 로드: {us_10y.index.min().date()} ~ {us_10y.index.max().date()}")
#     except Exception as e:
#         econ_errors.append(f"❌ 미국 10년물 국채 금리 로드 중 오류 발생: {e}")

#     if econ_errors:
#         for err in econ_errors:
#             st.error(err)
#         st.warning("일부 거시 경제 지표 데이터 로드에 실패했습니다. 해당 그래프가 올바르게 표시되지 않을 수 있습니다.")
#         return pd.DataFrame()

#     econ_df = pd.DataFrame()
#     for key, series in econ_data.items():
#         if not series.empty:
#             econ_df = pd.concat([econ_df, series], axis=1)

#     econ_df.index = pd.to_datetime(econ_df.index)
#     # 월별 데이터를 일별 데이터로 채우기 (CPI)
#     econ_df = econ_df.resample('D').ffill()
#     econ_df = econ_df.dropna(how='all') # 모든 컬럼이 NaN인 행 제거

#     if econ_df.empty:
#         st.warning("선택된 기간에 유효한 거시 경제 지표 데이터를 충분히 불러오지 못했습니다. 날짜 범위를 조정해 보세요.")
#         return pd.DataFrame()

#     st.success(f"✅ 거시 경제 지표 데이터 로드 완료! ({econ_df.index.min().date()} ~ {econ_df.index.max().date()})")
#     return econ_df


# # ------------------------
# # ✨ 예측 및 시각화 실행 버튼
# # ------------------------
# if st.button("🚀 LSTM 모델 학습 및 지표 시각화 실행"):
#     with st.spinner("데이터 로드 및 전처리 중..."):
#         df = load_crypto_data(symbol, start_date, end_date)
        
#         if df.empty:
#             st.error("데이터 로드에 실패하여 예측을 진행할 수 없습니다.")
#             st.stop()

#         # 기술적 지표 계산
#         df_with_indicators = calculate_technical_indicators(df.copy())
        
#         # 'close' 가격만 사용 (LSTM 예측용)
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
#                             validation_split=0.1, 
#                             callbacks=[early_stopping],
#                             verbose=0)
        
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
#     # ✨ 1. LSTM 예측 및 기술적 지표 시각화
#     # ------------------------
#     st.subheader("📊 LSTM 예측 및 기술적 지표 시각화")
#     # 5개의 서브플롯: 가격(LSTM), 모멘텀, RSI, MACD, OBV
#     fig_tech = make_subplots(rows=5, cols=1, shared_xaxes=True, 
#                              vertical_spacing=0.05,
#                              row_width=[0.4, 0.15, 0.15, 0.15, 0.15]) # 비율 조정

#     # 1행: 실제 가격 및 LSTM 예측
#     fig_tech.add_trace(go.Scatter(x=df_results.index, y=df_results['실제 가격'], 
#                                   mode='lines', name='실제 가격', line=dict(color='blue')), row=1, col=1)
#     fig_tech.add_trace(go.Scatter(x=df_results.index, y=df_results['학습 예측'], 
#                                   mode='lines', name='학습 예측', line=dict(color='green', dash='dot')), row=1, col=1)
#     fig_tech.add_trace(go.Scatter(x=df_results.index, y=df_results['테스트 예측'], 
#                                   mode='lines', name='테스트 예측', line=dict(color='red', dash='dot')), row=1, col=1)
#     fig_tech.update_yaxes(title_text="가격", row=1, col=1)


#     # 2행: 모멘텀
#     fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Momentum'], 
#                                   mode='lines', name='모멘텀', line=dict(color='purple')), row=2, col=1)
#     fig_tech.update_yaxes(title_text="모멘텀", row=2, col=1)
#     fig_tech.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1) # 0선 추가

#     # 3행: RSI
#     fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['RSI'], 
#                                   mode='lines', name='RSI', line=dict(color='orange')), row=3, col=1)
#     fig_tech.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
#     fig_tech.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
#     fig_tech.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

#     # 4행: MACD
#     fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD'], 
#                                   mode='lines', name='MACD', line=dict(color='blue')), row=4, col=1)
#     fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD_Signal'], 
#                                   mode='lines', name='Signal', line=dict(color='red', dash='dot')), row=4, col=1)
#     # MACD 히스토그램 (바 차트)
#     colors_macd_hist = ['rgba(0,255,0,0.5)' if val >= 0 else 'rgba(255,0,0,0.5)' for val in df_with_indicators['MACD_Hist']]
#     fig_tech.add_trace(go.Bar(x=df_with_indicators.index, y=df_with_indicators['MACD_Hist'], 
#                                name='MACD Hist', marker_color=colors_macd_hist), row=4, col=1)
#     fig_tech.update_yaxes(title_text="MACD", row=4, col=1)
#     fig_tech.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)

#     # 5행: OBV
#     fig_tech.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['OBV'], 
#                                   mode='lines', name='OBV', line=dict(color='darkgreen')), row=5, col=1)
#     fig_tech.update_yaxes(title_text="OBV", row=5, col=1)

#     fig_tech.update_layout(height=1000, title_text=f"{company_name} 가격 예측 및 기술적 지표",
#                             xaxis_rangeslider_visible=False)
#     fig_tech.update_xaxes(showgrid=True, tickangle=45)
#     st.plotly_chart(fig_tech, use_container_width=True)


#     # ------------------------
#     # ✨ 2. 온체인 데이터 시각화 (설명)
#     # ------------------------
#     st.subheader("🔗 온체인 데이터 분석 (개념 설명)")
#     st.warning("⚠️ **Upbit API는 온체인 데이터를 직접 제공하지 않습니다.**")
#     st.info("""
#     온체인 데이터는 블록체인 네트워크 상의 실제 활동(예: 거래량, 활성 주소, 고래 움직임)을 보여주므로, 시장 심리와 추세를 파악하는 데 매우 중요합니다.
#     하지만 Upbit과 같은 중앙화된 거래소의 API는 주로 거래소 내부의 가격 및 주문 정보만을 제공합니다.
    
#     온체인 데이터를 활용하려면 **Glassnode, CryptoQuant, CoinMetrics**와 같은 전문 온체인 데이터 분석 플랫폼의 API를 사용해야 합니다.
#     이러한 서비스는 대부분 유료 구독 모델을 제공합니다.
    
#     **주요 온체인 지표 (예시):**
#     - **고래 움직임 (Whale Movements):** 대규모 자금의 이동은 시장에 큰 영향을 미칠 수 있습니다.
#     - **거래소 입출금량 (Exchange Inflows/Outflows):** 거래소로 코인이 유입되면 매도 압력, 유출되면 매수 압력으로 해석될 수 있습니다.
#     - **해시레이트 (Hash Rate):** 비트코인 등 PoW 코인의 채굴 난이도 및 네트워크 보안성을 나타냅니다.
#     - **미실현 손익 (Unrealized Profit/Loss):** 현재 가격 기준으로 코인 보유자들이 얼마나 이익 또는 손실을 보고 있는지 추정합니다.
    
#     이러한 데이터를 가져올 수 있다면, 가격 데이터와 결합하여 모델의 예측 정확도를 더욱 높일 수 있습니다.
#     """)

#     # ------------------------
#     # ✨ 3. 거시 경제 지표 시각화 (FRED 데이터) - 분리된 차트
#     # ------------------------
#     st.subheader("🌍 거시 경제 지표와 암호화폐 가격 비교")
    
#     # FRED 데이터 로드 (기본 시작 날짜를 10년 전으로 설정)
#     default_fred_start_date = datetime.today() - timedelta(days=365 * 10) 
#     df_fred = load_fred_indicators(default_fred_start_date, end_date)

#     if not df_fred.empty:
#         # --- 3-1. 암호화폐 가격 vs. 소비자물가지수 (CPI) ---
#         st.markdown("#### 📈 암호화폐 가격 vs. 소비자물가지수 (CPI)")
#         df_cpi_combined = pd.merge(df_results[['실제 가격']], df_fred[['CPI']], 
#                                    left_index=True, right_index=True, how='inner')
#         df_cpi_combined = df_cpi_combined.dropna()

#         if df_cpi_combined.empty:
#             st.warning("선택된 기간에 암호화폐 가격과 CPI 데이터를 모두 포함하는 데이터가 충분하지 않습니다. 날짜 범위를 조정해 보세요.")
#         else:
#             st.success(f"✅ 암호화폐 가격-CPI 결합 데이터 로드 완료! ({df_cpi_combined.index.min().date()} ~ {df_cpi_combined.index.max().date()})")
#             fig_cpi = make_subplots(rows=2, cols=1, shared_xaxes=True,
#                                     vertical_spacing=0.1,
#                                     row_width=[0.5, 0.5])

#             fig_cpi.add_trace(go.Scatter(x=df_cpi_combined.index, y=df_cpi_combined['실제 가격'],
#                                          mode='lines', name=f'{company_name} 실제 가격', line=dict(color='blue')), row=1, col=1)
#             fig_cpi.update_yaxes(title_text=f"{company_name} 가격", row=1, col=1)

#             fig_cpi.add_trace(go.Scatter(x=df_cpi_combined.index, y=df_cpi_combined['CPI'],
#                                          mode='lines', name='소비자물가지수 (CPI)', line=dict(color='orange')), row=2, col=1)
#             fig_cpi.update_yaxes(title_text="CPI", row=2, col=1)

#             fig_cpi.update_layout(height=600, title_text=f"{company_name} 가격과 CPI 비교",
#                                   xaxis_rangeslider_visible=False)
#             fig_cpi.update_xaxes(showgrid=True, tickangle=45)
#             st.plotly_chart(fig_cpi, use_container_width=True)
        
#         st.markdown("---") # 구분선 추가

#         # --- 3-2. 암호화폐 가격 vs. 미국 10년물 국채 금리 ---
#         st.markdown("#### 📈 암호화폐 가격 vs. 미국 10년물 국채 금리")
#         df_us10y_combined = pd.merge(df_results[['실제 가격']], df_fred[['US_10Y_Yield']], 
#                                      left_index=True, right_index=True, how='inner')
#         df_us10y_combined = df_us10y_combined.dropna()

#         if df_us10y_combined.empty:
#             st.warning("선택된 기간에 암호화폐 가격과 미국 10년물 국채 금리 데이터를 모두 포함하는 데이터가 충분하지 않습니다. 날짜 범위를 조정해 보세요.")
#         else:
#             st.success(f"✅ 암호화폐 가격-미국 10년물 국채 금리 결합 데이터 로드 완료! ({df_us10y_combined.index.min().date()} ~ {df_us10y_combined.index.max().date()})")
#             fig_us10y = make_subplots(rows=2, cols=1, shared_xaxes=True,
#                                       vertical_spacing=0.1,
#                                       row_width=[0.5, 0.5])

#             fig_us10y.add_trace(go.Scatter(x=df_us10y_combined.index, y=df_us10y_combined['실제 가격'],
#                                           mode='lines', name=f'{company_name} 실제 가격', line=dict(color='blue')), row=1, col=1)
#             fig_us10y.update_yaxes(title_text=f"{company_name} 가격", row=1, col=1)

#             fig_us10y.add_trace(go.Scatter(x=df_us10y_combined.index, y=df_us10y_combined['US_10Y_Yield'],
#                                           mode='lines', name='미국 10년물 국채 금리', line=dict(color='green')), row=2, col=1)
#             fig_us10y.update_yaxes(title_text="미국 10년물 금리 (%)", row=2, col=1)

#             fig_us10y.update_layout(height=600, title_text=f"{company_name} 가격과 미국 10년물 국채 금리 비교",
#                                     xaxis_rangeslider_visible=False)
#             fig_us10y.update_xaxes(showgrid=True, tickangle=45)
#             st.plotly_chart(fig_us10y, use_container_width=True)

#     else:
#         st.warning("거시 경제 지표를 로드할 수 없어 시각화를 건너킵니다. FRED API 키를 확인하거나 날짜 범위를 조정해 보세요.")

#     st.markdown("---")
#     st.write("### 📝 추가 참고 사항")
#     st.write("""
#     - **데이터 통합**: 다양한 유형의 데이터를 모델에 통합할 때는 각 데이터의 빈도(일별, 월별 등)를 맞추는 것이 중요합니다. (예: 월별 데이터를 일별로 `ffill` 하는 방식)
#     - **피처 엔지니어링**: 단순히 원시 데이터를 사용하는 것을 넘어, 각 지표의 변화율, 이동평균선과의 관계, 특정 임계값 돌파 여부 등 새로운 피처를 생성하여 모델의 학습 능력을 향상시킬 수 있습니다.
#     - **모델 복잡도**: 더 많은 피처를 사용할수록 모델의 복잡도가 증가하며, 과적합(Overfitting) 위험이 커질 수 있습니다. 적절한 정규화(Regularization) 기법(예: Dropout)과 검증을 통해 이를 관리해야 합니다.
#     - **해석의 어려움**: 다양한 팩터를 포함할수록 모델의 '블랙박스' 특성이 강해져 예측 결과의 원인을 해석하기 어려워질 수 있습니다.
#     """)
