import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from json.decoder import JSONDecodeError
import time

# --- 페이지 설정 ---
st.set_page_config(page_title="암호화폐 LSTM 가격 예측", layout="wide")
st.title("📈 암호화폐 LSTM 가격 예측 및 시각화")

st.markdown("""
Upbit API를 통해 암호화폐 가격 데이터를 가져와 LSTM 딥러닝 모델로
미래 가격을 예측하고 시각화합니다.
""")

# ------------------------
# ✨ 한글 폰트 설정
# ------------------------
def get_korean_font():
    """matplotlib에서 한글 폰트 설정을 시도합니다."""
    font_path = ""
    for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'NanumGothic' in font:
            font_path = font
            break
        elif 'Malgun Gothic' in font:
            font_path = font
            break
        elif 'AppleGothic' in font:
            font_path = font
            break
    
    if font_path:
        fm.fontManager.addfont(font_path)
        plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
        plt.rc('axes', unicode_minus=False) # 마이너스 폰트 깨짐 방지
        st.info(f"✅ 한글 폰트 '{fm.FontProperties(fname=font_path).get_name()}'가 성공적으로 설정되었습니다.")
    else:
        st.warning("⚠️ 시스템에 한글 폰트(나눔고딕, 맑은고딕 등)가 설치되어 있지 않습니다. 차트의 한글이 깨질 수 있습니다.")

get_korean_font()

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
# ✨ 예측 실행 버튼
# ------------------------
if st.button("🚀 LSTM 모델 학습 및 예측 실행"):
    with st.spinner("데이터 로드 및 전처리 중..."):
        df = load_crypto_data(symbol, start_date, end_date)
        
        if df.empty:
            st.error("데이터 로드에 실패하여 예측을 진행할 수 없습니다.")
            st.stop()

        # 'close' 가격만 사용
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
                            validation_split=0.1, # 학습 데이터의 10%를 검증에 사용
                            callbacks=[early_stopping],
                            verbose=0) # Streamlit에서는 verbose를 0으로 설정하여 출력 줄임
        
        st.success("✅ LSTM 모델 학습 완료!")
        # st.write(f"최종 학습 손실 (Loss): {history.history['loss'][-1]:.4f}")
        # if 'val_loss' in history.history:
        #     st.write(f"최종 검증 손실 (Validation Loss): {history.history['val_loss'][-1]:.4f}")

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
    # ✨ 시각화
    # ------------------------
    st.subheader("📊 실제 가격 vs. LSTM 예측 가격")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_results.index, df_results['실제 가격'], label='실제 가격', color='blue')
    ax.plot(df_results.index, df_results['학습 예측'], label='학습 예측', color='green', linestyle='--')
    ax.plot(df_results.index, df_results['테스트 예측'], label='테스트 예측', color='red', linestyle='--')
    
    ax.set_title(f"{company_name} 가격 예측 (LSTM)")
    ax.set_xlabel("날짜")
    ax.set_ylabel("가격")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("---")
    st.write("### 📝 참고")
    st.write("""
    - **LSTM 모델**: 과거 데이터를 기반으로 미래 값을 예측하는 딥러닝 모델입니다. `look_back` 기간 동안의 데이터를 사용하여 다음 날의 가격을 예측합니다.
    - **데이터 정규화**: 모델 학습 효율을 높이기 위해 데이터를 0과 1 사이로 스케일링합니다. 예측 후 다시 원래 스케일로 역변환합니다.
    - **학습/테스트 분할**: 전체 데이터 중 일부를 학습에 사용하고, 나머지는 모델이 얼마나 잘 예측하는지 평가하는 데 사용합니다.
    - **예측의 한계**: 암호화폐 시장은 변동성이 매우 크고 다양한 외부 요인에 의해 영향을 받으므로, 딥러닝 모델도 완벽하게 예측하기는 어렵습니다. 이 앱은 예측 모델의 개념을 보여주는 예시이며, 실제 투자에 활용하기에는 추가적인 연구와 검증이 필요합니다.
    """)

