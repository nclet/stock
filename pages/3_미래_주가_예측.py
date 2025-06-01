# streamlit_test/pages/3_미래_주가_예측.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os # os 모듈 임포트

# 딥러닝 관련 라이브러리 임포트 (설치 필요: pip install tensorflow scikit-learn)
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    st.error("""
    **미래 주가 예측 기능을 사용하려면 다음 라이브러리를 설치해야 합니다:**
    `pip install tensorflow scikit-learn`
    """)
    st.stop() # 라이브러리 설치가 안 되어 있으면 앱 실행을 멈춥니다.


# ---
## Streamlit 페이지 설정
# st.set_page_config()는 반드시 파일의 가장 첫 번째 Streamlit 명령이어야 합니다.
st.set_page_config(layout="wide")

st.title("🔮 미래 주가 예측 (LSTM 기반)")
st.markdown("과거 주가 데이터와 기술적/펀더멘털 지표를 활용하여 미래 주가를 예측합니다.")

# ---
## 기술적 지표 계산 함수
@st.cache_data
def calculate_bollinger_bands_pred(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

@st.cache_data
def calculate_rsi_pred(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan) # 0으로 나누는 오류 방지
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---
## LSTM 모델 관련 함수
def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

#@st.cache_resource # 모델 학습 결과를 캐싱 (재실행 시 재학습 방지)
def train_and_predict_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_future_days, last_sequence, scaler):
    model_path = f"model_{selected_code}.h5"
    model = None

    if os.path.exists(model_path):
        model = load_model(model_path)
        st.success("✅ 저장된 모델 로드 완료")
    else:
        st.info("모델 학습이 필요합니다. 잠시만 기다려 주세요...")
        model = build_model(input_shape=(seq_len, n_features))
        with st.spinner("🔄 모델 학습 중 (시간이 다소 소요될 수 있습니다)..."):
            model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
        model.save(model_path)
        st.success("✅ 모델 학습 및 저장 완료")

    # 예측 함수
    def recursive_forecast(model, last_sequence, n_days, scaler, n_features):
        forecasts = []
        current_seq = last_sequence.copy()

        for _ in range(n_days):
            # LSTM 모델은 3D 입력 (samples, timesteps, features)을 기대합니다.
            # current_seq는 (timesteps, features) 형태이므로, 앞에 samples 차원을 추가합니다.
            pred = model.predict(current_seq.reshape(1, seq_len, n_features), verbose=0)[0][0]
            forecasts.append(pred)

            # 다음 예측을 위해 시퀀스 업데이트:
            # 가장 간단한 방법은 예측된 종가(pred)를 새로운 날의 모든 특징 값으로 사용하는 것입니다.
            # 하지만 이는 다른 특징(RSI, BB, PER, PBR)이 종가와 함께 움직인다고 가정하는 매우 단순한 방식입니다.
            # 더 정확한 예측을 위해서는 각 특징의 미래 값을 예측하거나, 외부 데이터를 가져와야 합니다.
            # 여기서는 편의를 위해 예측된 종가로 모든 특징 벡터를 구성합니다.
            new_feature_vector = np.full(n_features, pred)
            current_seq = np.vstack([current_seq[1:], new_feature_vector]) # 가장 오래된 데이터를 버리고 새 벡터 추가

        # 예측값을 역 스케일링하여 실제 주가 범위로 되돌립니다.
        # 스케일러는 fit_transform될 때 사용된 특징들의 순서를 기억합니다.
        # 예측값은 단일 컬럼이므로, 스케일러가 기대하는 2D 형태로 맞춰준 후 첫 번째 컬럼을 가져옵니다.
        # (원래 'Close'가 features 리스트의 첫 번째였으므로, 예측값도 Close의 스케일러를 사용해야 합니다.)
        dummy_array_for_inverse = np.zeros((len(forecasts), n_features))
        dummy_array_for_inverse[:, features.index('Close')] = forecasts # 예측값을 'Close' 위치에만 넣기
        forecasts_scaled = scaler.inverse_transform(dummy_array_for_inverse)[:, features.index('Close')]
        return forecasts_scaled

    future_preds = recursive_forecast(model, last_sequence, n_future_days, scaler, n_features)
    return future_preds

# ---
## Streamlit UI
# 데이터 로드
@st.cache_data
def load_merged_data():
    try:
        # ⚠️ 여기가 핵심 수정 부분입니다!
        # 현재 스크립트 파일 (3_미래_주가_예측.py)의 위치를 기준으로
        # 'merged_data_monthly_per_pbr.csv' 파일이 있는 상위 디렉토리로 이동하여 경로를 구성합니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 예를 들어, 스크립트가 'streamlit_test/pages/'에 있다면, root_dir은 'streamlit_test/'가 됩니다.
        root_dir = os.path.join(current_dir, '..')
        merged_data_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')

        df = pd.read_csv(merged_data_file_path) # 수정된 경로 사용
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str).str.zfill(6) # 종목코드 6자리로 채우기
        st.success(f"✅ 'merged_data_monthly_per_pbr.csv' 파일을 성공적으로 불러왔습니다. (경로: {merged_data_file_path})")
        return df
    except FileNotFoundError:
        st.error(f"❌ 'merged_data_monthly_per_pbr.csv' 파일을 찾을 수 없습니다. 예상 경로: {merged_data_file_path}")
        return pd.DataFrame() # 빈 데이터프레임을 반환하여 이후 오류 방지
    except Exception as e: # 파일을 찾았지만 읽는 중 다른 오류 발생 시
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return pd.DataFrame() # 빈 데이터프레임을 반환하여 이후 오류 방지

df_all_data = load_merged_data()

# 데이터 로드 성공 여부 확인
if not df_all_data.empty:
    # PER/PBR 컬럼 존재 여부 확인 로직
    # 컬럼이 존재하면 경고 메시지가 뜨지 않도록 합니다.
    if 'PER' not in df_all_data.columns or 'PBR' not in df_all_data.columns:
        st.warning("경고: 데이터 파일에 'PER' 또는 'PBR' 컬럼이 없어 예측에 사용되지 않습니다. 해당 컬럼이 없으면 정확도가 떨어질 수 있습니다.")
    else:
        st.info("데이터 파일에서 'PER' 및 'PBR' 컬럼을 성공적으로 확인했습니다.")

    name_code_dict = df_all_data.drop_duplicates(subset=['Code']).set_index('Name')['Code'].to_dict()

    # name_code_dict가 비어있을 수 있는 경우 (데이터프레임에 유효한 Name/Code가 없을 때) 처리
    if not name_code_dict:
        st.error("종목 리스트를 생성할 수 없습니다. 데이터 파일에 'Name' 또는 'Code' 컬럼이 올바르지 않은지 확인해주세요.")
        st.stop()

    selected_name = st.selectbox("🔮 예측할 종목을 선택하세요", sorted(name_code_dict.keys()))
    selected_code = name_code_dict[selected_name]

    n_days = st.slider("예측할 미래 일 수", 5, 60, 30)

    if st.button("🚀 주가 예측 시작"):
        df_stock = df_all_data[df_all_data['Code'] == selected_code].copy()
        df_stock.sort_values('Date', inplace=True)
        df_stock.set_index('Date', inplace=True) # 날짜를 인덱스로 설정

        # 주식 데이터 부족 시 처리
        if df_stock.empty:
            st.error(f"선택하신 종목 ({selected_name})에 대한 데이터가 없습니다. 다른 종목을 선택해주세요.")
            st.stop()

        # 필요한 컬럼 추가 (PER/PBR이 없는 경우 0으로 채움)
        df_stock['RSI'] = calculate_rsi_pred(df_stock['Close'])
        df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands_pred(df_stock['Close'])

        # PER/PBR 컬럼이 없으면 0으로 채우는 로직은 그대로 유지 (안전 장치)
        if 'PER' not in df_stock.columns:
            df_stock['PER'] = 0.0
        if 'PBR' not in df_stock.columns:
            df_stock['PBR'] = 0.0

        # 예측에 사용할 특징 (feature) 컬럼 정의
        features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR']
        target = 'Close' # 예측 대상은 'Close'

        # 예측에 필요한 데이터만 선택하고 NaN 값 제거
        df_processed = df_stock[features + [target]].dropna()

        seq_len = 20 # LSTM 시퀀스 길이 (과거 20일 데이터로 다음 날 예측)

        if len(df_processed) < seq_len + 1: # 최소 시퀀스 길이 + 1 (타겟값)
            st.warning(f"데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 최소 {seq_len + 1}일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_processed)}일)")
            st.stop()

        # 스케일링
        scaler = MinMaxScaler()
        # 모든 특징을 동시에 스케일링 (학습에 사용될 모든 특징)
        scaled_data = scaler.fit_transform(df_processed[features])

        X, y = [], []
        # 시퀀스 데이터 생성
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i:i+seq_len]) # 과거 seq_len일간의 특징 시퀀스
            y.append(scaled_data[i+seq_len, features.index(target)]) # 시퀀스 다음 날의 타겟 (Close) 값

        if not X: # X가 비어있을 경우 (데이터가 너무 적어 시퀀스 생성이 안 될 때)
            st.warning(f"데이터 전처리 후 남은 데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 시퀀스 길이를 조절하거나 더 많은 데이터를 확보해주세요.")
            st.stop()

        X, y = np.array(X), np.array(y)

        # 학습/테스트 데이터 분리 (시계열 데이터이므로 순서를 유지하며 분리)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        last_sequence = X[-1] # 예측의 시작점이 될 마지막 시퀀스
        n_features = X.shape[2] # 모델 입력 특징의 개수

        # 모델 학습 및 예측 함수 호출
        future_preds = train_and_predict_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_days, last_sequence, scaler)

        if future_preds is None: # 모델 학습/예측 실패 시 (예외 처리)
            st.error("미래 주가 예측에 실패했습니다. 데이터를 확인하거나 다시 시도해주세요.")
            st.stop()

        # 예측 날짜 생성
        last_date = df_processed.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]

        # ---
        ## 예측 결과 시각화
        st.subheader("📊 실제 주가 및 미래 예측 주가")
        fig, ax = plt.subplots(figsize=(12, 6))

        # 실제 주가 (최근 일부만 시각화하여 예측과 비교 용이)
        # 전체 데이터가 너무 많으면 그래프가 복잡해지므로 최근 1년(365일)치만 표시
        plot_df = df_processed.tail(365)
        ax.plot(plot_df.index, plot_df['Close'], label='실제 주가', color='blue')

        # 예측 주가
        ax.plot(future_dates, future_preds, label='미래 예측 주가', color='red', linestyle='--')

        ax.axvline(last_date, color='gray', linestyle=':', label='예측 기준일')
        ax.set_title(f"{selected_name} ({selected_code}) 주가 예측")
        ax.set_xlabel("날짜")
        ax.set_ylabel("가격 (원)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout() # 그래프 레이아웃 자동 조정
        st.pyplot(fig)

        st.subheader("📈 예측 기간 수익률")
        returns = (future_preds[-1] - future_preds[0]) / future_preds[0] * 100
        st.metric(label=f"예측 기간 수익률 ({future_dates[0].strftime('%Y-%m-%d')} ~ {future_dates[-1].strftime('%Y-%m-%d')})",
                  value=f"{returns:.2f}%")

else:
    # 데이터 로드 실패 시 메시지 (load_merged_data 함수에서 이미 에러 메시지 출력됨)
    st.info("데이터 로드 중 문제가 발생했습니다. 페이지 상단의 오류 메시지를 확인해주세요.")


st.markdown("---")
st.write("### 참고")
st.write("""
- **LSTM (Long Short-Term Memory):** 시계열 데이터와 같이 순서가 중요한 데이터를 처리하는 데 강점을 가진 딥러닝 모델의 한 종류입니다.
- **예측의 한계:** 주가 예측은 본질적으로 불확실성이 매우 높습니다. 이 모델은 과거 데이터를 기반으로 학습하므로, 급격한 시장 변화나 예상치 못한 외부 요인을 반영하기 어렵습니다. 참고 자료로만 활용하시기 바랍니다.
- **모델 재학습:** 새로운 데이터가 추가되거나 시간이 지남에 따라 모델을 재학습하는 것이 예측 정확도를 높이는 데 도움이 됩니다.
""")
