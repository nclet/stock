import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

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
st.set_page_config(layout="wide")

st.title("🔮 미래 주가 예측 (LSTM 기반)")
st.markdown("과거 주가 데이터와 기술적/펀더멘털 지표를 활용하여 미래 주가를 예측합니다.")

# ---
## 기술적 지표 계산 함수 (기존과 동일)
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
    rs = avg_gain / avg_loss.replace(0, np.nan)
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

# @st.cache_resource # 이 줄을 주석 처리하거나 삭제합니다.
def train_and_predict_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_future_days, last_sequence, scaler, features): # features 인자 추가
    model_path = f"model_{selected_code}.h5"
    model = None

    # Streamlit Cloud에서는 앱이 재시작될 때마다 파일 시스템이 초기화됩니다.
    # 따라서 이 로직은 매번 새로 학습하는 효과를 가져옵니다.
    if os.path.exists(model_path):
        st.info("⚠️ 로컬에 모델 파일이 있지만, Streamlit Cloud에서는 매번 재학습될 수 있습니다.")
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
    def recursive_forecast(model, last_sequence, n_days, scaler, n_features, features_list): # features_list 인자 추가
        forecasts = []
        current_seq = last_sequence.copy()

        for _ in range(n_days):
            pred = model.predict(current_seq.reshape(1, seq_len, n_features), verbose=0)[0][0]
            forecasts.append(pred)

            # 다음 예측을 위해 시퀀스 업데이트:
            # 예측된 종가(pred)를 새로운 날의 모든 특징 값으로 사용하는 단순한 방식.
            # 더 정확한 예측을 위해서는 각 특징의 미래 값을 예측하거나, 외부 데이터를 가져와야 합니다.
            new_feature_vector = np.full(n_features, pred)
            current_seq = np.vstack([current_seq[1:], new_feature_vector])

        # 예측값을 역 스케일링하여 실제 주가 범위로 되돌립니다.
        dummy_array_for_inverse = np.zeros((len(forecasts), n_features))
        dummy_array_for_inverse[:, features_list.index('Close')] = forecasts # 전달받은 features_list 사용
        forecasts_scaled = scaler.inverse_transform(dummy_array_for_inverse)[:, features_list.index('Close')]
        return forecasts_scaled

    future_preds = recursive_forecast(model, last_sequence, n_future_days, scaler, n_features, features) # features 전달
    return future_preds

# ---
## Streamlit UI
# 데이터 로드 (기존과 동일)
@st.cache_data
def load_merged_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(current_dir, '..')
        merged_data_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')

        df = pd.read_csv(merged_data_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        st.success(f"✅ 'merged_data_monthly_per_pbr.csv' 파일을 성공적으로 불러왔습니다. (경로: {merged_data_file_path})")
        return df
    except FileNotFoundError:
        st.error(f"❌ 'merged_data_monthly_per_pbr.csv' 파일을 찾을 수 없습니다. 예상 경로: {merged_data_file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

df_all_data = load_merged_data()

# 데이터 로드 성공 여부 확인
if not df_all_data.empty:
    st.write(f"Loaded DataFrame columns: {df_all_data.columns.tolist()}") # 이 줄을 추가
    if 'PER' not in df_all_data.columns or 'PBR' not in df_all_data.columns:
        st.warning("경고: 데이터 파일에 'PER' 또는 'PBR' 컬럼이 없어 예측에 사용되지 않습니다. 해당 컬럼이 없으면 정확도가 떨어질 수 있습니다.")
    else:
        st.info("데이터 파일에서 'PER' 및 'PBR' 컬럼을 성공적으로 확인했습니다.")

    name_code_dict = df_all_data.drop_duplicates(subset=['Code']).set_index('Name')['Code'].to_dict()

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

        if 'PER' not in df_stock.columns:
            df_stock['PER'] = 0.0
        if 'PBR' not in df_stock.columns:
            df_stock['PBR'] = 0.0

        # 예측에 사용할 특징 (feature) 컬럼 정의
        features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR'] # 이 부분은 `train_and_predict_model` 함수로 전달될 것입니다.
        target = 'Close' # 예측 대상은 'Close'

        # 예측에 필요한 데이터만 선택하고 NaN 값 제거
        df_processed = df_stock[features + [target]].dropna()

        seq_len = 20 # LSTM 시퀀스 길이 (과거 20일 데이터로 다음 날 예측)

        if len(df_processed) < seq_len + 1: # 최소 시퀀스 길이 + 1 (타겟값)
            st.warning(f"데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 최소 {seq_len + 1}일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_processed)}일)")
            st.stop()

        # 스케일링
        scaler = MinMaxScaler()
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
        # train_and_predict_model 함수에 `features` 리스트를 명시적으로 전달합니다.
        future_preds = train_and_predict_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_days, last_sequence, scaler, features)

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
        plot_df = df_processed.tail(365) # 최근 1년(365일)치만 표시
        ax.plot(plot_df.index, plot_df['Close'], label='실제 주가', color='blue')

        # 예측 주가
        ax.plot(future_dates, future_preds, label='미래 예측 주가', color='red', linestyle='--')

        ax.axvline(last_date, color='gray', linestyle=':', label='예측 기준일')
        ax.set_title(f"{selected_name} ({selected_code}) 주가 예측")
        ax.set_xlabel("날짜")
        ax.set_ylabel("가격 (원)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
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
- **모델 재학습:** Streamlit Cloud와 같은 배포 환경에서는 앱이 재시작될 때마다 파일 시스템이 초기화됩니다. 이로 인해 **매번 모델을 처음부터 다시 학습**하게 되며, 이는 **시간이 오래 걸릴 수 있습니다.** 만약 모델 학습 시간을 줄이고 싶다면, 학습된 모델을 Google Drive나 S3와 같은 **외부 스토리지에 저장하고 불러오는 방식**을 고려해야 합니다.
""")
