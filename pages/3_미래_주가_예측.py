import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# 딥러닝 관련 라이브러리 임포트 (기존과 동일)
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
    st.stop()


# Streamlit 페이지 설정 (기존과 동일)
st.set_page_config(layout="wide")

st.title("🔮 미래 주가 예측 (LSTM 기반)")
st.markdown("과거 주가 데이터와 기술적/펀더멘털 지표를 활용하여 미래 주가를 예측합니다.")

# 기술적 지표 계산 함수 (기존과 동일)
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

# LSTM 모델 관련 함수 (기존과 동일)
def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_predict_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_future_days, last_sequence, scaler, features):
    model_path = f"model_{selected_code}.h5"
    model = None

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

    def recursive_forecast(model, last_sequence, n_days, scaler, n_features, features_list):
        forecasts = []
        current_seq = last_sequence.copy()

        for _ in range(n_days):
            pred = model.predict(current_seq.reshape(1, seq_len, n_features), verbose=0)[0][0]
            forecasts.append(pred)
            new_feature_vector = np.full(n_features, pred)
            current_seq = np.vstack([current_seq[1:], new_feature_vector])

        dummy_array_for_inverse = np.zeros((len(forecasts), n_features))
        dummy_array_for_inverse[:, features_list.index('Close')] = forecasts
        forecasts_scaled = scaler.inverse_transform(dummy_array_for_inverse)[:, features_list.index('Close')]
        return forecasts_scaled

    future_preds = recursive_forecast(model, last_sequence, n_future_days, scaler, n_features, features)
    return future_preds

# ---
## Streamlit UI
# 데이터 로드
@st.cache_data
def load_merged_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(current_dir, '..')
        merged_data_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')

        df = pd.read_csv(merged_data_file_path)

        # ---------------------- 이 부분이 컬럼 공백을 제거하는 코드입니다 ----------------------
        df.columns = df.columns.str.strip()
        # -----------------------------------------------------------------------------------

        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        st.success(f"✅데이터를 성공적으로 불러왔습니다.")

        return df
    except FileNotFoundError:
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다. 죄송합니다. 코드를 수정중입니다.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

df_all_data = load_merged_data()

# 데이터 로드 성공 여부 확인 (기존과 동일)
if not df_all_data.empty:
    # 이 조건문에서 이제 'PER'과 'PBR'을 제대로 찾을 것입니다.
    # if 'PER' not in df_all_data.columns or 'PBR' not in df_all_data.columns:
    #     st.warning("경고: 데이터 파일에 'PER' 또는 'PBR' 컬럼이 없어 예측에 사용되지 않습니다. 해당 컬럼이 없으면 정확도가 떨어질 수 있습니다.")
    # else:
    #     st.info("데이터 파일에서 'PER' 및 'PBR' 컬럼을 성공적으로 확인했습니다.")

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
        df_stock.set_index('Date', inplace=True)

        if df_stock.empty:
            st.error(f"선택하신 종목 ({selected_name})에 대한 데이터가 없습니다. 다른 종목을 선택해주세요.")
            st.stop()

        df_stock['RSI'] = calculate_rsi_pred(df_stock['Close'])
        df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands_pred(df_stock['Close'])

        # 이 부분은 이제 거의 항상 건너뛰어질 것입니다 (컬럼 이름이 제대로 파싱될 것이므로)
        if 'PER' not in df_stock.columns:
            df_stock['PER'] = 0.0
        if 'PBR' not in df_stock.columns:
            df_stock['PBR'] = 0.0

        features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR']
        target = 'Close'

        df_processed = df_stock[features + [target]].dropna()

        seq_len = 20

        if len(df_processed) < seq_len + 1:
            st.warning(f"데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 최소 {seq_len + 1}일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_processed)}일)")
            st.stop()

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_processed[features])

        X, y = [], []
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i:i+seq_len])
            y.append(scaled_data[i+seq_len, features.index(target)])

        if not X:
            st.warning(f"데이터 전처리 후 남은 데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 시퀀스 길이를 조절하거나 더 많은 데이터를 확보해주세요.")
            st.stop()

        X, y = np.array(X), np.array(y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        last_sequence = X[-1]
        n_features = X.shape[2]

        future_preds = train_and_predict_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_days, last_sequence, scaler, features)

        if future_preds is None:
            st.error("미래 주가 예측에 실패했습니다. 데이터를 확인하거나 다시 시도해주세요.")
            st.stop()

        last_date = df_processed.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]

        st.subheader("📊 실제 주가 및 미래 예측 주가")
        fig, ax = plt.subplots(figsize=(12, 6))

        plot_df = df_processed.tail(365)
        ax.plot(plot_df.index, plot_df['Close'], label='실제 주가', color='blue')

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
    st.info("데이터 로드 중 문제가 발생했습니다. 페이지 상단의 오류 메시지를 확인해주세요.")


st.markdown("---")
st.write("### 참고")
st.write("""
- **LSTM (Long Short-Term Memory):** 시계열 데이터와 같이 순서가 중요한 데이터를 처리하는 데 강점을 가진 딥러닝 모델의 한 종류입니다.
- **예측의 한계:** 주가 예측은 본질적으로 불확실성이 매우 높습니다. 이 모델은 과거 데이터를 기반으로 학습하므로, 급격한 시장 변화나 예상치 못한 외부 요인을 반영하기 어렵습니다. 참고 자료로만 활용하시기 바랍니다.
- **모델 재학습:** Streamlit Cloud와 같은 배포 환경에서는 앱이 재시작될 때마다 파일 시스템이 초기화됩니다. 이로 인해 **매번 모델을 처음부터 다시 학습**하게 되며, 이는 **시간이 오래 걸릴 수 있습니다.** 만약 모델 학습 시간을 줄이고 싶다면, 학습된 모델을 Google Drive나 S3와 같은 **외부 스토리지에 저장하고 불러오는 방식**을 고려해야 합니다.
""")
