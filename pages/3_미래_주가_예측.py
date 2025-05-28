# streamlit_test/pages/3_미래_주가_예측.py
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
    st.stop()


st.set_page_config(layout="wide")
st.title("🔮 미래 주가 예측 (LSTM 기반)")
st.markdown("과거 주가 데이터와 기술적/펀더멘털 지표를 활용하여 미래 주가를 예측합니다.")

# --------------------------------------------
# 함수 정의 (이전 코드에서 가져와서 여기에 배치)
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

def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

@st.cache_resource # 모델 학습 결과를 캐싱 (재실행 시 재학습 방지)
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
            pred = model.predict(current_seq.reshape(1, seq_len, n_features), verbose=0)[0][0]
            forecasts.append(pred)

            # 다음 예측을 위해 시퀀스 업데이트 (첫 번째 특징(Close)만 사용)
            # 여기서는 마지막 예측값(pred)을 모든 특징 위치에 넣어주는 방식으로 단순화
            new_feature_vector = np.full(n_features, pred) 
            current_seq = np.vstack([current_seq[1:], new_feature_vector]) # 마지막에 새 벡터 추가

        # 스케일링 되돌리기
        forecasts_scaled = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))[:, 0]
        return forecasts_scaled

    future_preds = recursive_forecast(model, last_sequence, n_future_days, scaler, n_features)
    return future_preds

# --------------------------------------------
# Streamlit UI
# --------------------------------------------

# 데이터 로드
@st.cache_data
def load_merged_data():
    try:
        df = pd.read_csv('merged_data_monthly_per_pbr.csv') # 이 파일은 streamlit_test 폴더에 있어야 함
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str).str.zfill(6) # 종목코드 6자리로 채우기
        return df
    except FileNotFoundError:
        st.error("❌ 'merged_data_monthly_per_pbr.csv' 파일을 찾을 수 없습니다. 프로젝트 루트 디렉토리에 넣어주세요.")
        return pd.DataFrame()

df_all_data = load_merged_data()

if not df_all_data.empty:
    name_code_dict = df_all_data.drop_duplicates(subset=['Code']).set_index('Name')['Code'].to_dict()
    selected_name = st.selectbox("🔮 예측할 종목을 선택하세요", sorted(name_code_dict.keys()))
    selected_code = name_code_dict[selected_name]

    n_days = st.slider("예측할 미래 일 수", 5, 60, 30)
    
    # PER/PBR 데이터가 없는 경우를 대비하여 컬럼 확인
    if 'PER' not in df_all_data.columns or 'PBR' not in df_all_data.columns:
        st.warning("경고: 데이터 파일에 'PER' 또는 'PBR' 컬럼이 없어 예측에 사용되지 않습니다. 해당 컬럼이 없으면 정확도가 떨어질 수 있습니다.")
        
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
        
        # PER/PBR 컬럼이 없으면 0으로 채우거나 NaN 처리
        if 'PER' not in df_stock.columns:
            df_stock['PER'] = 0.0
        if 'PBR' not in df_stock.columns:
            df_stock['PBR'] = 0.0

        # 예측에 사용할 특징 (feature) 컬럼 정의
        features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR']
        target = 'Close' # 예측 대상은 'Close'

        # 예측에 필요한 데이터만 선택하고 NaN 값 제거
        df_processed = df_stock[features + [target]].dropna()
        
        seq_len = 20 # LSTM 시퀀스 길이
        
        if len(df_processed) < seq_len + 1: # 최소 시퀀스 길이 + 1 (타겟)
            st.warning(f"데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 최소 {seq_len + 1}일 이상의 데이터가 필요합니다. (현재 {len(df_processed)}일)")
            st.stop()
            
        scaler = MinMaxScaler()
        # 모든 특징을 동시에 스케일링
        scaled_data = scaler.fit_transform(df_processed[features])
        
        X, y = [], []
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i:i+seq_len]) # 시퀀스
            y.append(scaled_data[i+seq_len, features.index(target)]) # 예측 대상 (종가)

        if not X:
            st.warning(f"데이터 전처리 후 남은 데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 시퀀스 길이를 조절하거나 더 많은 데이터를 확보해주세요.")
            st.stop()

        X, y = np.array(X), np.array(y)
        
        # 학습/테스트 데이터 분리 (시계열 데이터이므로 shuffle=False)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        last_sequence = X[-1] # 마지막 시퀀스
        n_features = X.shape[2] # 특징의 개수

        # 모델 학습 및 예측 함수 호출
        future_preds = train_and_predict_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_days, last_sequence, scaler)

        if future_preds is None: # 모델 학습/예측 실패 시
            st.error("미래 주가 예측에 실패했습니다. 데이터를 확인하거나 다시 시도해주세요.")
            st.stop()

        last_date = df_processed.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]

        st.subheader("📊 실제 주가 및 미래 예측 주가")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 실제 주가 (최근 일부만 시각화하여 예측과 비교 용이)
        plot_df = df_processed.tail(365) # 최근 1년치 데이터
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
    st.info("데이터 로드 중 문제가 발생했습니다. 'merged_data_monthly_per_pbr.csv' 파일이 올바른 위치에 있는지 확인해주세요.")

st.markdown("---")
st.write("### 참고")
st.write("""
- **AI 모델:** LSTM(Long Short-Term Memory) 신경망을 사용하여 시계열 데이터를 학습하고 예측합니다.
- **모델 저장/로드:** 학습된 모델은 `model_종목코드.h5` 파일로 저장되어, 같은 종목 재분석 시 학습 시간을 절약합니다.
- **예측 한계:** AI 예측은 과거 데이터 패턴에 기반하며, 시장의 예측 불가능한 변화나 이벤트는 반영하기 어렵습니다. 따라서 예측은 참고 자료로만 활용해야 합니다.
""")
