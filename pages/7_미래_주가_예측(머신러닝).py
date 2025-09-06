import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import io

# 딥러닝 및 머신러닝 관련 라이브러리 임포트
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    st.error("""
    **필요 라이브러리가 설치되지 않았습니다. 다음 명령어를 실행하여 설치해주세요:**
    `pip install tensorflow scikit-learn matplotlib`
    """)
    st.stop()

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide")

st.title("🔮 통합 주가 예측 모델 (LSTM & RandomForest)")
st.markdown("하나의 애플리케이션에서 **미래 주가 예측(LSTM)**과 **단기 수익률 예측(RandomForest)**을 모두 확인하세요.")

# --- 공통 기술적 지표 계산 함수 ---
@st.cache_data
def calculate_bollinger_bands(prices, window=20, num_std=2):
    """볼린저 밴드 (Bollinger Bands)를 계산합니다."""
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

@st.cache_data
def calculate_rsi(series, period=14):
    """상대강도지수 (RSI)를 계산합니다."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- 데이터 로드 함수 (공통) ---
@st.cache_data
def load_merged_data():
    """CSV 파일에서 주가 데이터를 로드합니다."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(current_dir, '..')
        merged_data_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')

        # 파일이 존재하는지 확인
        if not os.path.exists(merged_data_file_path):
            st.error(f"❌ 데이터 파일을 찾을 수 없습니다: '{merged_data_file_path}'")
            st.info("죄송합니다. 데이터 파일이 소실되어 있는 상태입니다.")
            return pd.DataFrame()

        df = pd.read_csv(merged_data_file_path)

        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        
        st.success("✅ 데이터를 성공적으로 로드했습니다.")
        return df
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# --- 딥러닝 (LSTM) 관련 함수 ---
def build_lstm_model(input_shape):
    """LSTM 모델 구조를 생성합니다."""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

@st.cache_resource
def train_and_predict_lstm_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_future_days, last_sequence, scaler, features):
    """LSTM 모델을 학습하고 미래 주가를 예측합니다."""
    model_path = f"model_{selected_code}.h5"
    model = None

    if os.path.exists(model_path):
        st.info("✅ 기존 학습 모델을 로드합니다. (주가 예측 모형은 종가를 중심으로 RSI·볼린저밴드·PER·PBR,LSTM를 통해 분석하고 있습니다.)")
        model = load_model(model_path)
    else:
        st.info("🔄 새로운 LSTM 모델 학습이 필요합니다. 잠시만 기다려 주세요...")
        model = build_lstm_model(input_shape=(seq_len, n_features))
        with st.spinner("⏳ 모델 학습 중 (시간이 다소 소요될 수 있습니다)..."):
            model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
        model.save(model_path)
        st.success("✅ LSTM 모델 학습 완료 및 저장!")

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

# --- 머신러닝 (RandomForest) 관련 함수 ---
@st.cache_resource
def train_and_predict_random_forest(selected_code, df_stock_data, ml_features):
    """RandomForest 모델을 학습하고 다음 날 수익률을 예측합니다."""
    df_stock_data['Next_Day_Return'] = df_stock_data['Close'].pct_change().shift(-1) * 100
    df_ml = df_stock_data[ml_features + ['Next_Day_Return']].dropna()

    if len(df_ml) < 20:
        st.warning(f"데이터가 부족하여 수익률 예측을 할 수 없습니다. 최소 20일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_ml)}일)")
        return None, None, None, None, None

    X_ml = df_ml[ml_features].values
    y_ml = df_ml['Next_Day_Return'].values
    scaler_ml = MinMaxScaler()
    X_ml_scaled = scaler_ml.fit_transform(X_ml)
    test_size_ml = max(1, int(0.2 * len(X_ml_scaled)))
    X_train_ml, X_test_ml = X_ml_scaled[:-test_size_ml], X_ml_scaled[-test_size_ml:]
    y_train_ml, y_test_ml = y_ml[:-test_size_ml], y_ml[-test_size_ml:]

    if len(X_test_ml) == 0:
        st.warning("테스트 데이터가 부족하여 모델 평가를 수행할 수 없습니다.")
        return None, None, None, None, None

    with st.spinner(f"🔄 {selected_code} RandomForest 모델 학습 중..."):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_ml, y_train_ml)
    st.success("✅ RandomForest 모델 학습 완료!")

    y_pred_ml = rf_model.predict(X_test_ml)
    last_data_ml_raw = df_ml[ml_features].iloc[-1].values.reshape(1, -1)
    last_data_ml_scaled = scaler_ml.transform(last_data_ml_raw)
    next_day_return_pred_ml = rf_model.predict(last_data_ml_scaled)[0]

    return rf_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml

# --- Streamlit UI 시작 ---
df_all_data = load_merged_data()

if not df_all_data.empty:
    try:
        name_code_dict = df_all_data.drop_duplicates(subset=['Code']).set_index('Name')['Code'].to_dict()
    except KeyError:
        st.error("데이터 파일에 'Name' 또는 'Code' 컬럼이 없어 종목 리스트를 생성할 수 없습니다.")
        st.stop()
    
    selected_name = st.selectbox("🔮 **예측할 종목을 선택하세요**", sorted(name_code_dict.keys()))
    selected_code = name_code_dict[selected_name]

    n_days = st.slider("LSTM 예측 기간 (미래 일 수)", 5, 60, 30)
    
    if st.button("🚀 **통합 예측 시작**"):
        df_stock = df_all_data[df_all_data['Code'] == selected_code].copy()
        df_stock.sort_values('Date', inplace=True)
        df_stock.set_index('Date', inplace=True)

        if df_stock.empty:
            st.error(f"선택하신 종목 ({selected_name})에 대한 데이터가 없습니다. 다른 종목을 선택해주세요.")
            st.stop()
        
        # --- 기술적 지표 계산 ---
        df_stock['RSI'] = calculate_rsi(df_stock['Close'])
        df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands(df_stock['Close'])
        
        # --- LSTM 모델 예측 섹션 ---
        st.header("1️⃣ LSTM 모델: 미래 주가 예측")
        
        features_lstm = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR']
        target_lstm = 'Close'
        
        df_processed_lstm = df_stock[features_lstm].dropna()
        seq_len_lstm = 20

        if len(df_processed_lstm) < seq_len_lstm + 1:
            st.warning(f"LSTM 예측을 위한 데이터가 부족합니다. 최소 {seq_len_lstm + 1}일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_processed_lstm)}일)")
        else:
            scaler_lstm = MinMaxScaler()
            scaled_data_lstm = scaler_lstm.fit_transform(df_processed_lstm)
            X_lstm, y_lstm = [], []
            for i in range(len(scaled_data_lstm) - seq_len_lstm):
                X_lstm.append(scaled_data_lstm[i:i+seq_len_lstm])
                y_lstm.append(scaled_data_lstm[i+seq_len_lstm, features_lstm.index(target_lstm)])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
            last_sequence_lstm = X_lstm[-1]
            n_features_lstm = X_lstm.shape[2]
            
            future_preds_lstm = train_and_predict_lstm_model(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, seq_len_lstm, n_features_lstm, selected_code, n_days, last_sequence_lstm, scaler_lstm, features_lstm)

            last_date = df_processed_lstm.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]

            st.subheader("📊 LSTM 예측 주가 시각화")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_df = df_processed_lstm.tail(365)
            ax.plot(plot_df.index, plot_df['Close'], label='실제 주가', color='blue')
            ax.plot(future_dates, future_preds_lstm, label='미래 예측 주가', color='red', linestyle='--')
            ax.axvline(last_date, color='gray', linestyle=':', label='예측 기준일')
            ax.set_title(f"{selected_name} ({selected_code}) 미래 주가 예측")
            ax.set_xlabel("날짜")
            ax.set_ylabel("가격(₩/원)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("📈 LSTM 예측 기간 수익률")
            returns_lstm = (future_preds_lstm[-1] - future_preds_lstm[0]) / future_preds_lstm[0] * 100
            st.metric(label=f"예측 기간 수익률 ({future_dates[0].strftime('%Y-%m-%d')} ~ {future_dates[-1].strftime('%Y-%m-%d')})",
                      value=f"{returns_lstm:.2f}%")

        # --- RandomForest 모델 예측 섹션 ---
        st.markdown("---")
        st.header("2️⃣ RandomForest 모델: 단기 수익률 예측")
        
        ml_features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower']
        rf_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml = train_and_predict_random_forest(selected_code, df_stock.copy(), ml_features)

        if rf_model is not None:
            st.subheader("📊 RandomForest 모델 성능 평가")
            st.write(f"**평균 제곱 오차 (MSE)**: {mean_squared_error(y_test_ml, y_pred_ml):.2f}")
            st.write(f"**결정 계수 (R² Score)**: {r2_score(y_test_ml, y_pred_ml):.2f}")
            st.write(f"테스트 데이터의 **평균 실제 수익률**: {np.mean(y_test_ml):.2f}%")
            st.write(f"테스트 데이터의 **평균 예측 수익률**: {np.mean(y_pred_ml):.2f}%")

            st.subheader("📈 RandomForest 다음 날 수익률 예측")
            st.metric(label="예측된 수익률", value=f"{next_day_return_pred_ml:.2f}%")
            
            # 예측 시각화 (실제 수익률과 예측 수익률 비교)
            st.markdown("---")
            st.subheader("📉 RandomForest 예측 vs. 실제 수익률")
            fig_rf, ax_rf = plt.subplots(figsize=(12, 6))
            ax_rf.plot(y_test_ml, label='실제 수익률', color='blue', marker='o', linestyle='None', alpha=0.6)
            ax_rf.plot(y_pred_ml, label='예측 수익률', color='red', marker='x', linestyle='None', alpha=0.6)
            ax_rf.set_title(f"{selected_name} ({selected_code}) RandomForest 예측 수익률")
            ax_rf.set_xlabel("데이터 포인트 인덱스")
            ax_rf.set_ylabel("수익률(%)")
            ax_rf.legend()
            ax_rf.grid(True)
            plt.tight_layout()
            st.pyplot(fig_rf)

else:
    st.info("데이터 로드 중 문제가 발생했습니다. 페이지 상단의 오류 메시지를 확인해주세요.")


########################################################################
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import os

# # 머신러닝 라이브러리 임포트
# try:
#     from sklearn.preprocessing import MinMaxScaler
#     from sklearn.ensemble import RandomForestRegressor
#     from sklearn.metrics import mean_squared_error, r2_score
#     from sklearn.model_selection import train_test_split
# except ImportError:
#     st.error("""
#     죄송합니다. 라이브러리 오류입니다. 나중에 다시 시도해주세요.
#     """)
#     st.stop()

# # --- Streamlit 페이지 설정 ---
# st.set_page_config(layout="wide")

# st.title("🚀 주가 수익률 머신러닝 예측")
# st.markdown("데이터를 통해 랜덤포레스트(RandomForest)모델을 사용하여 **단기 주가 수익률**을 예측합니다.")

# # --- 기술적 지표 계산 함수 ---
# @st.cache_data
# def calculate_bollinger_bands_pred(prices, window=20, num_std=2):
#     """볼린저 밴드 (Bollinger Bands)를 계산합니다."""
#     rolling_mean = prices.rolling(window).mean()
#     rolling_std = prices.rolling(window).std()
#     upper_band = rolling_mean + (rolling_std * num_std)
#     lower_band = rolling_mean - (rolling_std * num_std)
#     return rolling_mean, upper_band, lower_band

# @st.cache_data
# def calculate_rsi_pred(series, period=14):
#     """상대강도지수 (RSI)를 계산합니다."""
#     delta = series.diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#     avg_gain = gain.rolling(window=period).mean()
#     avg_loss = loss.rolling(window=period).mean()
#     rs = avg_gain / avg_loss.replace(0, np.nan).fillna(0)
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

# # --- 데이터 로드 함수 (기존 CSV 파일 사용) ---
# @st.cache_data
# def load_merged_data():
#     """CSV 파일에서 주가 데이터를 로드합니다."""
#     try:
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         # Streamlit 앱 파일과 같은 디렉토리 또는 상위 디렉토리에서 CSV 파일 탐색
#         merged_data_file_path = os.path.join(current_dir, 'merged_data_monthly_per_pbr.csv')
#         if not os.path.exists(merged_data_file_path):
#             # 만약 현재 디렉토리에 없으면 상위 디렉토리를 시도
#             root_dir = os.path.join(current_dir, '..')
#             merged_data_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')

#         if not os.path.exists(merged_data_file_path):
#             st.error(f"❌ 데이터 파일을 찾을 수 없습니다: '{merged_data_file_path}'")
#             st.info("죄송합니다. 데이터 파일이 소실되어 있는 상태입니다.")
#             return pd.DataFrame()

#         df = pd.read_csv(merged_data_file_path)

#         df.columns = df.columns.str.strip() # 컬럼명 공백 제거
#         df['Date'] = pd.to_datetime(df['Date'])
#         df['Code'] = df['Code'].astype(str).str.zfill(6) # 종목코드 6자리로 통일
        
#         st.success(f"✅데이터를 성공적으로 로드했습니다. (총 {len(df)}개 데이터 포인트)")
#         return df
#     except Exception as e:
#         st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
#         return pd.DataFrame()

# # --- RandomForest 모델 학습 및 예측 함수 캐싱 ---
# @st.cache_resource
# def train_and_predict_random_forest(selected_code, df_stock_data, ml_features):
    
#     # 다음 날 수익률 계산 (RandomForest의 예측 목표)
#     df_stock_data['Next_Day_Return'] = df_stock_data['Close'].pct_change().shift(-1) * 100
    
#     # 결측치 제거
#     df_ml = df_stock_data[ml_features + ['Next_Day_Return']].dropna()

#     if len(df_ml) < 20: 
#         st.warning(f"[RandomForest] 데이터가 부족하여 수익률 예측을 할 수 없습니다. 최소 20일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_ml)}일)")
#         return None, None, None, None, None # 모델, 예측값, 다음날 예측 수익률, 실제 수익률, 테스트 데이터 반환 (없으면 None)
    
#     X_ml = df_ml[ml_features].values
#     y_ml = df_ml['Next_Day_Return'].values

#     # 데이터 스케일링 (특성 스케일링)
#     scaler_ml = MinMaxScaler()
#     X_ml_scaled = scaler_ml.fit_transform(X_ml)

#     # 학습/테스트 데이터셋 분리 (마지막 20%를 테스트 데이터로 사용)
#     test_size_ml = max(1, int(0.2 * len(X_ml_scaled))) 
#     X_train_ml, X_test_ml = X_ml_scaled[:-test_size_ml], X_ml_scaled[-test_size_ml:]
#     y_train_ml, y_test_ml = y_ml[:-test_size_ml], y_ml[-test_size_ml:]
    
#     # 테스트 데이터셋이 너무 작을 경우 처리
#     if len(X_test_ml) == 0:
#         st.warning(f"[RandomForest] 테스트 데이터가 부족하여 모델 평가를 수행할 수 없습니다. 대신 학습 데이터의 마지막 샘플로 평가를 진행합니다.")
#         # 학습 데이터에서 최소한의 샘플이라도 사용하여 평가 시도
#         if len(X_train_ml) > 0:
#             X_test_ml = X_train_ml[-1:] 
#             y_test_ml = y_train_ml[-1:] 
#         else: # 학습 데이터도 없는 극단적인 경우
#             st.error("모델 학습을 위한 데이터가 충분하지 않습니다. 파일의 데이터를 확인해주세요.")
#             return None, None, None, None, None

#     st.info("랜덤포레스트 모델 학습 중...")
#     # RandomForestRegressor 모델 초기화 및 학습
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1로 모든 코어 사용
#     with st.spinner(f"🔄 {selected_code} 랜덤포레스트 모델 학습 중..."):
#         rf_model.fit(X_train_ml, y_train_ml)
#     st.success("✅ RandomForest 모델 학습 완료!")

#     # 모델 성능 평가
#     y_pred_ml = rf_model.predict(X_test_ml)
    
#     # 다음 거래일 수익률 예측
#     last_data_ml_raw = df_ml[ml_features].iloc[-1].values.reshape(1, -1)
#     last_data_ml_scaled = scaler_ml.transform(last_data_ml_raw)
#     next_day_return_pred_ml = rf_model.predict(last_data_ml_scaled)[0]

#     return rf_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml 

# # --- Streamlit UI 시작 ---
# # 전체 데이터 로드 (첫 로드 시 시간이 걸릴 수 있음, @st.cache_data 덕분에 두번째부터는 빠름)
# df_all_data = load_merged_data()

# if df_all_data.empty:
#     st.info("데이터 로드 중 문제가 발생하여 앱이 실행되지 않습니다. 파일 경로/내용을 확인해주세요.")
#     st.stop()

# # 종목 리스트 생성
# try:
#     # 'Name'과 'Code' 컬럼이 있어야 함
#     name_code_dict = df_all_data.drop_duplicates(subset=['Code']).set_index('Name')['Code'].to_dict()
# except KeyError:
#     st.error("데이터 파일에 'Name' 또는 'Code' 컬럼이 없어 종목 리스트를 생성할 수 없습니다. 파일 형식을 확인해주세요.")
#     st.stop()

# if not name_code_dict:
#     st.error("종목 리스트를 생성할 수 없습니다. 데이터 파일에 유효한 종목 정보가 있는지 확인해주세요.")
#     st.stop()

# selected_name = st.selectbox("🔮 **예측할 종목을 선택하세요**", sorted(name_code_dict.keys()))
# selected_code = name_code_dict[selected_name]

# st.markdown("---")
# st.subheader("🤖 **랜덤포레스트 주가 예측**")
# st.info("RandomForest 모델은 과거 주가와 기술적 지표를 기반으로 단기 수익률을 예측합니다.")

# if st.button("🚀 **수익률 예측 시작하기**"):
#     with st.spinner(f"'{selected_name}' 데이터 준비 및 RandomForest 모델 예측 중..."):
#         # 선택된 종목의 데이터만 필터링
#         df_stock = df_all_data[df_all_data['Code'] == selected_code].copy()
#         df_stock.sort_values('Date', inplace=True)
#         df_stock.set_index('Date', inplace=True)

#         if df_stock.empty:
#             st.error(f"선택하신 종목 ({selected_name})에 대한 데이터가 없습니다. 다른 종목을 선택해주세요.")
#             st.stop()
        
#         # 기술적 지표 계산 (종가 기반)
#         df_stock['RSI'] = calculate_rsi_pred(df_stock['Close'])
#         df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands_pred(df_stock['Close'])

#         # RandomForest 모델에 사용할 Features 정의 (PER/PBR 제외)
#         # 이제 'PER', 'PBR' 컬럼이 없어도 코드가 정상 작동합니다.
#         ml_features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower']
        
#         # RandomForest 모델 학습 및 예측 (캐시된 함수 사용)
#         rf_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml = \
#             train_and_predict_random_forest(selected_code, df_stock.copy(), ml_features)
        
#         if rf_model is None: # 모델 학습/예측 실패 시
#             st.stop()

#         st.subheader("📊 **랜덤포레스트 모델 성능 평가**")
#         st.write(f"**평균 제곱 오차 (MSE)**: {mean_squared_error(y_test_ml, y_pred_ml):.2f}")
#         st.write(f"**결정 계수 (R² Score)**: {r2_score(y_test_ml, y_pred_ml):.2f}")
#         st.write(f"테스트 데이터의 **평균 실제 수익률**: {np.mean(y_test_ml):.2f}%")
#         st.write(f"테스트 데이터의 **평균 예측 수익률**: {np.mean(y_pred_ml):.2f}%")

#         st.subheader("📈 **RandomForest 단기 수익률 예측**")
#         st.metric(label="예측된 수익률", value=f"{next_day_return_pred_ml:.2f}%")

#         if next_day_return_pred_ml > 0.5:
#             st.success("✨ RandomForest 모델이 **강력한 상승**을 예측합니다!")
#         elif next_day_return_pred_ml > 0:
#             st.info("⬆️ RandomForest 모델이 **소폭 상승**을 예측합니다.")
#         elif next_day_return_pred_ml < -0.5:
#             st.error("🚨 RandomForest 모델이 **강력한 하락**을 예측합니다!")
#         elif next_day_return_pred_ml < 0:
#             st.warning("⬇️ RandomForest 모델이 **소폭 하락**을 예측합니다.")
#         else:
#             st.write("➖ RandomForest 모델이 **큰 변동 없음**을 예측합니다.")

#         # 예측 시각화 (실제 수익률과 예측 수익률 비교)
#         st.markdown("---")
#         st.subheader("📉 **RandomForest 모델 예측 vs. 실제 수익률**")
        
#         fig_rf, ax_rf = plt.subplots(figsize=(12, 6))
#         ax_rf.plot(y_test_ml, label='actual rate of return', color='blue', marker='o', linestyle='None', alpha=0.6)
#         ax_rf.plot(y_pred_ml, label='forecasted rate of return', color='red', marker='x', linestyle='None', alpha=0.6)
#         ax_rf.set_title(f"{selected_name} ({selected_code}) RandomForest forecasted rate of return")
#         ax_rf.set_xlabel("Data Point Index")
#         ax_rf.set_ylabel("the rate of return(%)")
#         ax_rf.legend()
#         ax_rf.grid(True)
#         plt.tight_layout()
#         st.pyplot(fig_rf)
