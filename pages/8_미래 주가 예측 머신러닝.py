import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# 금융 데이터 로더 라이브러리 임포트
try:
    import FinanceDataReader as fdr
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
except ImportError:
    st.error("""
    **필수 라이브러리가 설치되지 않았습니다!**
    아래 명령어를 실행하여 필요한 라이브러리를 설치해주세요:
    `pip install FinanceDataReader scikit-learn pandas matplotlib streamlit`
    """)
    st.stop()

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide")

st.title("🚀 주가 수익률 예측 대시보드 (RandomForest with FinanceDataReader)")
st.markdown("`FinanceDataReader`를 통해 실시간 데이터를 가져와 과거 주가 데이터와 기술적 지표를 활용하여 **다음 거래일의 수익률**을 예측합니다.")

# --- 기술적 지표 계산 함수 ---
@st.cache_data
def calculate_bollinger_bands_pred(prices, window=20, num_std=2):
    """볼린저 밴드 (Bollinger Bands)를 계산합니다."""
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

@st.cache_data
def calculate_rsi_pred(series, period=14):
    """상대강도지수 (RSI)를 계산합니다."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- FinanceDataReader를 이용한 종목 정보 및 주가 데이터 로드 함수 ---

@st.cache_data # 종목 코드 정보를 캐시하여 빠르게 로드
def get_krx_stock_list():
    """KRX 상장사 전체 종목 리스트를 가져옵니다."""
    try:
        df_krx = fdr.StockListing('KRX')
        # 'Code' 컬럼이 문자열이고 6자리로 채워져 있는지 확인 (선택 사항)
        df_krx['Code'] = df_krx['Code'].astype(str).str.zfill(6)
        # 종목명과 코드 매핑 딕셔너리 생성 (예: {'삼성전자': '005930', ...})
        name_code_dict = df_krx.set_index('Name')['Code'].to_dict()
        st.success("✅ KRX 종목 리스트를 성공적으로 로드했습니다.")
        return name_code_dict
    except Exception as e:
        st.error(f"❌ KRX 종목 리스트 로드 중 오류 발생: {e}")
        st.info("인터넷 연결을 확인하거나 'FinanceDataReader' 라이브러리 버전을 확인해보세요.")
        return {}

@st.cache_data # 개별 종목 주가 데이터를 캐시하여 빠르게 로드
def load_stock_data_from_fdr(stock_code, start_date=None, end_date=None):
    """FinanceDataReader를 사용하여 특정 종목의 주가 데이터를 로드합니다."""
    try:
        # 기본적으로 지난 5년간의 데이터를 가져오도록 설정 (RandomForest 훈련에 충분한 데이터)
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=5*365) # 대략 5년치 데이터

        df = fdr.DataReader(stock_code, start=start_date, end=end_date)
        
        if df.empty:
            st.warning(f"'{stock_code}'에 대한 데이터를 가져오지 못했습니다. 종목 코드를 확인해주세요.")
            return pd.DataFrame()
        
        # 컬럼명 통일 (FinanceDataReader는 'Close' 대신 'Close'를 사용)
        if 'Close' not in df.columns:
            st.error(f"'{stock_code}' 데이터에 'Close' 컬럼이 없습니다.")
            return pd.DataFrame()
        
        df.sort_index(inplace=True) # 날짜 순 정렬
        st.success(f"✅ '{stock_code}' 주가 데이터를 성공적으로 로드했습니다. (총 {len(df)}개 데이터 포인트)")
        return df
    except Exception as e:
        st.error(f"'{stock_code}' 주가 데이터 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

# --- Streamlit UI 시작 ---
# 모든 종목 코드 로드 (첫 로드 시 시간이 걸릴 수 있음)
name_code_dict = get_krx_stock_list()

if not name_code_dict:
    st.info("KRX 종목 리스트를 가져올 수 없습니다. 앱을 종료합니다.")
    st.stop()

# 종목 선택
selected_name = st.selectbox("🔮 **예측할 종목을 선택하세요**", sorted(name_code_dict.keys()))
selected_code = name_code_dict[selected_name]

if st.button("🚀 **예측 시작!**"):
    with st.spinner("데이터 로드 및 RandomForest 모델 예측 중..."):
        # FinanceDataReader를 통해 선택된 종목의 주가 데이터 로드
        df_stock = load_stock_data_from_fdr(selected_code)

        if df_stock.empty:
            st.stop()

        # 기술적 지표 계산
        df_stock['RSI'] = calculate_rsi_pred(df_stock['Close'])
        df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands_pred(df_stock['Close'])

        # RandomForest 모델에 사용할 Features와 Target 정의
        ml_features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower']
        
        # 다음 날 수익률 계산 (RandomForest의 예측 목표)
        df_stock['Next_Day_Return'] = df_stock['Close'].pct_change().shift(-1) * 100
        
        # 결측치 제거
        df_ml = df_stock[ml_features + ['Next_Day_Return']].dropna()

        if len(df_ml) < 20: 
            st.warning(f"[RandomForest] 데이터가 부족하여 수익률 예측을 할 수 없습니다. 최소 20일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_ml)}일)")
            st.stop()
        
        X_ml = df_ml[ml_features].values
        y_ml = df_ml['Next_Day_Return'].values

        # 데이터 스케일링 (특성 스케일링)
        scaler_ml = MinMaxScaler()
        X_ml_scaled = scaler_ml.fit_transform(X_ml)

        # 학습/테스트 데이터셋 분리 (마지막 20%를 테스트 데이터로 사용)
        test_size_ml = max(1, int(0.2 * len(X_ml_scaled))) 
        X_train_ml, X_test_ml = X_ml_scaled[:-test_size_ml], X_ml_scaled[-test_size_ml:]
        y_train_ml, y_test_ml = y_ml[:-test_size_ml], y_ml[-test_size_ml:]
        
        # 테스트 데이터셋이 너무 작을 경우 처리
        if len(X_test_ml) == 0:
            st.warning(f"[RandomForest] 테스트 데이터가 부족하여 모델 평가를 수행할 수 없습니다. 대신 학습 데이터의 마지막 샘플로 평가를 진행합니다.")
            X_test_ml = X_train_ml[-1:] # 학습 데이터의 마지막 샘플을 테스트로 사용
            y_test_ml = y_train_ml[-1:] 
        
        st.info("RandomForest 모델 학습 중...")
        # RandomForestRegressor 모델 초기화 및 학습
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1로 모든 코어 사용
        with st.spinner("🔄 RandomForest 모델 학습 중..."):
            rf_model.fit(X_train_ml, y_train_ml)
        st.success("✅ RandomForest 모델 학습 완료!")

        # 모델 성능 평가
        y_pred_ml = rf_model.predict(X_test_ml)
        
        st.subheader("📊 **RandomForest 모델 성능 평가 (테스트 데이터)**")
        st.write(f"**평균 제곱 오차 (MSE)**: {mean_squared_error(y_test_ml, y_pred_ml):.2f}")
        st.write(f"**결정 계수 (R² Score)**: {r2_score(y_test_ml, y_pred_ml):.2f}")
        st.write(f"테스트 데이터의 **평균 실제 수익률**: {np.mean(y_test_ml):.2f}%")
        st.write(f"테스트 데이터의 **평균 예측 수익률**: {np.mean(y_pred_ml):.2f}%")

        # 다음 거래일 수익률 예측
        last_data_ml = X_ml_scaled[-1].reshape(1, -1)
        next_day_return_pred_ml = rf_model.predict(last_data_ml)[0]

        st.subheader("📈 **RandomForest 다음 거래일 수익률 예측**")
        st.metric(label="예측된 다음 거래일 수익률", value=f"{next_day_return_pred_ml:.2f}%")

        if next_day_return_pred_ml > 0.5:
            st.success("✨ RandomForest 모델은 다음 거래일에 **강력한 상승**을 예측합니다!")
        elif next_day_return_pred_ml > 0:
            st.info("⬆️ RandomForest 모델은 다음 거래일에 **소폭 상승**을 예측합니다.")
        elif next_day_return_pred_ml < -0.5:
            st.error("🚨 RandomForest 모델은 다음 거래일에 **강력한 하락**을 예측합니다!")
        elif next_day_return_pred_ml < 0:
            st.warning("⬇️ RandomForest 모델은 다음 거래일에 **소폭 하락**을 예측합니다.")
        else:
            st.write("➖ RandomForest 모델은 다음 거래일에 **큰 변동 없음**을 예측합니다.")

        # 예측 시각화 (실제 수익률과 예측 수익률 비교)
        st.markdown("---")
        st.subheader("📉 **RandomForest 모델 예측 vs. 실제 수익률 (테스트 데이터)**")
        
        fig_rf, ax_rf = plt.subplots(figsize=(12, 6))
        ax_rf.plot(y_test_ml, label='실제 수익률', color='blue', marker='o', linestyle='None', alpha=0.6)
        ax_rf.plot(y_pred_ml, label='예측 수익률', color='red', marker='x', linestyle='None', alpha=0.6)
        ax_rf.set_title(f"{selected_name} ({selected_code}) RandomForest 예측 수익률")
        ax_rf.set_xlabel("데이터 포인트 인덱스 (테스트셋)")
        ax_rf.set_ylabel("수익률 (%)")
        ax_rf.legend()
        ax_rf.grid(True)
        plt.tight_layout()
        st.pyplot(fig_rf)
#########################################
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import os

# # 딥러닝 및 머신러닝 라이브러리 임포트
# try:
#     from sklearn.preprocessing import MinMaxScaler
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential, load_model
#     from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
#     from tensorflow.keras.callbacks import EarlyStopping
#     # Keras Backend 관련 모듈은 더 이상 learning_phase를 제공하지 않으므로 제거합니다.
#     # from tensorflow.keras import backend as K # 이 줄은 이제 필요 없습니다.

#     from sklearn.ensemble import RandomForestRegressor
#     from sklearn.metrics import mean_squared_error, r2_score
#     from sklearn.model_selection import train_test_split
# except ImportError:
#     st.error("""
#     **필수 라이브러리가 설치되지 않았습니다!**
#     아래 명령어를 실행하여 필요한 라이브러리를 설치해주세요:
#     `pip install tensorflow scikit-learn pandas matplotlib streamlit`
#     """)
#     st.stop()

# # --- Streamlit 페이지 설정 ---
# st.set_page_config(layout="wide")

# st.title("🔮 주가 예측 대시보드")
# st.markdown("과거 주가 데이터, 기술적/펀더멘털 지표, 그리고 딥러닝/머신러닝 모델을 활용하여 미래 주가 및 수익률을 예측합니다.")
# st.markdown("방대한 데이터로 인해 시간이 다소 오랫동안 소요될 수 있습니다.")
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

# # --- LSTM 모델 관련 함수 ---
# # Keras 3.x에서 Monte Carlo Dropout을 위한 커스텀 Dropout 레이어
# # @keras.saving.register_keras_serializable() 데코레이터 추가하여 저장/로드 문제 방지
# @tf.keras.saving.register_keras_serializable()
# class MCDropout(tf.keras.layers.Dropout):
#     def call(self, inputs):
#         # Dropout 레이어의 'training' 인자를 True로 강제하여 추론 시에도 Dropout이 활성화되도록 합니다.
#         # 이것이 Monte Carlo Dropout의 핵심입니다.
#         return super().call(inputs, training=True)

# def build_lstm_model(input_shape):
#     """LSTM 모델을 빌드합니다. Monte Carlo Dropout을 위해 커스텀 MCDropout 레이어를 포함합니다."""
#     model = Sequential([
#         Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
#         MCDropout(0.3), # 커스텀 MCDropout 레이어 사용
#         Bidirectional(LSTM(32, return_sequences=False)),
#         MCDropout(0.3), # 커스텀 MCDropout 레이어 사용
#         Dense(16, activation='relu'),
#         Dense(1)
#     ])
#     # 손실 함수를 명시적으로 인스턴스화하여 저장/로드 오류 방지
#     model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError()) 
#     return model

# @st.cache_resource 
# def train_and_predict_lstm_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_future_days, last_sequence, _scaler, features, n_monte_carlo_runs=100):
#     """
#     LSTM 모델을 학습하고 미래 주가를 예측합니다.
#     Monte Carlo Dropout을 사용하여 예측 불확실성 구간을 계산합니다.
#     """
#     model_path = f"lstm_model_{selected_code}.h5"
#     model = None

#     if os.path.exists(model_path):
#         try:
#             # 커스텀 객체(MCDropout)를 포함하는 모델 로드 시 custom_objects 인자 필수
#             model = load_model(model_path, custom_objects={'MCDropout': MCDropout})
#             st.success("✅ LSTM 모델을 성공적으로 로드했습니다.")
#         except Exception as e:
#             st.warning(f"⚠️ 기존 LSTM 모델 로드 중 오류 발생: {e}. 모델을 재학습합니다.")
#             os.remove(model_path) # 손상된 모델 파일 삭제
#             model = None
    
#     if model is None: # 모델 로드 실패 또는 파일 없음
#         st.info("LSTM 모델 학습이 필요합니다. 잠시만 기다려 주세요...")
#         model = build_lstm_model(input_shape=(seq_len, n_features))
#         with st.spinner("🔄 LSTM 모델 학습 중 (시간이 다소 소요될 수 있습니다)..."):
#             model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), 
#                       callbacks=[EarlyStopping(patience=7, restore_best_weights=True)], verbose=0)
#         model.save(model_path)
#         st.success("✅ LSTM 모델 학습 및 저장 완료!")

#     def recursive_forecast_with_uncertainty(model, last_sequence, n_days, _scaler_internal, n_features, features_list, n_runs):
#         all_forecasts = [] # 몬테카를로 예측 결과를 저장할 리스트

#         for _ in range(n_runs):
#             single_run_forecasts = []
#             current_seq = last_sequence.copy()
#             for __ in range(n_days):
#                 # MCDropout 레이어가 이미 training=True로 설정되어 있으므로 predict 호출 시 training 인자 불필요
#                 pred = model.predict(current_seq.reshape(1, seq_len, n_features), verbose=0)[0][0]
#                 single_run_forecasts.append(pred)

#                 new_feature_vector = np.full(n_features, 0.0)
#                 new_feature_vector[features_list.index('Close')] = pred
#                 current_seq = np.vstack([current_seq[1:], new_feature_vector])
#             all_forecasts.append(single_run_forecasts)
        
#         all_forecasts = np.array(all_forecasts) # (n_runs, n_days) 형태

#         forecasts_inverse_scaled = []
#         for run_forecast in all_forecasts:
#             dummy_array_for_inverse = np.zeros((len(run_forecast), n_features))
#             dummy_array_for_inverse[:, features_list.index('Close')] = run_forecast
#             forecasts_inverse_scaled.append(_scaler_internal.inverse_transform(dummy_array_for_inverse)[:, features_list.index('Close')])

#         forecasts_inverse_scaled = np.array(forecasts_inverse_scaled) # (n_runs, n_days) 형태

#         mean_forecast = np.mean(forecasts_inverse_scaled, axis=0)
#         std_forecast = np.std(forecasts_inverse_scaled, axis=0)

#         upper_bound = mean_forecast + 1.96 * std_forecast
#         lower_bound = mean_forecast - 1.96 * std_forecast

#         return mean_forecast, upper_bound, lower_bound

#     mean_future_preds, upper_bound_preds, lower_bound_preds = recursive_forecast_with_uncertainty(
#         model, last_sequence, n_future_days, _scaler, n_features, features, n_monte_carlo_runs
#     )
#     return mean_future_preds, upper_bound_preds, lower_bound_preds

# # --- 데이터 로드 함수 ---
# @st.cache_data
# def load_merged_data():
#     """CSV 파일에서 주가 데이터를 로드합니다."""
#     try:
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         root_dir = os.path.join(current_dir, '..')
#         merged_data_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')
        
#         if not os.path.exists(merged_data_file_path):
#             merged_data_file_path = os.path.join(current_dir, 'merged_data_monthly_per_pbr.csv')


#         df = pd.read_csv(merged_data_file_path)

#         df.columns = df.columns.str.strip()

#         df['Date'] = pd.to_datetime(df['Date'])
#         df['Code'] = df['Code'].astype(str).str.zfill(6)
#         st.success(f"✅ 데이터를 성공적으로 수집했습니다. (총 {len(df)}개 데이터 포인트)")

#         return df
#     except FileNotFoundError:
#         st.error(f"❌ 데이터 파일을 찾을 수 없습니다: '{merged_data_file_path}'")
#         st.info("데이터 파일(`merged_data_monthly_per_pbr.csv`)이 Streamlit 앱 파일과 같은 디렉토리 또는 상위 디렉토리에 있는지 확인해주세요.")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
#         return pd.DataFrame()

# # 전체 데이터 로드
# df_all_data = load_merged_data()

# # --- Streamlit UI 시작 ---
# if not df_all_data.empty:
#     name_code_dict = df_all_data.drop_duplicates(subset=['Code']).set_index('Name')['Code'].to_dict()

#     if not name_code_dict:
#         st.error("종목 리스트를 생성할 수 없습니다. 데이터 파일에 'Name' 또는 'Code' 컬럼이 올바르지 않은지 확인해주세요.")
#         st.stop()

#     selected_name = st.selectbox("🔮 **예측할 종목을 선택하세요**", sorted(name_code_dict.keys()))
#     selected_code = name_code_dict[selected_name]

#     n_days = st.slider("🗓️ **LSTM 예측 기간 (미래 일 수)**", 5, 60, 30)

#     st.markdown("---")
#     st.subheader("🤖 **머신러닝 (RandomForest) 모델 설정**")
#     st.info("RandomForest 모델은 LSTM과 별도로 다음 거래일의 수익률을 예측합니다.")

#     if st.button("🚀 **예측 시작!**"):
#         with st.spinner("데이터 준비 및 모델 예측 중..."):

#             df_stock = df_all_data[df_all_data['Code'] == selected_code].copy()
#             df_stock.sort_values('Date', inplace=True)
#             df_stock.set_index('Date', inplace=True)

#             if df_stock.empty:
#                 st.error(f"선택하신 종목 ({selected_name})에 대한 데이터가 없습니다. 다른 종목을 선택해주세요.")
#                 st.stop()

#             df_stock['RSI'] = calculate_rsi_pred(df_stock['Close'])
#             df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands_pred(df_stock['Close'])

#             if 'PER' not in df_stock.columns:
#                 st.warning("데이터에 'PER' 컬럼이 없어 0으로 처리됩니다. 정확도가 저하될 수 있습니다.")
#                 df_stock['PER'] = 0.0
#             if 'PBR' not in df_stock.columns:
#                 st.warning("데이터에 'PBR' 컬럼이 없어 0으로 처리됩니다. 정확도가 저하될 수 있습니다.")
#                 df_stock['PBR'] = 0.0

#             # --- 1. LSTM 모델 예측 ---
#             st.markdown("### **📈 LSTM 기반 미래 주가 예측**")

#             features_lstm = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR']
#             target_lstm = 'Close'

#             df_processed_lstm = df_stock[features_lstm + [target_lstm]].dropna()

#             seq_len = 20

#             if len(df_processed_lstm) < seq_len + 1:
#                 st.warning(f"[LSTM] 데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 최소 {seq_len + 1}일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_processed_lstm)}일)")
#                 mean_future_preds = None
#             else:
#                 scaler_lstm = MinMaxScaler()
#                 scaled_data_lstm = scaler_lstm.fit_transform(df_processed_lstm[features_lstm])

#                 X_lstm, y_lstm = [], []
#                 for i in range(len(scaled_data_lstm) - seq_len):
#                     X_lstm.append(scaled_data_lstm[i:i+seq_len])
#                     y_lstm.append(scaled_data_lstm[i+seq_len, features_lstm.index(target_lstm)])

#                 if not X_lstm:
#                     st.warning(f"[LSTM] 데이터 전처리 후 남은 데이터가 부족하여 {selected_name}의 미래 주가를 예측할 수 없습니다. 시퀀스 길이를 조절하거나 더 많은 데이터를 확보해주세요.")
#                     mean_future_preds = None
#                 else:
#                     X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

#                     test_split_ratio = 0.2
#                     split_idx_lstm = int(len(X_lstm) * (1 - test_split_ratio))
#                     X_train_lstm, X_test_lstm = X_lstm[:split_idx_lstm], X_lstm[split_idx_lstm:]
#                     y_train_lstm, y_test_lstm = y_lstm[:split_idx_lstm], y_lstm[split_idx_lstm:]
                    
#                     if len(X_test_lstm) == 0:
#                         st.warning(f"[LSTM] 테스트 데이터셋이 너무 작아 모델 평가에 제약이 있을 수 있습니다.")
#                         if len(X_train_lstm) < seq_len + 1:
#                             st.error("학습 데이터도 부족하여 LSTM 모델을 실행할 수 없습니다. 더 많은 데이터를 확보해주세요.")
#                             mean_future_preds = None
#                         else:
#                             X_test_lstm, y_test_lstm = X_train_lstm[-1:], y_train_lstm[-1:] 


#                     if X_train_lstm.shape[0] > 0:
#                         last_sequence_lstm = X_lstm[-1]
#                         n_features_lstm = X_lstm.shape[2]

#                         mean_future_preds, upper_bound_preds, lower_bound_preds = train_and_predict_lstm_model(
#                             X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, seq_len, n_features_lstm, 
#                             selected_code, n_days, last_sequence_lstm, scaler_lstm, features_lstm
#                         )
#                     else:
#                         st.error("LSTM 모델을 학습하기 위한 충분한 데이터가 없습니다.")
#                         mean_future_preds = None


#             if mean_future_preds is not None:
#                 last_date = df_processed_lstm.index[-1]
#                 future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]

#                 fig_lstm, ax_lstm = plt.subplots(figsize=(12, 6))

#                 plot_df_lstm = df_processed_lstm.tail(365)
#                 ax_lstm.plot(plot_df_lstm.index, plot_df_lstm['Close'], label='실제 주가', color='blue')

#                 ax_lstm.plot(future_dates, mean_future_preds, label='미래 예측 주가 (평균)', color='red', linestyle='--')
#                 ax_lstm.fill_between(future_dates, lower_bound_preds, upper_bound_preds, color='red', alpha=0.2, label='95% 신뢰 구간')

#                 ax_lstm.axvline(last_date, color='gray', linestyle=':', label='base date of forecast')
#                 ax_lstm.set_title(f"{selected_name} ({selected_code}) Future Stock Price Forecast(LSTM)")
#                 ax_lstm.set_xlabel("Date")
#                 ax_lstm.set_ylabel("Price(₩/won)")
#                 ax_lstm.legend()
#                 ax_lstm.grid(True)
#                 plt.tight_layout()
#                 st.pyplot(fig_lstm)

#                 returns_lstm = (mean_future_preds[-1] - mean_future_preds[0]) / mean_future_preds[0] * 100
#                 st.subheader("📈 **LSTM 예측 기간 수익률**")
#                 st.metric(label=f"예측 기간 수익률 ({future_dates[0].strftime('%Y-%m-%d')} ~ {future_dates[-1].strftime('%Y-%m-%d')})",
#                           value=f"{returns_lstm:.2f}%")
#             else:
#                 st.warning("LSTM 예측을 위한 데이터가 충분하지 않거나 오류가 발생했습니다.")


#             st.markdown("---")

#             # --- 2. RandomForestRegressor 모델 예측 ---
#             st.markdown("### **🚀 RandomForest 기반 다음 거래일 수익률 예측**")

#             ml_features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR']
#             df_ml = df_stock[ml_features].copy()
            
#             df_ml['Next_Day_Return'] = df_ml['Close'].pct_change().shift(-1) * 100
#             df_ml.dropna(inplace=True)

#             if len(df_ml) < 20: 
#                 st.warning(f"[RandomForest] 데이터가 부족하여 수익률 예측을 할 수 없습니다. 최소 20일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_ml)}일)")
#             else:
#                 X_ml = df_ml[ml_features].values
#                 y_ml = df_ml['Next_Day_Return'].values

#                 scaler_ml = MinMaxScaler()
#                 X_ml_scaled = scaler_ml.fit_transform(X_ml)

#                 test_size_ml = max(1, int(0.2 * len(X_ml_scaled))) 
#                 X_train_ml, X_test_ml = X_ml_scaled[:-test_size_ml], X_ml_scaled[-test_size_ml:]
#                 y_train_ml, y_test_ml = y_ml[:-test_size_ml], y_ml[-test_size_ml:]
                
#                 if len(X_test_ml) == 0:
#                     st.warning(f"[RandomForest] 테스트 데이터가 부족하여 모델 평가를 수행할 수 없습니다.")
#                     X_test_ml = X_train_ml[-1:]
#                     y_test_ml = y_train_ml[-1:] 


#                 st.info("RandomForest 모델 학습 중...")
#                 rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#                 with st.spinner("🔄 RandomForest 모델 학습 중..."):
#                     rf_model.fit(X_train_ml, y_train_ml)
#                 st.success("✅ RandomForest 모델 학습 완료!")

#                 y_pred_ml = rf_model.predict(X_test_ml)
                
#                 st.subheader("📊 **RandomForest 모델 성능 평가 (테스트 데이터)**")
#                 st.write(f"**평균 제곱 오차 (MSE)**: {mean_squared_error(y_test_ml, y_pred_ml):.2f}")
#                 st.write(f"**결정 계수 (R² Score)**: {r2_score(y_test_ml, y_pred_ml):.2f}")
#                 st.write(f"테스트 데이터의 **평균 실제 수익률**: {np.mean(y_test_ml):.2f}%")
#                 st.write(f"테스트 데이터의 **평균 예측 수익률**: {np.mean(y_pred_ml):.2f}%")

#                 last_data_ml = X_ml_scaled[-1].reshape(1, -1)
#                 next_day_return_pred_ml = rf_model.predict(last_data_ml)[0]

#                 st.subheader("📈 **RandomForest 결과**")
#                 st.metric(label="예측된 수익률", value=f"{next_day_return_pred_ml:.2f}%")

#                 if next_day_return_pred_ml > 0.5:
#                     st.success("✨ RandomForest 모델은 **강력한 상승**을 예측합니다!")
#                 elif next_day_return_pred_ml > 0:
#                     st.info("⬆️ RandomForest 모델은 **소폭 상승**을 예측합니다.")
#                 elif next_day_return_pred_ml < -0.5:
#                     st.error("🚨 RandomForest 모델은 **강력한 하락**을 예측합니다!")
#                 elif next_day_return_pred_ml < 0:
#                     st.warning("⬇️ RandomForest 모델은 **소폭 하락**을 예측합니다.")
#                 else:
#                     st.write("➖ RandomForest 모델은 **큰 변동 없음**을 예측합니다.")

# else:
#     st.info("데이터 로드 중 문제가 발생했습니다. 페이지 상단의 오류 메시지를 확인해주세요.")
