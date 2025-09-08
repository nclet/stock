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
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    import lightgbm as lgb
    import optuna
    import FinanceDataReader as fdr
except ImportError:
    st.error("""
    **필요 라이브러리가 설치되지 않았습니다. 다음 명령어를 실행하여 설치해주세요:**
    `pip install tensorflow scikit-learn matplotlib lightgbm optuna FinanceDataReader`
    """)
    st.stop()

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide")

st.title("🔮 통합 주가 예측 모델 (LSTM & LightGBM with Optuna)")
st.markdown("하나의 애플리케이션에서 **미래 주가 예측(LSTM)**과 **단기 수익률 예측(LightGBM)**을 모두 확인하세요.")

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

# --- 데이터 로드 함수 (기본) ---
@st.cache_data
def load_base_data():
    """CSV 파일에서 기본 주가 데이터를 로드하고 필터링합니다."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(current_dir, '..')
        merged_data_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')

        if not os.path.exists(merged_data_file_path):
            st.error(f"❌ 데이터 파일을 찾을 수 없습니다: '{merged_data_file_path}'")
            st.info("죄송합니다. 데이터 파일이 소실되어 있는 상태입니다.")
            return pd.DataFrame()

        df = pd.read_csv(merged_data_file_path)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        
        # --- 2015년 이후 데이터만 필터링하는 로직 추가 ---
        df = df[df['Date'] >= '2015-01-01'].copy()

        st.success("✅ 기본 데이터를 성공적으로 로드했습니다.")
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
def train_and_predict_lstm_model(X_train, y_train, X_test, y_test, seq_len, n_features, selected_code, n_future_days, last_sequence, _scaler, features):
    """LSTM 모델을 학습하고 미래 주가를 예측합니다."""
    model_path = f"model_lstm_{selected_code}.h5"
    model = None

    if os.path.exists(model_path):
        st.info("✅ 기존 학습 모델을 로드합니다.")
        model = load_model(model_path)
    else:
        st.info("🔄 새로운 LSTM 모델 학습이 필요합니다. 잠시만 기다려 주세요...")
        model = build_lstm_model(input_shape=(seq_len, n_features))
        with st.spinner("⏳ 모델 학습 중 (시간이 다소 소요될 수 있습니다)..."):
            model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
        model.save(model_path)
        st.success("✅ LSTM 모델 학습 완료 및 저장!")

    def recursive_forecast(model, last_sequence, n_days, scaler_inner, n_features, features_list):
        forecasts = []
        current_seq = last_sequence.copy()
        try:
            close_idx = features_list.index('Close')
        except ValueError:
            st.error("오류: 예측에 필요한 'Close' 가격 데이터가 없습니다.")
            return []
            
        for _ in range(n_days):
            pred = model.predict(current_seq.reshape(1, seq_len, n_features), verbose=0)[0][0]
            forecasts.append(pred)
            
            new_feature_vector = np.zeros(n_features)
            new_feature_vector[close_idx] = pred
            
            current_seq = np.vstack([current_seq[1:], new_feature_vector])
        
        dummy_array_for_inverse = np.zeros((len(forecasts), n_features))
        dummy_array_for_inverse[:, close_idx] = forecasts
        forecasts_scaled = scaler_inner.inverse_transform(dummy_array_for_inverse)[:, close_idx]
        return forecasts_scaled

    future_preds = recursive_forecast(model, last_sequence, n_future_days, _scaler, n_features, features)
    return future_preds

# --- 머신러닝 (LightGBM) 관련 함수 ---
@st.cache_resource
def train_and_predict_lightgbm_with_optuna(selected_code, df_stock_data, ml_features):
    """LightGBM 모델을 Optuna로 하이퍼파라미터 최적화하여 학습하고 다음 날 수익률을 예측합니다."""
    df_stock_data['Next_Day_Return'] = df_stock_data['Close'].pct_change().shift(-1) * 100
    df_ml = df_stock_data[ml_features + ['Next_Day_Return']].dropna()

    if len(df_ml) < 20:
        st.warning(f"데이터가 부족하여 수익률 예측을 할 수 없습니다. 최소 20일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_ml)}일)")
        return None, None, None, None, None, None

    X_ml = df_ml[ml_features].values
    y_ml = df_ml['Next_Day_Return'].values
    scaler_ml = MinMaxScaler()
    X_ml_scaled = scaler_ml.fit_transform(X_ml)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_index, test_index in tscv.split(X_ml_scaled):
            X_train_cv, X_test_cv = X_ml_scaled[train_index], X_ml_scaled[test_index]
            y_train_cv, y_test_cv = y_ml[train_index], y_ml[test_index]
            
            model_cv = lgb.LGBMRegressor(**params)
            model_cv.fit(X_train_cv, y_train_cv,
                         eval_set=[(X_test_cv, y_test_cv)],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
            
            y_pred_cv = model_cv.predict(X_test_cv)
            scores.append(mean_squared_error(y_test_cv, y_pred_cv))
        
        return np.mean(scores)

    with st.spinner(f"🔄 {selected_code} LightGBM 하이퍼파라미터 최적화 중 (Optuna)..."):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    st.success(f"✅ Optuna 최적화 완료! 최적의 파라미터:")
    st.json(study.best_params)

    test_size_ml = max(1, int(0.2 * len(X_ml_scaled)))
    X_train_ml, X_test_ml = X_ml_scaled[:-test_size_ml], X_ml_scaled[-test_size_ml:]
    y_train_ml, y_test_ml = y_ml[:-test_size_ml], y_ml[-test_size_ml:]
    
    lgbm_model = lgb.LGBMRegressor(**study.best_params)
    with st.spinner(f"🔄 {selected_code} 최적화된 LightGBM 모델 학습 중..."):
        lgbm_model.fit(X_train_ml, y_train_ml)
    st.success("✅ LightGBM 모델 최종 학습 완료!")

    y_pred_ml = lgbm_model.predict(X_test_ml)
    last_data_ml_raw = df_ml[ml_features].iloc[-1].values.reshape(1, -1)
    last_data_ml_scaled = scaler_ml.transform(last_data_ml_raw)
    next_day_return_pred_ml = lgbm_model.predict(last_data_ml_scaled)[0]

    return lgbm_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml, study.best_params

# --- Streamlit UI 시작 ---
df_all_data = load_base_data()

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
        
        # --- 추가 지표 로딩 (FinanceDataReader) 및 병합
        with st.spinner(f"🔄 {selected_name}의 추가 데이터 로딩 중..."):
            try:
                # FinanceDataReader를 사용하여 추가 지표 데이터 로드
                df_fdr = fdr.DataReader(selected_code, start=df_stock.index.min(), end=df_stock.index.max())
                df_fdr.reset_index(inplace=True)
                
                # 병합할 컬럼 목록 동적 생성 (존재하는 컬럼만 선택)
                fdr_cols = ['Date', 'Volume', 'Amount', 'Foreign_Net', 'Institution_Net']
                available_fdr_cols = [col for col in fdr_cols if col in df_fdr.columns]
                
                if len(available_fdr_cols) > 1: # 'Date' 컬럼 제외
                    df_stock.reset_index(inplace=True)
                    df_merged = pd.merge(df_stock, df_fdr[available_fdr_cols], on='Date', how='left')
                    df_merged.set_index('Date', inplace=True)
                    df_merged.sort_index(inplace=True)
                    st.success("✅ 추가 지표 데이터 로딩 및 병합 완료!")
                    df_stock = df_merged # 병합된 데이터로 업데이트
                else:
                    st.warning("⚠️ `finance-datareader`에서 추가 지표를 찾을 수 없습니다. 기존 데이터만 사용합니다.")

            except Exception as e:
                st.warning(f"⚠️ `finance-datareader` 데이터 로딩 또는 병합 중 오류가 발생했습니다: {e}")
                st.info("기존 데이터만 사용하여 예측을 진행합니다.")

        # --- LSTM 모델 예측 섹션 ---
        st.header("1️⃣ LSTM 모델: 미래 주가 예측")
        
        # 지표 목록을 데이터프레임에 존재하는 컬럼만으로 동적 구성
        potential_features_lstm = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'PER', 'PBR', 'BB_Mid', 
                                   'Volume', 'Amount', 'Foreign_Net', 'Institution_Net']
        features_lstm = [col for col in potential_features_lstm if col in df_stock.columns]
            
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

            # --- 날짜 계산 로직 수정: 인덱스 타입을 확실하게 변환 ---
            last_date = df_processed_lstm.index[-1]
            # 인덱스가 날짜 형식이 아닌 경우, 명시적으로 변환합니다.
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.to_datetime(last_date)

            future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]

            st.subheader("📊 LSTM 예측 주가 시각화")
            fig, ax = plt.subplots(figsize=(12, 6))
            # 시각화 데이터 기간 수정 (최근 1년만 표시)
            plot_df = df_processed_lstm.tail(252) # 영업일 기준 1년
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

        # --- LightGBM 모델 예측 섹션 ---
        st.markdown("---")
        st.header("2️⃣ LightGBM 모델: 단기 수익률 예측 (with Optuna)")
        
        # 지표 목록을 데이터프레임에 존재하는 컬럼만으로 동적 구성
        potential_features_ml = ['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'BB_Mid', 
                                 'Volume', 'Amount', 'Foreign_Net', 'Institution_Net']
        ml_features = [col for col in potential_features_ml if col in df_stock.columns]
            
        lgbm_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml, best_params = train_and_predict_lightgbm_with_optuna(selected_code, df_stock.copy(), ml_features)

        if lgbm_model is not None:
            st.subheader("📊 LightGBM 모델 성능 평가")
            st.write(f"**평균 제곱 오차 (MSE)**: {mean_squared_error(y_test_ml, y_pred_ml):.2f}")
            st.write(f"**결정 계수 (R² Score)**: {r2_score(y_test_ml, y_pred_ml):.2f}")
            st.write(f"테스트 데이터의 **평균 실제 수익률**: {np.mean(y_test_ml):.2f}%")
            st.write(f"테스트 데이터의 **평균 예측 수익률**: {np.mean(y_pred_ml):.2f}%")

            st.subheader("📈 LightGBM 다음 날 수익률 예측")
            st.metric(label="예측된 수익률", value=f"{next_day_return_pred_ml:.2f}%")
            
            # 예측 시각화 (실제 수익률과 예측 수익률 비교)
            st.markdown("---")
            st.subheader("📉 LightGBM 예측 vs. 실제 수익률")
            fig_lgbm, ax_lgbm = plt.subplots(figsize=(12, 6))
            ax_lgbm.plot(y_test_ml, label='실제 수익률', color='blue', marker='o', linestyle='None', alpha=0.6)
            ax_lgbm.plot(y_pred_ml, label='예측 수익률', color='red', marker='x', linestyle='None', alpha=0.6)
            ax_lgbm.set_title(f"{selected_name} ({selected_code}) LightGBM 예측 수익률")
            ax_lgbm.set_xlabel("데이터 포인트 인덱스")
            ax_lgbm.set_ylabel("수익률(%)")
            ax_lgbm.legend()
            ax_lgbm.grid(True)
            plt.tight_layout()
            st.pyplot(fig_lgbm)

else:
    st.info("데이터 로드 중 문제가 발생했습니다. 페이지 상단의 오류 메시지를 확인해주세요.")
