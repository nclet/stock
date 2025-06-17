import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 금융 데이터 로더 및 머신러닝 라이브러리 임포트
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

st.title("🚀 주가 수익률 예측 대시보드 (RandomForest)")
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
        # 현재 시간으로 인한 캐시 무효화 방지 (캐시 지속 시간 설정)
        # KRX 종목 리스트는 매일 크게 바뀌지 않으므로 24시간 캐시 유지
        # st.cache_data(ttl=3600*24)
        df_krx = fdr.StockListing('KRX')
        # 'Code' 컬럼이 문자열이고 6자리로 채워져 있는지 확인 (선택 사항)
        df_krx['Code'] = df_krx['Code'].astype(str).str.zfill(6)
        # 종목명과 코드 매핑 딕셔너리 생성 (예: {'삼성전자': '005930', ...})
        name_code_dict = df_krx.set_index('Name')['Code'].to_dict()
        st.success("✅ KRX 종목 리스트를 성공적으로 로드했습니다.")
        return name_code_dict
    except Exception as e:
        st.error(f"❌ KRX 종목 리스트 로드 중 오류 발생: {e}")
        st.info("인터넷 연결을 확인하거나 'FinanceDataReader' 라이브러리 버전을 확인해보세요 (`pip install --upgrade FinanceDataReader`).")
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

# --- RandomForest 모델 학습 및 예측 함수 캐싱 ---
# 같은 종목을 다시 선택하면 모델을 다시 학습하지 않도록 캐시
@st.cache_resource
def train_and_predict_random_forest(selected_code, df_stock, ml_features):
    
    # 다음 날 수익률 계산 (RandomForest의 예측 목표)
    df_stock['Next_Day_Return'] = df_stock['Close'].pct_change().shift(-1) * 100
    
    # 결측치 제거
    df_ml = df_stock[ml_features + ['Next_Day_Return']].dropna()

    if len(df_ml) < 20: 
        st.warning(f"[RandomForest] 데이터가 부족하여 수익률 예측을 할 수 없습니다. 최소 20일 이상의 유효한 데이터가 필요합니다. (현재 {len(df_ml)}일)")
        return None, None, None # 모델, 예측값, 다음날 예측 수익률 반환 (없으면 None)
    
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
    with st.spinner(f"🔄 {selected_code} RandomForest 모델 학습 중..."):
        rf_model.fit(X_train_ml, y_train_ml)
    st.success("✅ RandomForest 모델 학습 완료!")

    # 모델 성능 평가
    y_pred_ml = rf_model.predict(X_test_ml)
    
    # 다음 거래일 수익률 예측
    last_data_ml = X_ml_scaled[-1].reshape(1, -1)
    next_day_return_pred_ml = rf_model.predict(last_data_ml)[0]

    return rf_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml # y_test_ml, X_test_ml도 반환하여 성능 평가에 사용

# --- Streamlit UI 시작 ---
# 모든 종목 코드 로드 (첫 로드 시 시간이 걸릴 수 있음)
name_code_dict = get_krx_stock_list()

if not name_code_dict:
    st.info("KRX 종목 리스트를 가져올 수 없습니다. 앱을 종료합니다.")
    st.stop()

# 종목 선택
selected_name = st.selectbox("🔮 **예측할 종목을 선택하세요**", sorted(name_code_dict.keys()))
selected_code = name_code_dict[selected_name]

st.markdown("---")
st.subheader("🤖 **RandomForest 모델 예측 설정**")
st.info("RandomForest 모델은 과거 주가와 기술적 지표를 기반으로 다음 거래일의 수익률을 예측합니다.")

if st.button("🚀 **예측 시작!**"):
    with st.spinner(f"'{selected_name}' 데이터 준비 및 RandomForest 모델 예측 중..."):
        # FinanceDataReader를 통해 선택된 종목의 주가 데이터 로드
        df_stock = load_stock_data_from_fdr(selected_code)

        if df_stock.empty:
            st.stop()

        # 기술적 지표 계산
        df_stock['RSI'] = calculate_rsi_pred(df_stock['Close'])
        df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands_pred(df_stock['Close'])

        # RandomForest 모델에 사용할 Features 정의 (PER/PBR 제외)
        ml_features = ['Close', 'RSI', 'BB_Upper', 'BB_Lower']
        
        # RandomForest 모델 학습 및 예측 (캐시된 함수 사용)
        rf_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml = \
            train_and_predict_random_forest(selected_code, df_stock.copy(), ml_features)
        
        if rf_model is None: # 데이터 부족 등으로 모델 학습 실패 시
            st.stop()

        st.subheader("📊 **RandomForest 모델 성능 평가 (테스트 데이터)**")
        st.write(f"**평균 제곱 오차 (MSE)**: {mean_squared_error(y_test_ml, y_pred_ml):.2f}")
        st.write(f"**결정 계수 (R² Score)**: {r2_score(y_test_ml, y_pred_ml):.2f}")
        st.write(f"테스트 데이터의 **평균 실제 수익률**: {np.mean(y_test_ml):.2f}%")
        st.write(f"테스트 데이터의 **평균 예측 수익률**: {np.mean(y_pred_ml):.2f}%")

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
