import streamlit as st
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
# RobustScaler, StandardScaler 모두 사용
from sklearn.preprocessing import RobustScaler, StandardScaler 
import plotly.express as px
import plotly.graph_objects as go

# --- 데이터 로딩 라이브러리 및 API (동일) ---
import FinanceDataReader as fdr
import pyupbit
import requests
from json.decoder import JSONDecodeError

# --- 상수 정의 ---
TARGET_PERIOD = 10 # 예측할 미래 일수

# 시장 매핑
MARKET_MAPPING = {
    "KRX": "한국 주식 (KRX)",
    "NASDAQ": "미국 증시 (NASDAQ)",
    "COIN": "코인 (Upbit)"
}

# --- LightGBM 모델 하이퍼파라미터 (최적화 값 유지) ---
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1500,
    'learning_rate': 0.015,
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 1,
    'num_leaves': 40,
    'max_depth': 10,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'min_child_samples': 20,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# --------------------------
# 1. 멀티 마켓 종목 목록 로딩 함수 (동일)
# --------------------------

@st.cache_data(ttl=60*60*24)
def get_stock_listing(market_name):
    if market_name == 'KRX':
        market_code = 'KRX'
    elif market_name == 'NASDAQ':
        market_code = 'NASDAQ'
    else:
        return pd.DataFrame()
        
    try:
        df = fdr.StockListing(market_code)
        if 'Code' not in df.columns and 'Symbol' in df.columns:
            df.rename(columns={'Symbol': 'Code'}, inplace=True)
        if 'Code' not in df.columns or df.empty:
            st.error(f"데이터에 'Code' 또는 'Symbol' 열이 없습니다. ({market_name})")
            return pd.DataFrame()

        df['Code'] = df['Code'].astype(str)
        name_col = 'Name' if 'Name' in df.columns else 'Symbol'
        df['label'] = df[name_col].astype(str) + ' (' + df['Code'] + ')'
        return df
    except Exception as e:
        st.error(f"{market_name} 종목 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=60*60*24)
def get_coin_listing():
    try:
        url = "https://api.upbit.com/v1/market/all"
        response = requests.get(url, params={'isDetails': 'false'})
        response.raise_for_status()
        all_markets = response.json()
        
        krw_markets = [market for market in all_markets if market['market'].startswith('KRW-')]
        df_coin = pd.DataFrame(krw_markets)
        df_coin.rename(columns={'market': 'Code', 'korean_name': 'Name'}, inplace=True)
        
        df_coin['label'] = df_coin['Name'].astype(str) + ' (' + df_coin['Code'].str.replace('KRW-', '') + ')'
        
        return df_coin
    except Exception as e:
        st.error(f"코인 리스트를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


# --------------------------
# 2. 피처 엔지니어링 함수 (모멘텀, 변동성, 거래량 비율 기반으로 변경)
# --------------------------
def create_features(df, is_for_training=True):
    """
    LightGBM 모델 훈련을 위한 시계열 피처를 생성합니다. (모멘텀, 변동성, 거래량 비율 포함)
    """
    df = df.copy()

    # 1. 시간 기반 피처
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear
    
    # 2. 지연 피처 (Lag Features)
    lags = [1, 3, 7, 14, 30]
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
    # 3. 이동 평균 및 볼륨 지표
    windows = [5, 20, 60]
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Vol_{window}'] = df['Close'].rolling(window=window).std()

    # 4. 상대적인 변화율 (차분 피처) - 변동성 계산에 사용됨
    df['Daily_Change'] = df['Close'].pct_change()
    
    # --- 5. 모멘텀, 변동성, 거래량 기반 신규 지표 (핵심 지표) ---
    
    # 5. 20일 모멘텀 (가격 방향성)
    # 현재 종가가 20일 전 종가 대비 얼마나 변화했는지 (%)
    df['Momentum_20'] = df['Close'].pct_change(periods=20)
    
    # 6. 20일 변동성 (리스크)
    # 일일 변화율(Daily_Change)의 20일 표준편차 (단위 기간의 가격 변화의 표준편차)
    df['Volatility_20'] = df['Daily_Change'].rolling(window=20).std()

    # 7. 거래량 비율 (시장 참여 강도)
    # 현재 거래량을 20일 평균 거래량으로 나눈 비율
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # 기존 RSI, MACD 지표는 제거됨
    
    if is_for_training:
        # 8. 타겟 변수 (미래 1일 후의 종가) - 훈련 시에만 필요
        df['Target'] = df['Close'].shift(-1) 

    # dropna()는 가장 긴 지연 피처(Lag 30)와 이동 평균(MA 60)을 기준으로 유효한 데이터만 남깁니다.
    df = df.dropna()
    
    return df

# --------------------------
# 3. 데이터 로드 함수 (동일)
# --------------------------
@st.cache_data(ttl=60*60*4) 
def load_data(ticker, market, train_days):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=train_days + 100) 
    
    data = None
    
    try:
        if market in ['KRX', 'NASDAQ']:
            data = fdr.DataReader(ticker, start_date, end_date)
            data.index.name = 'Date'
            
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                st.error(f"데이터에 'Close' 또는 'Volume' 컬럼이 없어 처리를 계속할 수 없습니다.")
                return None
            
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
            
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        elif market == 'COIN':
            days_diff = (end_date - start_date).days
            count = days_diff + 1
            
            df_coin = pyupbit.get_ohlcv(ticker=ticker, interval='day', count=count)
            
            if df_coin is None or df_coin.empty:
                st.warning(f"오류: [{ticker}] 코인에 대한 데이터를 찾을 수 없습니다.")
                return None
                
            df_coin.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'value']
            df_coin.index.name = 'Date'
            
            df_coin['Adj Close'] = df_coin['Close']
            data = df_coin[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
        if data is None or data.empty:
            return None

        return data.dropna()
        
    except Exception as e:
        st.error(f"'{ticker}' 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# --------------------------
# 4. 모델 훈련 및 예측 함수 (스케일러 선택 기능 및 Feature Importance 포함)
# --------------------------
def train_and_validate_model(data_features, scaler_type):
    """
    시계열 분할을 이용해 LightGBM 모델을 훈련 및 검증하고 결과를 반환합니다.
    """
    
    X = data_features.drop('Target', axis=1)
    y = data_features['Target']
    
    # LightGBM이 인식할 수 있도록 컬럼명 정리
    sanitized_columns = [
        str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_').replace(' ', '_').replace(',', '')
        for col in X.columns
    ]
    X.columns = sanitized_columns
    
    # 스케일러 선택
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else: # StandardScaler
        scaler = StandardScaler() 
        
    st.info(f"선택된 스케일러: **{scaler_type}**를 사용하여 특징 데이터를 전처리합니다.")
    
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    rmse_scores = []
    
    st.markdown("##### 🚀 모델 훈련 및 시계열 검증 진행 중...")
    progress_bar = st.progress(0)
    
    final_model = None
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_scaled_df)):
        X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=-1)]
        )
        
        val_predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        rmse_scores.append(rmse)
        
        progress_bar.progress((fold + 1) / 3)
        st.caption(f"Fold {fold+1} 검증 완료. RMSE: {rmse:.4f}")
        final_model = model

    avg_rmse = np.mean(rmse_scores)
    st.success(f"✅ 모델 훈련 완료. 평균 검증 RMSE: {avg_rmse:.4f}")
    
    return final_model, scaler, X.columns, avg_rmse

def predict_future(model, scaler, last_data, feature_columns):
    """
    훈련된 모델을 사용하여 향후 TARGET_PERIOD 일간의 주가를 Walk-Forward 방식으로 예측합니다.
    """
    
    future_dates = [last_data.index[-1] + datetime.timedelta(days=i) for i in range(1, TARGET_PERIOD + 1)]
    
    predictions = []
    
    for date in future_dates:
        
        # 가상의 다음 날 데이터 생성 (Walk-Forward 방식)
        new_row = pd.DataFrame(index=[date])
        new_row['Close'] = predictions[-1] if predictions else last_data['Close'].iloc[-1]
        new_row['Volume'] = last_data['Volume'].iloc[-1] 
        
        price_cols = ['Open', 'High', 'Low', 'Adj Close']
        for col in price_cols:
              new_row[col] = new_row['Close'].iloc[0]
              
        # 다음 날 피처 생성을 위해 기존 데이터에 가상 데이터 추가
        temp_df = last_data.iloc[-60:].copy() 
        temp_df = pd.concat([temp_df, new_row])
        
        # 피처 생성 (새로운 모멘텀, 변동성 지표 포함)
        temp_df = create_features(temp_df, is_for_training=False)
        
        # 컬럼 정리
        sanitized_temp_columns = [
            str(col).replace(' ', '_') for col in temp_df.columns
        ]
        temp_df.columns = sanitized_temp_columns
        
        X_future_data = temp_df.iloc[-1].to_frame().T
        
        X_future = X_future_data[feature_columns].fillna(0)
        X_future.columns = feature_columns

        # 스케일러로 변환
        X_future_scaled = scaler.transform(X_future)
        
        next_price = model.predict(X_future_scaled)[0]
        predictions.append(next_price)
        
        # 다음 예측을 위해 'last_data' 업데이트
        last_data = pd.concat([last_data, new_row])
        
    return pd.Series(predictions, index=future_dates)

def display_feature_importance(model, feature_columns):
    """
    LightGBM 모델의 Feature Importance를 시각화합니다.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(20) # 상위 20개만 표시

    fig = px.bar(
        feature_importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='모델 특징 중요도 (Feature Importance)',
        labels={'Importance': '중요도', 'Feature': '특징 이름'}
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 5. Streamlit 메인 앱
# --------------------------
st.set_page_config(layout="wide", page_title="LGBM 멀티 자산 예측 시스템 (진단 강화)")

def app():
    st.title("🔬 LightGBM 예측 시스템: 성능 진단 강화 (모멘텀/변동성 기반)")
    st.markdown("특징을 **20일 모멘텀, 변동성, 거래량 비율**로 간결화하고, **특징 중요도**를 통해 기여도를 진단합니다.")
    st.markdown("---")

    # 1. 시장 및 종목 선택
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1]) 
    
    with col1:
        selected_market_name = st.selectbox(
            "📊 예측할 자산 시장 선택",
            list(MARKET_MAPPING.values()),
            key='market_select'
        )
    
    market_key = [k for k, v in MARKET_MAPPING.items() if v == selected_market_name][0]

    with col3:
        # 훈련 기간 선택 UI
        selected_train_days = st.number_input(
            "📅 훈련 기간 (일 단위)",
            min_value=120,
            max_value=3650,
            value=365,
            step=30,
            key='train_days_input',
            help="모델 훈련에 사용할 과거 데이터 기간 설정."
        )

    with col4:
        # 스케일러 선택 UI 추가
        selected_scaler = st.selectbox(
            "⚖️ 스케일러 선택",
            ["RobustScaler", "StandardScaler"],
            key='scaler_select',
            help="RobustScaler는 이상치에 강하고, StandardScaler는 일반적인 정규 분포에 적합합니다. 성능 비교를 위해 변경해 보세요."
        )
        
    with col2:
        # 2. 종목 목록 동적 로딩
        stock_list_df = pd.DataFrame()
        default_ticker = ""

        if market_key == 'KRX':
            stock_list_df = get_stock_listing('KRX')
            default_ticker = '005930'
            
        elif market_key == 'NASDAQ':
            stock_list_df = get_stock_listing('NASDAQ')
            default_ticker = 'AAPL'
            
        elif market_key == 'COIN':
            stock_list_df = get_coin_listing()
            default_ticker = 'KRW-BTC'
        
        if not stock_list_df.empty:
            options = stock_list_df['label'].tolist()
            try:
                default_index = options.index(stock_list_df[stock_list_df['Code'] == default_ticker]['label'].iloc[0])
            except:
                default_index = 0
                
            selected_label = st.selectbox(
                f"🏷️ 예측할 {selected_market_name} 종목/코인 선택", 
                options, 
                index=default_index, 
                key='ticker_label_select'
            )
            
            selected_ticker = stock_list_df[stock_list_df['label'] == selected_label]['Code'].iloc[0].upper().strip()
            
        else:
            st.warning("선택한 시장의 종목 목록을 불러올 수 없습니다.")
            selected_ticker = ""
    
    st.markdown("---")
    
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 3, 1])
    with col_btn_center:
        run_button = st.button("모델 훈련 및 예측 실행", type="primary", use_container_width=True)

    # 입력 검증
    if run_button and (not selected_ticker or selected_train_days < 120):
        st.warning("예측할 종목을 선택하고, 훈련 기간은 최소 120일 이상 설정해주세요.")
        return

    if run_button:
        current_market = market_key 
        
        with st.spinner(f"⏳ '{selected_ticker}' ({current_market}) 데이터 로드 및 피처 생성 중..."):
            
            # 2. 데이터 로드 및 피처 생성
            raw_data = load_data(selected_ticker, current_market, selected_train_days) 
            if raw_data is None:
                return

            data_features = create_features(raw_data, is_for_training=True)
            
            min_data_needed = 60
            if len(data_features) < min_data_needed:
                st.error(f"피처 생성 후 데이터가 너무 적습니다 ({len(data_features)}일). 훈련 기간을 최소 {selected_train_days + 50}일 이상 늘리거나 다른 종목을 선택하세요.")
                return

            train_data = data_features
            
            st.subheader(f"📊 분석 결과: {selected_label}")
            
            # 3. 모델 훈련 및 검증
            model, scaler, feature_columns, avg_rmse = train_and_validate_model(train_data, selected_scaler)
            
            # 4. 특징 중요도 시각화 (진단 기능)
            st.markdown("---")
            st.subheader("💡 훈련 모델 진단: 특징 중요도 분석")
            st.markdown(f"모델의 평균 검증 오차(RMSE): **{avg_rmse:.4f}**")
            st.markdown("새로 추가된 **Momentum_20, Volatility_20, Volume_Ratio_20** 특징들이 상위에 랭크되었는지 확인해 보세요!")
            display_feature_importance(model, feature_columns)
            
            # 5. 미래 예측
            with st.spinner(f"🔮 미래 {TARGET_PERIOD}일 예측 중 (Walk-Forward)..."):
                
                last_actual_close = raw_data['Close'].iloc[-1]
                # 예측에 필요한 충분한 과거 데이터 (기술적 지표 계산을 위해)
                last_data_for_prediction = raw_data.iloc[-100:].copy() 
                
                future_predictions_series = predict_future(
                    model, 
                    scaler, 
                    last_data_for_prediction, 
                    feature_columns 
                )
                
                st.markdown("---")
                st.subheader(f"📈 {selected_label} 가격 예측 시각화")
                
                # 결과 시각화
                past_prices = raw_data['Close'].iloc[-90:]
                
                predicted_df = pd.DataFrame({
                    'Actual': past_prices,
                    'Predicted': np.nan 
                })
                
                future_df = pd.DataFrame({
                    'Actual': np.nan,
                    'Predicted': future_predictions_series
                })
                
                final_df = pd.concat([predicted_df, future_df])
                
                st.line_chart(final_df)
                
                currency = "원" if current_market in ['KRX', 'COIN'] else "$"
                st.caption(f"마지막 실제 종가: {currency}{last_actual_close:,.2f}")

                # 예측 수치 테이블
                st.markdown(f"##### 🗓️ 향후 {TARGET_PERIOD}일 예측 종가")
                st.dataframe(future_predictions_series.to_frame(name='예측 종가').style.format(f'{currency}{{:.2f}}'))


if __name__ == "__main__":
    app()
