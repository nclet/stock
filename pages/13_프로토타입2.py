import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import os
import urllib.parse

# 딥러닝 관련 라이브러리 임포트
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from huggingface_hub import login # Hugging Face 로그인 함수 임포트
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import xgboost as xgb # XGBoost 임포트
except ImportError:
    st.error("""
    **필요한 라이브러리가 설치되지 않았습니다. 다음 명령어를 실행해주세요:**
    `pip install streamlit pandas numpy matplotlib requests FinanceDataReader scikit-learn tensorflow transformers huggingface_hub xgboost`
    """)
    st.stop()

# ------------------------
# ✨ 페이지 설정
# ------------------------
st.set_page_config(page_title="뉴스 감성 + LSTM/XGBoost 주가 예측", layout="wide")
st.title("뉴스 감성 + 모멘텀 + VIX + 딥러닝/XGBoost 통합 주가 예측 전략")

st.markdown("""
네이버 뉴스 감성, VIX(변동성 지수), 모멘텀 데이터를 딥러닝(LSTM) 또는 XGBoost 모델과 결합하여
기업의 미래 주가를 더 정교하게 예측하는 통합 전략 예제입니다.
""")

# ------------------------
# ✨ Hugging Face 토큰 설정 및 로그인
# ------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN")

if HF_TOKEN and HF_TOKEN.strip():
    try:
        login(token=HF_TOKEN, add_to_git_credential=True)
        st.success("✅ Hugging Face Hub에 성공적으로 로그인했습니다.")
    except Exception as e:
        st.warning(f"❌ Hugging Face Hub 로그인 중 오류 발생: {e}. 토큰이 유효한지 확인해주세요.")
else:
    st.warning("⚠️ Hugging Face 토큰이 Streamlit Secrets에 설정되지 않았거나 유효하지 않습니다. 공개 모델은 토큰 없이 시도합니다.")
    HF_TOKEN = None

# ------------------------
# ✨ 감성 분석 모델 로드
# ------------------------
@st.cache_resource
def load_sentiment_model(hf_token_val):
    model_name = "snunlp/KR-FinBert-SC"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token_val)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token_val)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        st.success(f"✅ 감성 분석 모델 '{model_name}' 로드 완료! (장치: {device})")
        st.write(f"모델 라벨 맵핑: {model.config.id2label}")
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
        st.stop()
        return None, None, None

tokenizer, sentiment_model, sentiment_device = load_sentiment_model(HF_TOKEN)

def analyze_sentiment(text):
    if not text:
        return 0.0
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(sentiment_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    
    neg_idx = None
    neu_idx = None
    pos_idx = None
    for idx, label in sentiment_model.config.id2label.items():
        if 'negative' in label.lower() or '부정' in label:
            neg_idx = idx
        elif 'neutral' in label.lower() or '중립' in label:
            neu_idx = idx
        elif 'positive' in label.lower() or '긍정' in label:
            pos_idx = idx
    
    negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    neutral_score = probabilities[neu_idx].item() if neu_idx is not None else 0
    positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

    sentiment_score = positive_score - negative_score 
    
    return sentiment_score

# ------------------------
# ✨ 종목 선택 UI
# ------------------------
@st.cache_resource
def get_company_list(market):
    return fdr.StockListing(market)

market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
company_list = get_company_list(market_option)
company_names = company_list['Name'].tolist()

if "selected_company" not in st.session_state:
    st.session_state.selected_company = "삼성전자" if "삼성전자" in company_names else company_names[0]

company_name = st.selectbox(
    "✅ 분석할 기업 선택",
    company_names,
    index=company_names.index(st.session_state.selected_company),
    key="selected_company"
)

stock_code = company_list.loc[company_list['Name'] == st.session_state.selected_company, 'Code'].values[0]

# --- 날짜 및 예측 기간 설정 ---
start_date = st.date_input("데이터 시작일", datetime.now() - timedelta(days=365 * 2)) # 2년치 데이터 기본
end_date = st.date_input("데이터 종료일", datetime.now())
prediction_horizon = st.slider("미래 주가 예측 기간 (일)", 1, 30, 5) # 최대 30일 예측

# --- 모델 선택 UI ---
selected_prediction_model = st.selectbox(
    "✅ 예측 모델 선택",
    ["LSTM 모델", "XGBoost 모델"]
)

# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(company_name, display=30, start=1, sort="date"):
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError as e:
        st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
        st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
        return pd.DataFrame()

    enc_query = urllib.parse.quote(company_name)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        items = data.get('items', [])
        news_data = []
        for item in items:
            title = item.get('title', '')
            pub_date = item.get('pubDate', '')
            try:
                pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except Exception:
                pub_date_dt = None
            news_data.append({
                'Date': pub_date_dt,
                'Title': title
            })
        df = pd.DataFrame(news_data)
        return df
    else:
        st.error(f"API 요청 실패: 상태 코드 {response.status_code}")
        return pd.DataFrame()

# ------------------------
# ✨ 기술적 지표 계산 함수 (추가)
# ------------------------
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

# ------------------------
# ✨ LSTM 모델 관련 함수
# ------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 시퀀스 데이터 생성 함수 (LSTM용)
def create_lstm_sequences(data, seq_len, n_features, target_col_idx, horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:(i + seq_len), :])
        y.append(data[i + seq_len + horizon - 1, target_col_idx])
    return np.array(X), np.array(y)

# LSTM 미래 예측 함수
def recursive_lstm_forecast(model, last_sequence, n_days, scaler, features_list, target_col_idx, seq_len, n_features):
    forecasts = []
    current_seq = last_sequence.copy()

    for _ in range(n_days):
        pred_scaled = model.predict(current_seq.reshape(1, seq_len, n_features), verbose=0)[0][0]
        
        dummy_array_for_inverse = np.zeros((1, n_features))
        dummy_array_for_inverse[0, target_col_idx] = pred_scaled
        pred_original_scale = scaler.inverse_transform(dummy_array_for_inverse)[0, target_col_idx]
        
        forecasts.append(pred_original_scale)

        new_feature_vector_scaled = current_seq[-1].copy()
        new_feature_vector_scaled[target_col_idx] = pred_scaled

        current_seq = np.vstack([current_seq[1:], new_feature_vector_scaled])
    
    return np.array(forecasts)

# ------------------------
# ✨ XGBoost 모델 관련 함수
# ------------------------
def build_xgboost_model(n_features):
    model = xgb.XGBRegressor(objective='reg:squarederror', # 회귀 문제
                             n_estimators=100,             # 트리의 개수
                             learning_rate=0.1,            # 학습률
                             max_depth=5,                  # 트리의 최대 깊이
                             subsample=0.8,                # 각 트리 학습에 사용할 샘플 비율
                             colsample_bytree=0.8,         # 각 트리 학습에 사용할 피처 비율
                             random_state=42,
                             n_jobs=-1)                    # 모든 코어 사용
    return model

# XGBoost 미래 예측 함수
def recursive_xgboost_forecast(model, last_features, n_days, scaler, features_list, target_col_idx):
    forecasts = []
    current_features_scaled = last_features.copy() # 마지막 시점의 스케일링된 특징 벡터

    for _ in range(n_days):
        # 예측 (스케일링된 값)
        pred_scaled = model.predict(current_features_scaled.reshape(1, -1))[0]
        
        # 예측된 종가를 원래 스케일로 변환
        dummy_array_for_inverse = np.zeros((1, len(features_list)))
        dummy_array_for_inverse[0, target_col_idx] = pred_scaled
        pred_original_scale = scaler.inverse_transform(dummy_array_for_inverse)[0, target_col_idx]
        
        forecasts.append(pred_original_scale)

        # 다음 예측을 위해 'Close' 특징을 예측된 값으로 업데이트 (스케일링된 상태로)
        current_features_scaled[target_col_idx] = pred_scaled
        
        # 다른 특징들은 마지막 값을 재활용 (단순화)
        # 만약 다른 특징들도 미래 예측이 필요하다면, 더 복잡한 로직이 필요합니다.
    
    return np.array(forecasts)

# ------------------------
# ✨ 실행 버튼
# ------------------------
max_news = st.slider("최대 뉴스 건수 (일별)", min_value=10, max_value=100, value=50, step=10)


if st.button("🚀 데이터 수집 및 예측 시작"):
    with st.spinner("뉴스 데이터 수집 및 감성 분석 중..."):
        all_news = pd.DataFrame()
        for start_idx in range(1, max_news + 1, 100):
            count = min(100, max_news - start_idx + 1)
            df_part = get_naver_news_api(company_name, display=count, start=start_idx)
            all_news = pd.concat([all_news, df_part], ignore_index=True)
            if len(df_part) < count:
                break
        all_news = all_news.dropna(subset=['Date'])
        filtered_news = all_news[(all_news['Date'] >= start_date) & (all_news['Date'] <= end_date)]

    if filtered_news.empty:
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 기업명을 확인해주세요.")
        st.stop()
    else:
        filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
        st.success("✅ 뉴스 감성 분석 완료!")
        st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False).head(10))

        # 일별 감성 점수 평균
        filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()

        # ------------------------
        # ✨ 주가 데이터 및 기타 지표 수집
        # ------------------------
        st.info(f"📈 {company_name} 주가 및 VIX 데이터 로드 중...")
        
        # LSTM 예측을 위해 필요한 데이터 기간보다 더 길게 가져와야 합니다.
        # 시퀀스 길이(seq_len)와 예측 기간(prediction_horizon)을 고려하여 충분히 여유롭게 가져옵니다.
        data_fetch_start_date = start_date - timedelta(days=200) 
        data_fetch_end_date = end_date + timedelta(days=prediction_horizon + 7)

        df_stock = fdr.DataReader(stock_code, data_fetch_start_date, data_fetch_end_date)
        if df_stock.empty:
            st.error("❌ 주가 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
            st.stop()
        df_stock = df_stock.reset_index()[['Date', 'Close', 'Volume']]
        df_stock['Date'] = pd.to_datetime(df_stock['Date'])
        df_stock.set_index('Date', inplace=True)
        df_stock.sort_index(inplace=True)

        # VIX 데이터
        try:
            vix_raw = fdr.DataReader('VIX', start=data_fetch_start_date, end=data_fetch_end_date)
            if vix_raw.empty:
                st.warning("⚠️ VIX 데이터를 가져오지 못했습니다. 예측에 포함되지 않습니다.")
                df_stock['VIX_Close'] = np.nan
            else:
                vix_raw.index.name = 'Date'
                vix_processed = vix_raw.reset_index()[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})
                vix_processed['Date'] = pd.to_datetime(vix_processed['Date'])
                vix_processed.set_index('Date', inplace=True)
                df_stock = df_stock.merge(vix_processed, left_index=True, right_index=True, how='left')
                st.success("✅ VIX 데이터 로드 완료!")
        except Exception as e:
            st.warning(f"⚠️ VIX 데이터 로드 중 오류 발생: {e}. 예측에 포함되지 않습니다.")
            df_stock['VIX_Close'] = np.nan

        # 모멘텀 계산
        df_stock['Momentum'] = df_stock['Close'].diff()

        # 기술적 지표 추가 (RSI, 볼린저밴드)
        df_stock['RSI'] = calculate_rsi_pred(df_stock['Close'])
        df_stock['BB_Mid'], df_stock['BB_Upper'], df_stock['BB_Lower'] = calculate_bollinger_bands_pred(df_stock['Close'])

        # 뉴스 감성 점수 병합
        filtered_news_grouped.set_index('Date', inplace=True)
        df_stock = df_stock.merge(filtered_news_grouped, left_index=True, right_index=True, how='left')
        
        # 데이터 정렬 및 결측치 처리 (fillna 전략 변경)
        df_stock['Sentiment_Score'] = df_stock['Sentiment_Score'].fillna(0)
        df_stock['VIX_Close'] = df_stock['VIX_Close'].ffill().bfill().fillna(df_stock['VIX_Close'].mean())
        df_stock['Momentum'] = df_stock['Momentum'].fillna(0)
        df_stock['RSI'] = df_stock['RSI'].ffill().bfill().fillna(df_stock['RSI'].mean())
        df_stock['BB_Mid'] = df_stock['BB_Mid'].ffill().bfill().fillna(df_stock['BB_Mid'].mean())
        df_stock['BB_Upper'] = df_stock['BB_Upper'].ffill().bfill().fillna(df_stock['BB_Upper'].mean())
        df_stock['BB_Lower'] = df_stock['BB_Lower'].ffill().bfill().fillna(df_stock['BB_Lower'].mean())

        # LSTM/XGBoost 모델에 사용할 최종 특징 목록
        features = ['Close', 'Volume', 'Sentiment_Score', 'Momentum', 'VIX_Close', 'RSI', 'BB_Upper', 'BB_Lower', 'BB_Mid']
        
        df_processed = df_stock[features].copy()
        
        # 데이터가 충분한지 다시 확인
        seq_len = 20 # LSTM 시퀀스 길이
        if len(df_processed) < seq_len + prediction_horizon:
            st.warning(f"데이터 부족: 모델 학습 및 예측에 필요한 최소 데이터 ({seq_len + prediction_horizon}일)가 부족합니다. 현재 {len(df_processed)}일. 데이터 시작일을 더 과거로 설정하거나 예측 기간을 줄여보세요.")
            st.stop()

        # 데이터 스케일링
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_processed)
        
        # 타겟 컬럼 인덱스 (Close)
        target_col_idx = features.index('Close')

        # LSTM을 위한 스케일링 및 시퀀스 생성
        # df_train_predict_cleaned를 사용하지 않고, scaled_data 전체를 사용하여 시퀀스 생성
        X_lstm, y_lstm = create_lstm_sequences(scaled_data, seq_len, len(features), target_col_idx, prediction_horizon)

        if len(X_lstm) == 0:
            st.warning(f"LSTM 학습을 위한 시퀀스가 생성되지 않았습니다. 시퀀스 길이({seq_len})나 예측 기간({prediction_horizon})을 확인해주세요.")
            st.stop()

        # XGBoost를 위한 스케일링 및 데이터 준비
        # XGBoost는 시퀀스 필요 없이 평평한 특징 벡터를 사용
        # y_xgb_flat은 scaled_data의 prediction_horizon 이후의 target_col_idx 값
        # X_xgb_flat은 y_xgb_flat과 길이가 맞도록 조정
        X_xgb_flat = scaled_data[:len(scaled_data) - prediction_horizon, :]
        y_xgb_flat = scaled_data[prediction_horizon:, target_col_idx]

        # 학습/테스트 데이터 분할 (LSTM과 XGBoost 모두 동일한 분할 사용)
        train_size_common = int(len(X_lstm) * 0.8) # LSTM 시퀀스 기준으로 분할

        # LSTM 데이터 분할
        X_lstm_train, X_lstm_test = X_lstm[:train_size_common], X_lstm[train_size_common:]
        y_lstm_train, y_lstm_test = y_lstm[:train_size_common], y_lstm[train_size_common:]

        # XGBoost 데이터 분할
        # X_xgb_flat과 y_xgb_flat의 길이를 X_lstm, y_lstm과 맞추기 위해 train_size_common을 활용
        # X_xgb_flat은 길이가 len(scaled_data) - prediction_horizon
        # y_xgb_flat은 길이가 len(scaled_data) - prediction_horizon
        # 그러므로, X_xgb_train, y_xgb_train은 X_lstm_train, y_lstm_train의 길이에 맞춰야 함
        # LSTM 시퀀스 생성 시, (len(data) - seq_len - horizon + 1) 만큼의 샘플이 생성됨
        # XGBoost는 (len(data) - horizon) 만큼의 샘플이 생성됨
        # 따라서, X_xgb_flat과 y_xgb_flat의 시작 인덱스를 조정하여 LSTM과 동일한 학습 데이터 기간을 사용하도록 합니다.
        
        # LSTM의 첫 번째 시퀀스가 시작하는 인덱스 (scaled_data 기준)
        lstm_start_idx_in_scaled_data = 0 
        # LSTM의 마지막 시퀀스가 끝나는 인덱스 (scaled_data 기준)
        lstm_end_idx_in_scaled_data = len(scaled_data) - prediction_horizon
        
        # XGBoost 학습 데이터는 LSTM 학습 데이터와 동일한 특징을 사용하지만 시퀀스 형태가 아님
        # X_xgb_train은 scaled_data[lstm_start_idx_in_scaled_data : lstm_start_idx_in_scaled_data + train_size_common]
        # y_xgb_train은 y_lstm_train과 동일
        
        # X_xgb_train, y_xgb_train을 LSTM과 동일한 샘플 수로 맞춥니다.
        # X_xgb_flat에서 LSTM 학습 데이터에 해당하는 부분만 추출
        X_xgb_train_aligned = scaled_data[ : train_size_common + seq_len -1, :] # LSTM의 X_train이 참조하는 scaled_data 범위
        y_xgb_train_aligned = y_lstm_train # LSTM의 y_train과 동일
        
        # 실제 XGBoost에 필요한 X_train은 시퀀스가 아닌 평평한 특징이므로, X_lstm_train의 마지막 시점의 특징을 사용
        # 또는, 단순히 scaled_data에서 해당 기간의 데이터를 직접 가져와서 사용
        
        # XGBoost는 시퀀스 개념이 없으므로, LSTM의 X_train에 해당하는 기간의
        # '현재 시점'의 특징들을 사용하고, 'N일 후의 종가'를 예측하도록 합니다.
        # 즉, X_xgb_train은 scaled_data의 (0 ~ train_size_common-1) 인덱스에 해당하고,
        # y_xgb_train은 scaled_data의 (prediction_horizon ~ prediction_horizon + train_size_common -1) 인덱스에 해당
        
        # 이 부분을 정확히 맞추기 위해 df_train_predict_cleaned에서 직접 데이터 추출
        X_xgb_train_df = df_train_predict_cleaned[features].iloc[:train_size_common]
        y_xgb_train_df = df_train_predict_cleaned['Future_Close'].iloc[:train_size_common]
        
        X_xgb_test_df = df_train_predict_cleaned[features].iloc[train_size_common:]
        y_xgb_test_df = df_train_predict_cleaned['Future_Close'].iloc[train_size_common:]

        # 스케일링된 값으로 변환
        X_xgb_train_scaled = scaler.transform(X_xgb_train_df)
        y_xgb_train_scaled = scaler.transform(y_xgb_train_df.values.reshape(-1, 1))[:, target_col_idx]

        X_xgb_test_scaled = scaler.transform(X_xgb_test_df)
        y_xgb_test_scaled = scaler.transform(y_xgb_test_df.values.reshape(-1, 1))[:, target_col_idx]


        # ------------------------
        # ✨ 모델 학습 및 예측 (선택된 모델에 따라)
        # ------------------------
        future_preds = np.array([])
        
        if selected_prediction_model == "LSTM 모델":
            # LSTM 모델 학습 또는 로드
            model_path = f"model_lstm_{stock_code}_finbert.h5"
            lstm_model = None

            if os.path.exists(model_path):
                try:
                    lstm_model = load_model(model_path)
                    st.info("✅ 기존 LSTM 모델 로드 완료!")
                except Exception as e:
                    st.warning(f"⚠️ 기존 LSTM 모델 로드 중 오류 발생: {e}. 모델을 다시 학습합니다.")
                    lstm_model = None
            
            if lstm_model is None:
                st.info("🔄 LSTM 모델 학습 중 (시간이 다소 소요될 수 있습니다)...")
                lstm_model = build_lstm_model(input_shape=(seq_len, len(features)))
                early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
                with st.spinner("⏳ LSTM 모델 학습 중..."):
                    lstm_model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32, validation_data=(X_lstm_test, y_lstm_test),
                                     callbacks=[early_stopping], verbose=0)
                lstm_model.save(model_path)
                st.success("✅ LSTM 모델 학습 및 저장 완료!")

            # 미래 주가 예측
            last_sequence_lstm = scaled_data[-seq_len:] # 마지막 시퀀스
            future_preds = recursive_lstm_forecast(lstm_model, last_sequence_lstm, prediction_horizon, scaler, features, target_col_idx, seq_len, len(features))
            st.success("✅ LSTM 모델 예측 완료!")

        elif selected_prediction_model == "XGBoost 모델":
            # XGBoost 모델 학습 또는 로드
            model_path = f"model_xgb_{stock_code}_finbert.json" # XGBoost는 JSON 형식으로 저장
            xgb_model = None

            if os.path.exists(model_path):
                try:
                    xgb_model = xgb.XGBRegressor()
                    xgb_model.load_model(model_path)
                    st.info("✅ 기존 XGBoost 모델 로드 완료!")
                except Exception as e:
                    st.warning(f"⚠️ 기존 XGBoost 모델 로드 중 오류 발생: {e}. 모델을 다시 학습합니다.")
                    xgb_model = None
            
            if xgb_model is None:
                st.info("🔄 XGBoost 모델 학습 중 (시간이 다소 소요될 수 있습니다)...")
                xgb_model = build_xgboost_model(len(features))
                with st.spinner("⏳ XGBoost 모델 학습 중..."):
                    # early_stopping_rounds와 verbose 인자 제거
                    xgb_model.fit(X_xgb_train_scaled, y_xgb_train_scaled,
                                  eval_set=[(X_xgb_test_scaled, y_xgb_test_scaled)])
                xgb_model.save_model(model_path)
                st.success("✅ XGBoost 모델 학습 및 저장 완료!")

            # 미래 주가 예측
            last_features_xgb = scaled_data[-1] # 마지막 날의 스케일링된 특징
            future_preds = recursive_xgboost_forecast(xgb_model, last_features_xgb, prediction_horizon, scaler, features, target_col_idx)
            st.success("✅ XGBoost 모델 예측 완료!")

        if len(future_preds) == 0:
            st.warning("예측된 주가 데이터가 없습니다. 모델 학습 또는 예측 과정에 문제가 있을 수 있습니다.")
            st.stop()

        # ------------------------
        # ✨ 결과 시각화
        # ------------------------
        st.subheader("📊 실제 주가 및 미래 예측 주가")
        fig, ax = plt.subplots(figsize=(14, 7))

        plot_df_actual = df_processed.loc[start_date:end_date]
        ax.plot(plot_df_actual.index, plot_df_actual['Close'], label='실제 주가', color='blue')

        last_actual_date = df_processed.index[-1]
        future_dates = [last_actual_date + timedelta(days=i) for i in range(1, prediction_horizon + 1)]

        ax.plot(future_dates, future_preds, label=f'예측 주가 ({prediction_horizon}일 후)', color='red', linestyle='--')

        ax.axvline(last_actual_date, color='gray', linestyle=':', label='예측 기준일')
        ax.set_title(f"{company_name} ({stock_code}) 주가 예측 ({selected_prediction_model})")
        ax.set_xlabel("날짜")
        ax.set_ylabel("종가 (₩)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📈 예측 기간 수익률")
        if len(future_preds) > 0:
            initial_price = plot_df_actual['Close'].iloc[-1]
            final_predicted_price = future_preds[-1]
            returns_predicted = (final_predicted_price - initial_price) / initial_price * 100
            st.metric(label=f"예측 기간 수익률 ({prediction_horizon}일)", value=f"{returns_predicted:.2f}%")
        else:
            st.info("예측된 주가 데이터가 없습니다.")

        st.markdown("---")
        st.subheader("📰 일별 뉴스 감성 점수 변화")
        if not filtered_news_grouped.empty:
            fig_sentiment, ax_sentiment = plt.subplots(figsize=(14, 4))
            ax_sentiment.plot(filtered_news_grouped.index, filtered_news_grouped['Sentiment_Score'], label='일별 평균 감성 점수', color='green')
            ax_sentiment.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            ax_sentiment.set_title(f"{company_name} 일별 평균 뉴스 감성 점수")
            ax_sentiment.set_xlabel("날짜")
            ax_sentiment.set_ylabel("감성 점수 (-1 ~ 1)")
            ax_sentiment.legend()
            ax_sentiment.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_sentiment)
        else:
            st.info("일별 감성 점수를 시각화할 뉴스 데이터가 없습니다.")

        st.markdown("---")
        st.write("### 참고")
        st.write("""
        - **LSTM (Long Short-Term Memory):** 시계열 데이터의 장기적인 의존성을 학습하는 데 강점을 가진 딥러닝 모델입니다.
        - **XGBoost (eXtreme Gradient Boosting):** 강력한 부스팅 기반 앙상블 모델로, 다양한 유형의 데이터에서 높은 예측 성능을 보여줍니다.
        - **통합 전략:** 뉴스 감성, 모멘텀, VIX와 같은 외부 정보가 모델의 입력 특징으로 사용되어 주가 예측의 정확도를 높이는 데 기여합니다.
        - **예측의 한계:** 주가 예측은 본질적으로 불확실성이 매우 높습니다. 이 모델들은 과거 데이터를 기반으로 학습하므로, 급격한 시장 변화나 예상치 못한 외부 요인을 완벽하게 반영하기 어렵니다. 참고 자료로만 활용하시기 바랍니다.
        """)
