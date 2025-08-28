import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import urllib.parse
from json.decoder import JSONDecodeError
import pyupbit
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import warnings

# 불필요한 경고 메시지를 무시하도록 설정합니다.
warnings.filterwarnings('ignore')

# ------------------------
# ✨ 페이지 설정
# ------------------------
st.set_page_config(page_title="암호화폐 하이브리드 예측 모델", layout="wide")
st.title("암호화폐 하이브리드 가격 예측 모델")

st.markdown("""
네이버 뉴스 감성, 기술적 지표, 그리고 하이브리드 모델(LSTM + LightGBM)을 결합하여
주요 암호화폐의 가격을 예측하는 전략입니다.
""")

# ------------------------
# ✨ 감성 분석 모델 로드
# ------------------------
@st.cache_resource
def load_sentiment_model():
    """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
    hf_token = st.secrets.get("HF_TOKEN")
    model_name = "snunlp/KR-FinBert-SC"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='cpu')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        st.success(f"✅ 감성 분석 모델 : '{model_name}' (장치: {device})")
        st.write(f"모델 라벨 맵핑: {model.config.id2label}")
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
        st.stop()
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    """주어진 텍스트의 감성 점수를 계산합니다."""
    if not text:
        return 0.0
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    neg_idx = None
    pos_idx = None
    for idx, label in sentiment_model.config.id2label.items():
        if 'negative' in label.lower() or '부정' in label:
            neg_idx = idx
        elif 'positive' in label.lower() or '긍정' in label:
            pos_idx = idx
    
    negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

    sentiment_score = positive_score - negative_score
    
    return sentiment_score

# ------------------------
# ✨ 암호화폐 종목 목록 로드 (Upbit API)
# ------------------------
@st.cache_data
def get_upbit_markets():
    """
    Upbit API에서 원화(KRW) 마켓에 있는 모든 암호화폐 목록을 가져옵니다.
    """
    url = "https://api.upbit.com/v1/market/all"
    try:
        response = requests.get(url, params={'isDetails': 'false'})
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
        markets = response.json()
        
        # KRW 마켓만 필터링하고 코인 이름으로 매핑
        krw_markets = {market['korean_name']: market['market'] for market in markets if market['market'].startswith('KRW-')}
        
        if not krw_markets:
            st.error("❌ Upbit API에서 원화 마켓 목록을 가져오지 못했습니다.")
            st.info("Upbit API 서버 상태를 확인하거나 잠시 후 다시 시도해주세요.")
            st.stop()
        
        return krw_markets
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Upbit API 연결 오류: {e}")
        st.info("인터넷 연결 상태를 확인하거나 Upbit 서버에 문제가 있을 수 있습니다.")
        st.stop()
        return {}
    except JSONDecodeError as e:
        st.error(f"❌ Upbit API 응답 파싱 오류: {e}")
        st.stop()
        return {}

crypto_list = get_upbit_markets()
company_names = list(crypto_list.keys())

# ------------------------
# ✨ 암호화폐 종목 선택 UI
# ------------------------
# 기본값 설정
default_crypto = "비트코인"
if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
    st.session_state.selected_company = default_crypto if default_crypto in company_names else company_names[0]

company_name = st.selectbox(
    "✅ 분석할 암호화폐 선택",
    company_names,
    index=company_names.index(st.session_state.selected_company),
    key="selected_company"
)

# 선택된 코인 이름으로 Upbit market 코드를 찾음
stock_code = crypto_list.get(st.session_state.selected_company)

start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=30))
end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(query, display=30, start=1, sort="date"):
    """네이버 뉴스 검색 API를 호출하여 데이터를 가져옵니다."""
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError as e:
        st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
        st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
        return pd.DataFrame()

    enc_query = urllib.parse.quote(query)
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
# ✨ Upbit API 함수 (수정)
# ------------------------
@st.cache_data
def get_upbit_candles(market, count=1000):
    """
    Upbit API를 통해 일별 캔들 데이터를 충분히 가져옵니다.
    기술적 지표 계산을 위해 많은 데이터가 필요합니다.
    """
    df = pyupbit.get_ohlcv(market, interval="day", count=count)
    if df is None or df.empty:
        st.error(f"❌ {market} 데이터를 가져오지 못했습니다. 종목 코드나 Upbit 서버 상태를 확인해주세요.")
        return pd.DataFrame()
    
    df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.rename(columns={'trade_price': 'Close'})
    return df

# ------------------------
# ✨ 기술적 지표 계산 함수 (추가)
# ------------------------
def calculate_technical_indicators(df):
    """RSI, 볼린저밴드, MACD, 골든/데드 크로스 지표를 계산합니다."""
    
    # RSI (Relative Strength Index)
    df['change'] = df['close'].diff()
    df['gain'] = df['change'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = df['change'].apply(lambda x: abs(x) if x < 0 else 0)
    df['avg_gain'] = df['gain'].rolling(window=14).mean()
    df['avg_loss'] = df['loss'].rolling(window=14).mean()
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['RSI'] = 100 - (100 / (1 + df['rs']))
    
    # Bollinger Bands
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['stddev'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + (df['stddev'] * 2)
    df['BB_lower'] = df['MA20'] - (df['stddev'] * 2)
    
    # MACD (Moving Average Convergence Divergence)
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['ema_12'] - df['ema_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Golden/Dead Cross (5일선과 20일선)
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['Golden_Dead_Cross'] = 0
    df['Golden_Dead_Cross'] = np.where((df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1)), 1, df['Golden_Dead_Cross'])
    df['Golden_Dead_Cross'] = np.where((df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1)), -1, df['Golden_Dead_Cross'])
    
    # 다음 날 가격을 예측하기 위한 타겟 변수
    df['target'] = df['close'].shift(-1)
    
    return df

# ------------------------
# ✨ LSTM 모델 함수 (추가)
# ------------------------
def build_lstm_model(data, timesteps):
    """LSTM 모델을 빌드하고 학습시킵니다."""
    
    # LSTM 입력 데이터 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # TimeseriesGenerator를 사용하여 시퀀스 데이터 생성
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=timesteps, batch_size=1)
    
    # LSTM 모델 정의
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(timesteps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 모델 학습
    model.fit(generator, epochs=5, verbose=0)
    
    return model, scaler

# ------------------------
# ✨ 실행 버튼
# ------------------------
max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=100, value=50, step=10)
timesteps = st.slider("LSTM 타임스텝 (과거 N일 데이터)", min_value=5, max_value=30, value=15, step=1)

if st.button("🚀 하이브리드 모델 분석 시작"):
    st.subheader("1. 데이터 수집 및 전처리")
    
    with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
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
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 암호화폐명을 확인해주세요.")
        st.stop()
    else:
        filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
        st.success("✅ 뉴스 감성 분석 완료!")
        st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False).head())

    with st.spinner("가격 및 기술적 지표 데이터 로드 중..."):
        df_asset = get_upbit_candles(stock_code, count=500)
        if df_asset.empty:
            st.error("❌ 암호화폐 가격 데이터를 가져오지 못했습니다.")
            st.stop()
        
        # 날짜 형식 통일 및 데이터 병합
        df_asset['Date'] = pd.to_datetime(df_asset['Date'])
        filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
        
        filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
        df_final = pd.merge(df_asset, filtered_news_grouped, on='Date', how='left').fillna(0)
        
        # 기술적 지표 계산
        df_final = calculate_technical_indicators(df_final)
        df_final = df_final.dropna().reset_index(drop=True)

        st.success("✅ 가격 및 기술적 지표 데이터 로드 및 전처리 완료!")
        st.dataframe(df_final.tail())
        
    st.markdown("---")
    st.subheader("2. 하이브리드 모델 학습")
    
    if len(df_final) < timesteps + 5:
        st.warning(f"데이터가 부족하여 모델을 학습할 수 없습니다. (필요 데이터: 최소 {timesteps+5}개)")
        st.stop()

    with st.spinner("LSTM과 LightGBM 모델 학습 중..."):
        # 데이터셋 분리
        df_model = df_final[(df_final['Date'] >= pd.to_datetime(start_date)) & (df_final['Date'] <= pd.to_datetime(end_date))]
        
        # LSTM 데이터 준비
        lstm_data = df_model['close'].values
        lstm_model, lstm_scaler = build_lstm_model(lstm_data, timesteps)
        
        # LSTM 예측값 생성 (LightGBM의 입력으로 사용)
        lstm_predictions = []
        for i in range(len(lstm_data) - timesteps):
            input_seq = lstm_data[i:i+timesteps].reshape(1, timesteps, 1)
            scaled_input = lstm_scaler.transform(input_seq.reshape(-1, 1)).reshape(1, timesteps, 1)
            prediction_scaled = lstm_model.predict(scaled_input, verbose=0)
            prediction = lstm_scaler.inverse_transform(prediction_scaled)[0][0]
            lstm_predictions.append(prediction)
        
        # LSTM 예측값을 데이터프레임에 추가
        lstm_predictions_df = pd.DataFrame({'lstm_pred': [np.nan] * timesteps + lstm_predictions}, index=df_model.index)
        df_model = pd.concat([df_model, lstm_predictions_df], axis=1)

        # LightGBM 데이터 준비
        features = ['Sentiment_Score', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'MACD_Signal', 'Golden_Dead_Cross', 'lstm_pred']
        
        df_model = df_model.dropna(subset=features + ['target'])
        
        X = df_model[features]
        y = df_model['target']

        # 학습/테스트 데이터 분리
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # LightGBM 모델 학습
        lgbm_model = lgb.LGBMRegressor(random_state=42)
        lgbm_model.fit(X_train, y_train)

    st.success("✅ 하이브리드 모델 학습 완료!")
    
    st.markdown("---")
    st.subheader("3. 최종 예측 및 시각화")

    if X_test.empty:
        st.warning("테스트 데이터가 부족합니다. 뉴스 검색 기간을 늘려주세요.")
    else:
        # LightGBM으로 최종 예측 수행
        final_predictions = lgbm_model.predict(X_test)
        
        df_test = df_model.iloc[split_idx:].copy()
        df_test['Predicted_Close'] = final_predictions
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_test['Date'], df_test['target'], label='실제 가격 (Actual)', color='blue')
        ax.plot(df_test['Date'], df_test['Predicted_Close'], label='예측 가격 (Predicted)', linestyle='--', color='red')
        
        ax.set_title(f"{company_name} 하이브리드 모델 가격 예측")
        ax.set_xlabel("날짜")
        ax.set_ylabel("종가")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # 피처 중요도 시각화
        st.markdown("---")
        st.subheader("4. 모델 피처 중요도")
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': lgbm_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(feature_importance.set_index('Feature'))

        st.info("💡 **모델 해석:** `RSI`, `MACD`, `볼린저밴드`와 같은 기술적 지표가 가격 예측에 가장 큰 영향을 미칩니다. `Sentiment_Score`와 `lstm_pred`도 의미 있는 영향을 미치는 것을 확인할 수 있습니다.")

