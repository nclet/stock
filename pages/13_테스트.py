import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ----------------------
# Streamlit 기본 설정
# ----------------------
st.set_page_config(page_title="Crypto Sentiment Predictor", layout="wide")

# ----------------------
# 감성 분석 모델 로드
# ----------------------
@st.cache_resource
def load_sentiment_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

# ----------------------
# 뉴스 데이터 가져오기
# ----------------------
@st.cache_data(ttl=3600)
def get_crypto_news(query="비트코인", days=7):
    url = "https://newsapi.org/v2/everything"
    api_key = st.secrets["NEWS_API_KEY"] if "NEWS_API_KEY" in st.secrets else None
    if not api_key:
        st.warning("API Key가 필요합니다. Streamlit secrets에 NEWS_API_KEY 추가하세요.")
        return []
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "ko",
        "apiKey": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    except Exception as e:
        st.error(f"뉴스 API 호출 오류: {e}")
        return []

# ----------------------
# 감성 분석 함수
# ----------------------
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    # 긍정, 부정, 중립을 모두 feature로 반영
    return {
        "sent_pos": probs[-1],
        "sent_neg": probs[0],
        "sent_score": probs[-1] - probs[0]
    }

# ----------------------
# 기술적 지표 계산
# ----------------------
def calculate_technical_indicators(df):
    df["Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["STD20"] = df["Close"].rolling(window=20).std()
    df["Upper"] = df["MA20"] + 2*df["STD20"]
    df["Lower"] = df["MA20"] - 2*df["STD20"]
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["Volatility"] = df["Return"].rolling(window=20).std()
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ----------------------
# LSTM 모델 정의
# ----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ----------------------
# 데이터 수집 & 전처리
# ----------------------
def get_market_data(ticker="BTC-USD", period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    df = calculate_technical_indicators(df)
    return df

# ----------------------
# 메인 실행
# ----------------------
def main():
    st.title("📊 Crypto Sentiment + Technical Hybrid Predictor")

    # 데이터 로딩
    df = get_market_data()

    # 뉴스 가져오기 & 감성 점수 반영
    articles = get_crypto_news("비트코인")
    if articles:
        sentiments = [analyze_sentiment(a["title"]) for a in articles[:20]]
        sent_df = pd.DataFrame(sentiments)
        avg_sentiment = sent_df.mean().to_dict()
        for k, v in avg_sentiment.items():
            df[k] = v
    else:
        df["sent_pos"] = df["sent_neg"] = df["sent_score"] = 0

    df = df.dropna()

    # Feature / Target 정의
    features = ["MA20", "Upper", "Lower", "MACD", "Signal", "RSI", "Volatility", "sent_pos", "sent_neg", "sent_score"]
    X = df[features].values
    y = df["Close"].values

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # LSTM 학습
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = LSTMModel(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

    # 예측값 생성
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    lstm_pred = model(X_test_torch).detach().numpy().flatten()

    # LightGBM 학습 (LSTM 예측 포함)
    lgb_train = lgb.Dataset(np.hstack([X_train, model(X_train_torch).detach().numpy()]), label=y_train)
    lgb_test = np.hstack([X_test, lstm_pred.reshape(-1, 1)])

    params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
    lgbm = lgb.train(params, lgb_train, num_boost_round=100)

    final_pred = lgbm.predict(lgb_test)

    # 성능 지표 출력
    rmse = mean_squared_error(y_test, final_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, final_pred)
    st.write(f"✅ RMSE: {rmse:.2f}, MAPE: {mape:.2%}")

    # 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, name="실제값"))
    fig.add_trace(go.Scatter(y=final_pred, name="예측값"))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import urllib.parse
# from json.decoder import JSONDecodeError
# import pyupbit
# import lightgbm as lgb
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# import warnings

# # 불필요한 경고 메시지를 무시하도록 설정합니다.
# warnings.filterwarnings('ignore')

# # ------------------------
# # ✨ 페이지 설정
# # ------------------------
# st.set_page_config(page_title="암호화폐 하이브리드 예측 모델", layout="wide")
# st.title("암호화폐 하이브리드 가격 예측 모델")

# st.markdown("""
# 네이버 뉴스 감성, 기술적 지표, 그리고 하이브리드 모델(LSTM + LightGBM)을 결합하여
# 주요 암호화폐의 가격을 예측하는 전략입니다.
# """)

# # ------------------------
# # ✨ 감성 분석 모델 로드
# # ------------------------
# @st.cache_resource
# def load_sentiment_model():
#     """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
#     # Streamlit Cloud 배포를 위해 허깅페이스 토큰을 secrets에서 가져옵니다.
#     hf_token = st.secrets.get("HF_TOKEN")
#     model_name = "snunlp/KR-FinBert-SC"
    
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
#         model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='cpu')
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
        
#         st.success(f"✅ 감성 분석 모델 : '{model_name}' (장치: {device})")
#         st.write(f"모델 라벨 맵핑: {model.config.id2label}")
        
#         return tokenizer, model, device
#     except Exception as e:
#         st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
#         st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
#         st.stop()
#         return None, None, None

# # Streamlit 앱 시작 시 모델 로드
# tokenizer, sentiment_model, device = load_sentiment_model()

# def analyze_sentiment(text):
#     """주어진 텍스트의 감성 점수를 계산합니다."""
#     if not text:
#         return 0.0
    
#     # 텍스트를 토큰화하고 모델에 입력
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = sentiment_model(**inputs)
    
#     # 소프트맥스 함수를 적용하여 확률로 변환
#     probabilities = torch.softmax(outputs.logits, dim=1)[0]

#     neg_idx = None
#     pos_idx = None
#     # 모델의 라벨 맵핑을 기반으로 긍정/부정 인덱스 찾기
#     for idx, label in sentiment_model.config.id2label.items():
#         if 'negative' in label.lower() or '부정' in label:
#             neg_idx = idx
#         elif 'positive' in label.lower() or '긍정' in label:
#             pos_idx = idx
    
#     # 긍정 점수에서 부정 점수를 빼서 최종 감성 점수 계산
#     negative_score = probabilities[neg_idx].item() if neg_idx is not None else 0
#     positive_score = probabilities[pos_idx].item() if pos_idx is not None else 0

#     sentiment_score = positive_score - negative_score
    
#     return sentiment_score

# # ------------------------
# # ✨ 암호화폐 종목 목록 로드 (Upbit API)
# # ------------------------
# @st.cache_data
# def get_upbit_markets():
#     """
#     Upbit API에서 원화(KRW) 마켓에 있는 모든 암호화폐 목록을 가져옵니다.
#     """
#     url = "https://api.upbit.com/v1/market/all"
#     try:
#         response = requests.get(url, params={'isDetails': 'false'})
#         response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
#         markets = response.json()
        
#         # KRW 마켓만 필터링하고 코인 이름으로 매핑
#         krw_markets = {market['korean_name']: market['market'] for market in markets if market['market'].startswith('KRW-')}
        
#         if not krw_markets:
#             st.error("❌ Upbit API에서 원화 마켓 목록을 가져오지 못했습니다.")
#             st.info("Upbit API 서버 상태를 확인하거나 잠시 후 다시 시도해주세요.")
#             st.stop()
        
#         return krw_markets
    
#     except requests.exceptions.RequestException as e:
#         st.error(f"❌ Upbit API 연결 오류: {e}")
#         st.info("인터넷 연결 상태를 확인하거나 Upbit 서버에 문제가 있을 수 있습니다.")
#         st.stop()
#         return {}
#     except JSONDecodeError as e:
#         st.error(f"❌ Upbit API 응답 파싱 오류: {e}")
#         st.stop()
#         return {}

# crypto_list = get_upbit_markets()
# company_names = list(crypto_list.keys())

# # ------------------------
# # ✨ 암호화폐 종목 선택 UI
# # ------------------------
# # 기본값 설정
# default_crypto = "비트코인"
# if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
#     st.session_state.selected_company = default_crypto if default_crypto in company_names else company_names[0]

# company_name = st.selectbox(
#     "✅ 분석할 암호화폐 선택",
#     company_names,
#     index=company_names.index(st.session_state.selected_company),
#     key="selected_company"
# )

# # 선택된 코인 이름으로 Upbit market 코드를 찾음
# stock_code = crypto_list.get(st.session_state.selected_company)

# start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=30))
# end_date = st.date_input("뉴스 검색 종료일", datetime.now())

# # ------------------------
# # ✨ 네이버 뉴스 API 함수
# # ------------------------
# def get_naver_news_api(query, display=30, start=1, sort="date"):
#     """네이버 뉴스 검색 API를 호출하여 데이터를 가져옵니다."""
#     try:
#         client_id = st.secrets["naver"]["client_id"]
#         client_secret = st.secrets["naver"]["client_secret"]
#     except KeyError as e:
#         st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
#         st.info("Streamlit Cloud 대시보드의 Settings -> Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
#         return pd.DataFrame()

#     enc_query = urllib.parse.quote(query)
#     url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

#     headers = {
#         "X-Naver-Client-Id": client_id,
#         "X-Naver-Client-Secret": client_secret
#     }

#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         data = response.json()
#         items = data.get('items', [])
#         news_data = []
#         for item in items:
#             title = item.get('title', '')
#             pub_date = item.get('pubDate', '')
#             try:
#                 pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
#             except Exception:
#                 pub_date_dt = None
#             news_data.append({
#                 'Date': pub_date_dt,
#                 'Title': title
#             })
#         df = pd.DataFrame(news_data)
#         return df
#     else:
#         st.error(f"API 요청 실패: 상태 코드 {response.status_code}")
#         return pd.DataFrame()

# # ------------------------
# # ✨ Upbit API 함수 (수정)
# # ------------------------
# @st.cache_data
# def get_upbit_candles(market, count=1000):
#     """
#     Upbit API를 통해 일별 캔들 데이터를 충분히 가져옵니다.
#     기술적 지표 계산을 위해 많은 데이터가 필요합니다.
#     """
#     df = pyupbit.get_ohlcv(market, interval="day", count=count)
#     if df is None or df.empty:
#         st.error(f"❌ {market} 데이터를 가져오지 못했습니다. 종목 코드나 Upbit 서버 상태를 확인해주세요.")
#         return pd.DataFrame()
    
#     df = df.reset_index().rename(columns={'index': 'Date'})
#     df['Date'] = pd.to_datetime(df['Date']).dt.date
#     df = df.rename(columns={'trade_price': 'Close'})
#     return df

# # ------------------------
# # ✨ 기술적 지표 계산 함수 (추가)
# # ------------------------
# def calculate_technical_indicators(df):
#     """RSI, 볼린저밴드, MACD, 골든/데드 크로스 지표를 계산합니다."""
    
#     df = df.copy()  # 원본 데이터프레임 손상 방지
    
#     # RSI (Relative Strength Index)
#     df['change'] = df['Close'].diff()
#     df['gain'] = df['change'].apply(lambda x: x if x > 0 else 0)
#     df['loss'] = df['change'].apply(lambda x: abs(x) if x < 0 else 0)
#     df['avg_gain'] = df['gain'].rolling(window=14).mean()
#     df['avg_loss'] = df['loss'].rolling(window=14).mean()
#     df['rs'] = df['avg_gain'] / df['avg_loss']
#     df['RSI'] = 100 - (100 / (1 + df['rs']))
    
#     # Bollinger Bands
#     df['MA20'] = df['Close'].rolling(window=20).mean()
#     df['stddev'] = df['Close'].rolling(window=20).std()
#     df['BB_upper'] = df['MA20'] + (df['stddev'] * 2)
#     df['BB_lower'] = df['MA20'] - (df['stddev'] * 2)
    
#     # MACD (Moving Average Convergence Divergence)
#     df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
#     df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = df['ema_12'] - df['ema_26']
#     df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
#     # Golden/Dead Cross (5일선과 20일선)
#     df['MA5'] = df['Close'].rolling(window=5).mean()
#     df['Golden_Dead_Cross'] = 0
#     df['Golden_Dead_Cross'] = np.where((df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1)), 1, df['Golden_Dead_Cross'])
#     df['Golden_Dead_Cross'] = np.where((df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1)), -1, df['Golden_Dead_Cross'])
    
#     # 다음 날 가격을 예측하기 위한 타겟 변수
#     df['target'] = df['Close'].shift(-1)
    
#     return df

# # ------------------------
# # ✨ LSTM 모델 함수 (추가)
# # ------------------------
# def build_lstm_model(data, timesteps):
#     """LSTM 모델을 빌드하고 학습시킵니다."""
    
#     # LSTM 입력 데이터 스케일링
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
#     # TimeseriesGenerator를 사용하여 시퀀스 데이터 생성
#     generator = TimeseriesGenerator(scaled_data, scaled_data, length=timesteps, batch_size=1)
    
#     # LSTM 모델 정의
#     model = Sequential([
#         LSTM(50, activation='relu', input_shape=(timesteps, 1)),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
    
#     # 모델 학습
#     model.fit(generator, epochs=5, verbose=0)
    
#     return model, scaler

# # ------------------------
# # ✨ 실행 버튼
# # ------------------------
# max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=100, value=50, step=10)
# timesteps = st.slider("LSTM 타임스텝 (과거 N일 데이터)", min_value=5, max_value=30, value=15, step=1)

# if st.button("🚀 하이브리드 모델 분석 시작"):
#     st.subheader("1. 데이터 수집 및 전처리")
    
#     with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
#         all_news = pd.DataFrame()
#         for start_idx in range(1, max_news + 1, 100):
#             count = min(100, max_news - start_idx + 1)
#             df_part = get_naver_news_api(company_name, display=count, start=start_idx)
#             all_news = pd.concat([all_news, df_part], ignore_index=True)
#             if len(df_part) < count:
#                 break
        
#         all_news = all_news.dropna(subset=['Date'])
#         filtered_news = all_news[(all_news['Date'] >= start_date) & (all_news['Date'] <= end_date)]
        
#     if filtered_news.empty:
#         st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 암호화폐명을 확인해주세요.")
#         st.stop()
#     else:
#         filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
#         st.success("✅ 뉴스 감성 분석 완료!")
#         st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False).head())

#     with st.spinner("가격 및 기술적 지표 데이터 로드 중..."):
#         df_asset = get_upbit_candles(stock_code, count=500)
#         if df_asset.empty:
#             st.error("❌ 암호화폐 가격 데이터를 가져오지 못했습니다.")
#             st.stop()
        
#         # 날짜 형식 통일 및 데이터 병합
#         df_asset['Date'] = pd.to_datetime(df_asset['Date']).dt.date
#         filtered_news['Date'] = pd.to_datetime(filtered_news['Date']).dt.date
        
#         filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
#         df_final = pd.merge(df_asset, filtered_news_grouped, on='Date', how='left').fillna(0)
        
#         # 기술적 지표 계산
#         df_final = calculate_technical_indicators(df_final)
        
#         # 날짜 범위를 다시 필터링
#         df_model = df_final[(df_final['Date'] >= pd.to_datetime(start_date).date()) & (df_final['Date'] <= pd.to_datetime(end_date).date())]
        
#         st.success("✅ 가격 및 기술적 지표 데이터 로드 및 전처리 완료!")
#         st.dataframe(df_model.tail())
        
#     st.markdown("---")
#     st.subheader("2. 하이브리드 모델 학습")
    
#     # LSTM 데이터 준비 및 학습
#     # 학습에 필요한 최소 데이터 수량 체크
#     required_data_count = timesteps + 50  # LSTM 훈련에 필요한 데이터 + 예측에 사용할 데이터
#     if len(df_model) < required_data_count:
#         st.warning(f"데이터가 부족하여 모델을 학습할 수 없습니다. (필요 데이터: 최소 {required_data_count}개)")
#         st.stop()
    
#     with st.spinner("LSTM과 LightGBM 모델 학습 중..."):
#         # LSTM 예측값 생성을 위한 데이터셋 준비
#         lstm_data_for_pred = df_model['Close'].values
#         lstm_scaler = MinMaxScaler(feature_range=(0, 1))
#         scaled_lstm_data = lstm_scaler.fit_transform(lstm_data_for_pred.reshape(-1, 1))
        
#         # LSTM 모델 학습 (훈련에 필요한 충분한 데이터 사용)
#         generator_train = TimeseriesGenerator(scaled_lstm_data, scaled_lstm_data, length=timesteps, batch_size=1)
#         lstm_model = Sequential([
#             LSTM(50, activation='relu', input_shape=(timesteps, 1)),
#             Dense(1)
#         ])
#         lstm_model.compile(optimizer='adam', loss='mean_squared_error')
#         lstm_model.fit(generator_train, epochs=5, verbose=0)
        
#         # LSTM 예측값 생성 (전체 데이터셋에 대해)
#         lstm_predictions = []
#         for i in range(len(scaled_lstm_data)):
#             if i < timesteps:
#                 lstm_predictions.append(np.nan)
#             else:
#                 input_seq = scaled_lstm_data[i-timesteps:i]
#                 prediction = lstm_model.predict(input_seq.reshape(1, timesteps, 1), verbose=0)[0][0]
#                 lstm_predictions.append(prediction)
        
#         # 스케일링된 예측값을 원래 가격 범위로 복원
#         lstm_predictions_original = lstm_scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()
#         df_model['lstm_pred'] = lstm_predictions_original
        
#         # LightGBM 데이터 준비
#         features = ['Sentiment_Score', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'MACD_Signal', 'Golden_Dead_Cross', 'lstm_pred']
        
#         # LSTM 예측값과 다른 기술적 지표로 인해 발생한 NaN 값 제거
#         df_model = df_model.dropna(subset=features + ['target'])
        
#         if df_model.empty:
#             st.error("❌ 데이터 전처리 후 학습에 사용할 데이터가 없습니다. 검색 기간을 넓혀주세요.")
#             st.stop()
            
#         X = df_model[features]
#         y = df_model['target']

#         # 학습/테스트 데이터 분리
#         split_idx = int(len(X) * 0.8)
#         X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#         y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
#         # LightGBM 모델 학습
#         lgbm_model = lgb.LGBMRegressor(random_state=42)
#         lgbm_model.fit(X_train, y_train, feature_name=features)

#     st.success("✅ 하이브리드 모델 학습 완료!")
    
#     st.markdown("---")
#     st.subheader("3. 최종 예측 및 시각화")

#     if X_test.empty:
#         st.warning("테스트 데이터가 부족합니다. 뉴스 검색 기간을 늘려주세요.")
#     else:
#         # LightGBM으로 최종 예측 수행
#         final_predictions = lgbm_model.predict(X_test)
        
#         df_test = df_model.iloc[split_idx:].copy()
#         df_test['Predicted_Close'] = final_predictions
        
#         fig, ax = plt.subplots(figsize=(12, 6))
#         ax.plot(df_test['Date'], df_test['target'], label='실제 가격 (Actual)', color='blue')
#         ax.plot(df_test['Date'], df_test['Predicted_Close'], label='예측 가격 (Predicted)', linestyle='--', color='red')
        
#         ax.set_title(f"{company_name} 하이브리드 모델 가격 예측")
#         ax.set_xlabel("날짜")
#         ax.set_ylabel("종가")
#         ax.legend()
#         ax.grid(True)
#         plt.xticks(rotation=45)
#         st.pyplot(fig)
        
#         # 피처 중요도 시각화
#         st.markdown("---")
#         st.subheader("4. 모델 피처 중요도")
#         feature_importance = pd.DataFrame({
#             'Feature': lgbm_model.feature_name_,
#             'Importance': lgbm_model.feature_importances_
#         }).sort_values(by='Importance', ascending=False)
#         st.bar_chart(feature_importance.set_index('Feature'))

#         st.info("💡 **모델 해석:** `RSI`, `MACD`, `볼린저밴드`와 같은 기술적 지표가 가격 예측에 가장 큰 영향을 미칩니다. `Sentiment_Score`와 `lstm_pred`도 의미 있는 영향을 미치는 것을 확인할 수 있습니다.")

