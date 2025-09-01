# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# import torch
# from torch import nn
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
# import lightgbm as lgb
# import pyupbit
# import urllib.parse
# import matplotlib.pyplot as plt
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from datetime import datetime, timedelta
# import warnings
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from sklearn.preprocessing import MinMaxScaler
# from json.decoder import JSONDecodeError
# import pytrends
# from pytrends.request import TrendReq

# # Ignore unnecessary warnings
# warnings.filterwarnings('ignore')

# # ----------------------
# # Streamlit App Setup
# # ----------------------
# st.set_page_config(page_title="Crypto Sentiment & Trend Predictor", layout="wide")
# st.title("📊 하이브리드 암호화폐 가격 예측 모델 (고급)")

# st.markdown("""
# 이 모델은 **기술적 지표**, **뉴스 감성**, **시장 심리(Fear & Greed Index)**, 그리고
# **구글 트렌드**를 결합하여 주요 암호화폐의 가격을 예측합니다.
# """)

# # ----------------------
# # Hugging Face Sentiment Model Loading
# # ----------------------
# @st.cache_resource
# def load_sentiment_model():
#     """Loads a Korean sentiment analysis model from Hugging Face."""
#     hf_token = st.secrets.get("HF_TOKEN")
#     model_name = "snunlp/KR-FinBert-SC"
    
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
#         model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='cpu')
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
        
#         st.success(f"✅ 감성 분석 모델 '{model_name}' (장치: {device}) 로드 완료")
#         st.write(f"모델 라벨 맵핑: {model.config.id2label}")
        
#         return tokenizer, model, device
#     except Exception as e:
#         st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
#         st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
#         return None, None, None

# tokenizer, sentiment_model, device = load_sentiment_model()

# # ----------------------
# # Sentiment Analysis Function
# # ----------------------
# def analyze_sentiment(text):
#     """Calculates sentiment scores for the given text."""
#     if not text:
#         return {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0, 'sentiment_score': 0.0}
    
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = sentiment_model(**inputs)
    
#     probabilities = torch.softmax(outputs.logits, dim=1)[0]

#     neg_idx = None
#     pos_idx = None
#     neu_idx = None
#     for idx, label in sentiment_model.config.id2label.items():
#         if 'negative' in label.lower() or '부정' in label:
#             neg_idx = idx
#         elif 'positive' in label.lower() or '긍정' in label:
#             pos_idx = idx
#         elif 'neutral' in label.lower() or '중립' in label:
#             neu_idx = idx
    
#     pos_score = probabilities[pos_idx].item() if pos_idx is not None else 0
#     neu_score = probabilities[neu_idx].item() if neu_idx is not None else 0
#     neg_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    
#     sentiment_score = pos_score - neg_score
    
#     return {
#         'positive': pos_score,
#         'neutral': neu_score,
#         'negative': neg_score,
#         'sentiment_score': sentiment_score
#     }

# # ----------------------
# # Upbit API Integration
# # ----------------------
# @st.cache_data
# def get_upbit_markets():
#     """Fetches all KRW crypto markets from Upbit."""
#     url = "https://api.upbit.com/v1/market/all"
#     try:
#         response = requests.get(url, params={'isDetails': 'false'})
#         response.raise_for_status()
#         markets = response.json()
        
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

# # ----------------------
# # Naver News API
# # ----------------------
# def get_naver_news_api(query, display=30, start=1, sort="date"):
#     """Fetches news data from Naver News API."""
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

# # ----------------------
# # Fear & Greed Index
# # ----------------------
# @st.cache_data
# def get_fear_greed_index():
#     """Fetches Fear & Greed Index from alternative source."""
#     url = "https://api.alternative.me/fng/?limit=1000"
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json().get('data', [])
#         df = pd.DataFrame(data)
#         df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
#         df['value'] = df['value'].astype(float)
#         return df[['timestamp', 'value']].rename(columns={'timestamp': 'date', 'value': 'fng_index'})
#     except Exception as e:
#         st.error(f"❌ Fear & Greed Index 데이터를 가져오는 중 오류가 발생했습니다: {e}")
#         return pd.DataFrame()

# # ----------------------
# # Google Trends Data
# # ----------------------
# @st.cache_data
# def get_google_trends(keyword, start_date, end_date):
#     """Fetches Google Trends data for a keyword."""
#     pytrends = TrendReq(hl='ko-KR', tz=360)
#     df = pd.DataFrame()
#     try:
#         pytrends.build_payload([keyword], cat=0, timeframe=f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}', geo='', gprop='')
#         df = pytrends.interest_over_time()
#         if not df.empty:
#             df = df.reset_index()
#             df.rename(columns={'date': 'date', keyword: 'google_trends'}, inplace=True)
#             return df[['date', 'google_trends']]
#     except Exception as e:
#         st.error(f"❌ 구글 트렌드 데이터를 가져오는 중 오류가 발생했습니다: {e}")
#         st.info("429 오류는 잠시 후 다시 시도하면 해결될 수 있습니다.")
#     return pd.DataFrame()

# # ----------------------
# # Technical Indicators Calculation
# # ----------------------
# def calculate_technical_indicators(df):
#     """Calculates RSI, Bollinger Bands, MACD, and Volatility."""
#     df = df.copy()
    
#     # RSI (Relative Stength Index)
#     df['change'] = df['close'].diff()
#     df['gain'] = df['change'].apply(lambda x: x if x > 0 else 0)
#     df['loss'] = df['change'].apply(lambda x: abs(x) if x < 0 else 0)
#     df['avg_gain'] = df['gain'].rolling(window=14).mean()
#     df['avg_loss'] = df['loss'].rolling(window=14).mean()
#     df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-8)
#     df['RSI'] = 100 - (100 / (1 + df['rs']))
    
#     # Bollinger Bands
#     df['MA20'] = df['close'].rolling(window=20).mean()
#     df['stddev'] = df['close'].rolling(window=20).std()
#     df['BB_upper'] = df['MA20'] + (df['stddev'] * 2)
#     df['BB_lower'] = df['MA20'] - (df['stddev'] * 2)
    
#     # MACD (Moving Average Convergence Divergence)
#     df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
#     df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = df['ema_12'] - df['ema_26']
#     df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

#     # Volatility
#     df['Daily_Return'] = df['close'].pct_change()
#     df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
#     # Target variable for next day's price prediction
#     df['target'] = df['close'].shift(-1)
    
#     return df

# # ----------------------
# # Main App Logic
# # ----------------------
# def main():
#     # Crypto selection UI
#     default_crypto = "비트코인"
#     if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
#         st.session_state.selected_company = default_crypto if default_crypto in company_names else company_names[0]

#     company_name = st.selectbox(
#         "✅ 분석할 암호화폐 선택",
#         company_names,
#         index=company_names.index(st.session_state.selected_company),
#         key="selected_company"
#     )
#     stock_code = crypto_list.get(st.session_state.selected_company)

#     # Main area for parameters
#     st.markdown("---")
#     st.subheader("⚙️ 모델 파라미터 설정")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         period = st.selectbox("📊 가격 데이터 기간", ["100일", "300일", "500일"], index=1)
#         news_period = st.selectbox("📰 감성 분석 기간", ["1일", "3일", "5일", "7일"], index=1)
#     with col2:
#         lstm_epochs = st.slider("📈 LSTM 에포크 수", min_value=10, max_value=100, value=30, step=5)
#         timesteps = st.slider("⏳ LSTM 시퀀스 길이", min_value=5, max_value=30, value=15, step=1)

#     count_map = {"100일": 100, "300일": 300, "500일": 500}
#     data_count = count_map.get(period, 300)
#     news_days = int(news_period.replace('일', ''))

#     st.markdown("---")

#     if st.button("🚀 하이브리드 모델 분석 시작", use_container_width=True):
#         st.subheader("1. 데이터 수집 및 전처리")
        
#         with st.spinner("가격 데이터 로드 및 전처리 중..."):
#             df_asset = pyupbit.get_ohlcv(stock_code, interval="day", count=data_count)
#             if df_asset is None or df_asset.empty:
#                 st.error(f"❌ {stock_code} 데이터를 가져오지 못했습니다. 종목 코드나 Upbit 서버 상태를 확인해주세요.")
#                 st.stop()
            
#             df_asset = df_asset.reset_index().rename(columns={'index': 'date', 'trade_price': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'})
#             df_asset['date'] = pd.to_datetime(df_asset['date']).dt.date
#             df_asset = calculate_technical_indicators(df_asset)

#         with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
#             end_date_news = datetime.now().date()
#             start_date_news = end_date_news - timedelta(days=news_days)
#             date_range = [start_date_news + timedelta(days=i) for i in range(news_days + 1)]
            
#             all_news = pd.DataFrame()
#             for date_day in date_range:
#                 date_str = date_day.strftime("%Y.%m.%d")
#                 df_part = get_naver_news_api(f"{company_name} {date_str}", display=50)
#                 df_part['Date'] = date_day
#                 all_news = pd.concat([all_news, df_part], ignore_index=True)
            
#             all_news = all_news.dropna(subset=['Date'])
#             sentiment_results = all_news['Title'].apply(lambda x: analyze_sentiment(x))
#             sentiment_df = pd.json_normalize(sentiment_results)
#             all_news = pd.concat([all_news, sentiment_df], axis=1)

#             news_grouped = all_news.groupby('Date').agg(
#                 positive=('positive', 'mean'),
#                 neutral=('neutral', 'mean'),
#                 negative=('negative', 'mean')
#             ).reset_index()

#             news_grouped.rename(columns={'Date': 'date'}, inplace=True)
#             df_final = pd.merge(df_asset, news_grouped, on='date', how='left').fillna(0)
            
#         with st.spinner("Fear & Greed Index 및 구글 트렌드 데이터 로드 중..."):
#             df_fng = get_fear_greed_index()
#             df_fng['date'] = df_fng['date'].dt.date
#             df_final = pd.merge(df_final, df_fng, on='date', how='left').fillna(0)

#             keyword_to_search = company_name.lower()
#             df_trends = get_google_trends(keyword_to_search, df_final['date'].min(), df_final['date'].max())
            
#             if not df_trends.empty:
#                 df_trends['date'] = pd.to_datetime(df_trends['date']).dt.date
#                 df_final = pd.merge(df_final, df_trends, on='date', how='left').fillna(0)
#             else:
#                 st.warning("⚠️ 구글 트렌드 데이터를 불러오는 데 실패했습니다. 해당 지표를 제외하고 분석을 진행합니다.")
                
#         st.success("✅ 모든 데이터 수집 및 병합 완료!")
#         st.dataframe(df_final[['date', 'close', 'positive', 'fng_index', 'google_trends']].tail())
            
#         st.markdown("---")
#         st.subheader("2. 하이브리드 모델 학습")
        
#         # LSTM 데이터 준비 및 학습
#         features_lstm = ['close']
#         X_lstm = df_final[features_lstm].values
#         scaler_lstm = MinMaxScaler(feature_range=(0,1))
#         scaled_lstm_data = scaler_lstm.fit_transform(X_lstm)
        
#         generator_train = TimeseriesGenerator(
#             scaled_lstm_data, scaled_lstm_data, length=timesteps, batch_size=1
#         )
        
#         model_lstm = Sequential([
#             LSTM(50, activation='relu', input_shape=(timesteps, len(features_lstm))),
#             Dense(1)
#         ])
#         model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        
#         with st.spinner("LSTM 모델 학습 중..."):
#             model_lstm.fit(generator_train, epochs=lstm_epochs, verbose=0)
        
#         lstm_predictions = []
#         for i in range(len(scaled_lstm_data)):
#             if i < timesteps:
#                 lstm_predictions.append(np.nan)
#             else:
#                 input_seq = scaled_lstm_data[i - timesteps:i].reshape(1, timesteps, len(features_lstm))
#                 prediction_scaled = model_lstm.predict(input_seq, verbose=0)[0][0]
#                 lstm_predictions.append(prediction_scaled)

#         lstm_predictions_original = scaler_lstm.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()
#         df_final['lstm_pred'] = lstm_predictions_original
        
#         # LightGBM 데이터 준비
#         features_lgbm = [
#             'open', 'high', 'low', 'close', 'volume', 'positive', 'neutral', 'negative', 
#             'RSI', 'BB_upper', 'BB_lower', 'MACD', 'MACD_Signal', 'Volatility', 'lstm_pred',
#             'fng_index', 'google_trends'
#         ]
        
#         df_model = df_final.dropna(subset=features_lgbm + ['target'])
        
#         if df_model.empty:
#             st.error("❌ 데이터 전처리 후 학습에 사용할 데이터가 없습니다. 기간을 늘려주세요.")
#             st.stop()
            
#         X = df_model[features_lgbm]
#         y = df_model['target']

#         split_idx = int(len(X) * 0.8)
#         X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#         y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
#         with st.spinner("LightGBM 모델 학습 중..."):
#             lgbm_model = lgb.LGBMRegressor(random_state=42)
#             lgbm_model.fit(X_train, y_train, feature_name=features_lgbm)

#         st.success("✅ 하이브리드 모델 학습 완료!")
        
#         st.markdown("---")
#         st.subheader("3. 최종 예측 및 모델 성능 평가")

#         if X_test.empty:
#             st.warning("테스트 데이터가 부족합니다. 기간을 늘려주세요.")
#         else:
#             final_predictions = lgbm_model.predict(X_test)
            
#             # Performance metrics
#             rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
#             mape = mean_absolute_percentage_error(y_test, final_predictions) * 100
#             r2 = r2_score(y_test, final_predictions)
            
#             st.markdown(f"""
#             - **RMSE (제곱근 평균 제곱 오차):** `{rmse:.2f}`
#             - **MAPE (평균 절대 백분율 오차):** `{mape:.2f}%`
#             - **R² (결정 계수):** `{r2:.2f}`
#             """)

#             # Visualization
#             st.subheader("📈 가격 예측 차트")
#             df_test = df_final.iloc[split_idx:].copy()
#             df_test.loc[df_test.index, 'Predicted_Close'] = pd.Series(final_predictions, index=X_test.index)
            
#             fig, ax = plt.subplots(figsize=(12, 6))
#             ax.plot(df_test['date'], df_test['close'], label='실제 가격 (Actual)', color='blue')
#             ax.plot(df_test['date'], df_test['Predicted_Close'], label='예측 가격 (Predicted)', linestyle='--', color='red')
            
#             ax.set_title(f"{company_name} 하이브리드 모델 가격 예측")
#             ax.set_xlabel("날짜")
#             ax.set_ylabel("종가")
#             ax.legend()
#             ax.grid(True)
#             plt.xticks(rotation=45)
#             st.pyplot(fig)
            
#             st.markdown("---")
#             st.subheader("4. 추가 지표 시각화")
            
#             # F&G Index Visualization
#             fig_fng, ax_fng = plt.subplots(figsize=(12, 4))
#             ax_fng.plot(df_final['date'], df_final['fng_index'], label='Fear & Greed Index', color='purple')
#             ax_fng.set_title("Fear & Greed Index")
#             ax_fng.set_xlabel("날짜")
#             ax_fng.set_ylabel("지수 (0-100)")
#             ax_fng.legend()
#             ax_fng.grid(True)
#             plt.xticks(rotation=45)
#             st.pyplot(fig_fng)
            
#             # Google Trends Visualization
#             fig_trends, ax_trends = plt.subplots(figsize=(12, 4))
#             ax_trends.plot(df_final['date'], df_final['google_trends'], label='Google Trends', color='orange')
#             ax_trends.set_title("Google 검색 트렌드")
#             ax_trends.set_xlabel("날짜")
#             ax_trends.set_ylabel("상대적 검색량")
#             ax_trends.legend()
#             ax_trends.grid(True)
#             plt.xticks(rotation=45)
#             st.pyplot(fig_trends)
            

#             # Feature Importance
#             st.markdown("---")
#             st.subheader("5. 모델 피처 중요도")
#             feature_importance = pd.DataFrame({
#                 'Feature': lgbm_model.feature_name_,
#                 'Importance': lgbm_model.feature_importances_
#             }).sort_values(by='Importance', ascending=False)
#             st.bar_chart(feature_importance.set_index('Feature'))
            
#             st.info("💡 **모델 해석:** `RSI`, `MACD`, `볼린저밴드`와 같은 기술적 지표가 가격 예측에 가장 큰 영향을 미칩니다. 뉴스 감성, Fear & Greed Index, 그리고 구글 트렌드 값도 중요한 영향을 미치는 것을 확인할 수 있습니다.")
            
# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import numpy as np
import requests
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import lightgbm as lgb
import pyupbit
import urllib.parse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from json.decoder import JSONDecodeError
import pytrends
from pytrends.request import TrendReq

# Ignore unnecessary warnings
warnings.filterwarnings('ignore')

# ----------------------
# Streamlit App Setup
# ----------------------
st.set_page_config(page_title="Crypto Sentiment & Trend Predictor", layout="wide")
st.title("📊 하이브리드 암호화폐 가격 예측 모델 (고급)")

st.markdown("""
이 모델은 **기술적 지표**, **뉴스 감성**, **시장 심리(Fear & Greed Index)**, 그리고
**구글 트렌드**를 결합하여 주요 암호화폐의 가격을 예측합니다.
""")

# ----------------------
# Hugging Face Sentiment Model Loading
# ----------------------
@st.cache_resource
def load_sentiment_model():
    """Loads a Korean sentiment analysis model from Hugging Face."""
    hf_token = st.secrets.get("HF_TOKEN")
    model_name = "snunlp/KR-FinBert-SC"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='cpu')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        st.success(f"✅ 감성 분석 모델 '{model_name}' (장치: {device}) 로드 완료")
        st.write(f"모델 라벨 맵핑: {model.config.id2label}")
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰이 Streamlit Secrets에 올바르게 설정되었는지, 라이브러리 버전이 최신인지 확인해주세요.")
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

# ----------------------
# Sentiment Analysis Function
# ----------------------
def analyze_sentiment(text):
    """Calculates sentiment scores for the given text."""
    if not text:
        return {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0, 'sentiment_score': 0.0}
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    neg_idx = None
    pos_idx = None
    neu_idx = None
    for idx, label in sentiment_model.config.id2label.items():
        if 'negative' in label.lower() or '부정' in label:
            neg_idx = idx
        elif 'positive' in label.lower() or '긍정' in label:
            pos_idx = idx
        elif 'neutral' in label.lower() or '중립' in label:
            neu_idx = idx
    
    pos_score = probabilities[pos_idx].item() if pos_idx is not None else 0
    neu_score = probabilities[neu_idx].item() if neu_idx is not None else 0
    neg_score = probabilities[neg_idx].item() if neg_idx is not None else 0
    
    sentiment_score = pos_score - neg_score
    
    return {
        'positive': pos_score,
        'neutral': neu_score,
        'negative': neg_score,
        'sentiment_score': sentiment_score
    }

# ----------------------
# Upbit API Integration
# ----------------------
@st.cache_data
def get_upbit_markets():
    """Fetches all KRW crypto markets from Upbit."""
    url = "https://api.upbit.com/v1/market/all"
    try:
        response = requests.get(url, params={'isDetails': 'false'})
        response.raise_for_status()
        markets = response.json()
        
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

# ----------------------
# Naver News API
# ----------------------
def get_naver_news_api(query, display=30, start=1, sort="date"):
    """Fetches news data from Naver News API."""
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

# ----------------------
# Fear & Greed Index
# ----------------------
@st.cache_data
def get_fear_greed_index():
    """Fetches Fear & Greed Index from alternative source."""
    url = "https://api.alternative.me/fng/?limit=1000"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = df['value'].astype(float)
        return df[['timestamp', 'value']].rename(columns={'timestamp': 'date', 'value': 'fng_index'})
    except Exception as e:
        st.error(f"❌ Fear & Greed Index 데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# ----------------------
# Google Trends Data
# ----------------------
@st.cache_data
def get_google_trends(keyword, start_date, end_date):
    """Fetches Google Trends data for a keyword."""
    pytrends = TrendReq(hl='ko-KR', tz=360)
    df = pd.DataFrame()
    try:
        pytrends.build_payload([keyword], cat=0, timeframe=f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}', geo='', gprop='')
        df = pytrends.interest_over_time()
        if not df.empty:
            df = df.reset_index()
            df.rename(columns={'date': 'date', keyword: 'google_trends'}, inplace=True)
            return df[['date', 'google_trends']]
    except Exception as e:
        st.error(f"❌ 구글 트렌드 데이터를 가져오는 중 오류가 발생했습니다: {e}")
        st.info("429 오류는 잠시 후 다시 시도하면 해결될 수 있습니다.")
    return pd.DataFrame()

# ----------------------
# Technical Indicators Calculation
# ----------------------
def calculate_technical_indicators(df, ma_periods):
    """Calculates RSI, Bollinger Bands, MACD, and Volatility."""
    df = df.copy()
    
    # RSI (Relative Stength Index)
    df['change'] = df['close'].diff()
    df['gain'] = df['change'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = df['change'].apply(lambda x: abs(x) if x < 0 else 0)
    df['avg_gain'] = df['gain'].rolling(window=14).mean()
    df['avg_loss'] = df['loss'].rolling(window=14).mean()
    df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-8)
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

    # Volatility
    df['Daily_Return'] = df['close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Additional Moving Averages based on user input
    for period in ma_periods:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()

    # Target variable for next day's price prediction
    df['target'] = df['close'].shift(-1)
    
    return df

# ----------------------
# Main App Logic
# ----------------------
def main():
    # Crypto selection UI
    default_crypto = "비트코인"
    if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
        st.session_state.selected_company = default_crypto if default_crypto in company_names else company_names[0]

    company_name = st.selectbox(
        "✅ 분석할 암호화폐 선택",
        company_names,
        index=company_names.index(st.session_state.selected_company),
        key="selected_company"
    )
    stock_code = crypto_list.get(st.session_state.selected_company)

    # Main area for parameters
    st.markdown("---")
    st.subheader("⚙️ 모델 파라미터 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("📊 가격 데이터 기간", ["100일", "300일", "500일"], index=1)
        news_period = st.selectbox("📰 감성 분석 기간", ["1일", "3일", "5일", "7일"], index=1)
    with col2:
        lstm_epochs = st.slider("📈 LSTM 에포크 수", min_value=10, max_value=100, value=30, step=5)
        timesteps = st.slider("⏳ LSTM 시퀀스 길이", min_value=5, max_value=30, value=15, step=1)

    st.markdown("---")
    st.subheader("⚙️ 피처 엔지니어링 설정")

    col3, col4 = st.columns(2)
    with col3:
        lag_period = st.slider("⏪ 지연 피처(Lag Features) 기간", min_value=1, max_value=10, value=3, step=1, help="과거 N일 전의 데이터를 새로운 피처로 추가합니다.")
    with col4:
        ma_periods = st.multiselect(
            "🧮 추가 이동평균(MA) 기간",
            options=[5, 10, 50, 100],
            default=[5, 50],
            help="모델에 추가할 이동평균 기간을 선택하세요."
        )

    count_map = {"100일": 100, "300일": 300, "500일": 500}
    data_count = count_map.get(period, 300)
    news_days = int(news_period.replace('일', ''))

    st.markdown("---")

    if st.button("🚀 하이브리드 모델 분석 시작", use_container_width=True):
        st.subheader("1. 데이터 수집 및 전처리")
        
        with st.spinner("가격 데이터 로드 및 전처리 중..."):
            df_asset = pyupbit.get_ohlcv(stock_code, interval="day", count=data_count)
            if df_asset is None or df_asset.empty:
                st.error(f"❌ {stock_code} 데이터를 가져오지 못했습니다. 종목 코드나 Upbit 서버 상태를 확인해주세요.")
                st.stop()
            
            df_asset = df_asset.reset_index().rename(columns={'index': 'date', 'trade_price': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'})
            df_asset['date'] = pd.to_datetime(df_asset['date']).dt.date
            df_asset = calculate_technical_indicators(df_asset, ma_periods)
            
            # Add Lag Features
            for lag in range(1, lag_period + 1):
                df_asset[f'close_lag_{lag}'] = df_asset['close'].shift(lag)
                df_asset[f'volume_lag_{lag}'] = df_asset['volume'].shift(lag)

        with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
            end_date_news = datetime.now().date()
            start_date_news = end_date_news - timedelta(days=news_days)
            date_range = [start_date_news + timedelta(days=i) for i in range(news_days + 1)]
            
            all_news = pd.DataFrame()
            for date_day in date_range:
                date_str = date_day.strftime("%Y.%m.%d")
                df_part = get_naver_news_api(f"{company_name} {date_str}", display=50)
                df_part['Date'] = date_day
                all_news = pd.concat([all_news, df_part], ignore_index=True)
            
            all_news = all_news.dropna(subset=['Date'])
            sentiment_results = all_news['Title'].apply(lambda x: analyze_sentiment(x))
            sentiment_df = pd.json_normalize(sentiment_results)
            all_news = pd.concat([all_news, sentiment_df], axis=1)

            news_grouped = all_news.groupby('Date').agg(
                positive=('positive', 'mean'),
                neutral=('neutral', 'mean'),
                negative=('negative', 'mean')
            ).reset_index()

            news_grouped.rename(columns={'Date': 'date'}, inplace=True)
            df_final = pd.merge(df_asset, news_grouped, on='date', how='left')
            
        with st.spinner("Fear & Greed Index 및 구글 트렌드 데이터 로드 중..."):
            df_fng = get_fear_greed_index()
            df_fng['date'] = df_fng['date'].dt.date
            df_final = pd.merge(df_final, df_fng, on='date', how='left')

            keyword_to_search = company_name.lower()
            df_trends = get_google_trends(keyword_to_search, df_final['date'].min(), df_final['date'].max())
            
            if not df_trends.empty:
                df_trends['date'] = pd.to_datetime(df_trends['date']).dt.date
                df_final = pd.merge(df_final, df_trends, on='date', how='left')
            else:
                st.warning("⚠️ 구글 트렌드 데이터를 불러오는 데 실패했습니다. 해당 지표를 제외하고 분석을 진행합니다.")
            
        # 결측치를 이전 값으로 채우기
        df_final = df_final.ffill().bfill()
        
        st.success("✅ 모든 데이터 수집 및 병합 완료!")
        st.dataframe(df_final[['date', 'close', 'positive', 'fng_index', 'google_trends']].tail())
            
        st.markdown("---")
        st.subheader("2. 하이브리드 모델 학습")
        
        # LSTM 데이터 준비 및 학습
        features_lstm = ['close']
        X_lstm = df_final[features_lstm].values
        scaler_lstm = MinMaxScaler(feature_range=(0,1))
        scaled_lstm_data = scaler_lstm.fit_transform(X_lstm)
        
        generator_train = TimeseriesGenerator(
            scaled_lstm_data, scaled_lstm_data, length=timesteps, batch_size=1
        )
        
        model_lstm = Sequential([
            LSTM(50, activation='relu', input_shape=(timesteps, len(features_lstm))),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        
        with st.spinner("LSTM 모델 학습 중..."):
            model_lstm.fit(generator_train, epochs=lstm_epochs, verbose=0)
        
        lstm_predictions = []
        for i in range(len(scaled_lstm_data)):
            if i < timesteps:
                lstm_predictions.append(np.nan)
            else:
                input_seq = scaled_lstm_data[i - timesteps:i].reshape(1, timesteps, len(features_lstm))
                prediction_scaled = model_lstm.predict(input_seq, verbose=0)[0][0]
                lstm_predictions.append(prediction_scaled)

        lstm_predictions_original = scaler_lstm.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()
        df_final['lstm_pred'] = lstm_predictions_original
        
        # LightGBM 데이터 준비
        features_lgbm = [
            'open', 'high', 'low', 'close', 'volume', 'positive', 'neutral', 'negative', 
            'RSI', 'BB_upper', 'BB_lower', 'MACD', 'MACD_Signal', 'Volatility', 'lstm_pred',
            'fng_index', 'google_trends'
        ]
        
        # Add dynamic features
        for period in ma_periods:
            features_lgbm.append(f'MA{period}')
        for lag in range(1, lag_period + 1):
            features_lgbm.append(f'close_lag_{lag}')
            features_lgbm.append(f'volume_lag_{lag}')
        
        df_model = df_final.dropna(subset=features_lgbm + ['target'])
        
        if df_model.empty:
            st.error("❌ 데이터 전처리 후 학습에 사용할 데이터가 없습니다. 기간을 늘려주거나 피처 설정을 조정해주세요.")
            st.stop()
            
        X = df_model[features_lgbm]
        y = df_model['target']

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        with st.spinner("LightGBM 모델 학습 중..."):
            lgbm_model = lgb.LGBMRegressor(random_state=42)
            lgbm_model.fit(X_train, y_train, feature_name=features_lgbm)

        st.success("✅ 하이브리드 모델 학습 완료!")
        
        st.markdown("---")
        st.subheader("3. 최종 예측 및 모델 성능 평가")

        if X_test.empty:
            st.warning("테스트 데이터가 부족합니다. 기간을 늘려주세요.")
        else:
            final_predictions = lgbm_model.predict(X_test)
            
            # Performance metrics
            rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
            mape = mean_absolute_percentage_error(y_test, final_predictions) * 100
            r2 = r2_score(y_test, final_predictions)
            
            st.markdown(f"""
            - **RMSE (제곱근 평균 제곱 오차):** `{rmse:.2f}`
            - **MAPE (평균 절대 백분율 오차):** `{mape:.2f}%`
            - **R² (결정 계수):** `{r2:.2f}`
            """)

            # Visualization
            st.subheader("📈 가격 예측 차트")
            df_test = df_final.iloc[split_idx:].copy()
            df_test.loc[df_test.index, 'Predicted_Close'] = pd.Series(final_predictions, index=X_test.index)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_test['date'], df_test['close'], label='실제 가격 (Actual)', color='blue')
            ax.plot(df_test['date'], df_test['Predicted_Close'], label='예측 가격 (Predicted)', linestyle='--', color='red')
            
            ax.set_title(f"{company_name} 하이브리드 모델 가격 예측")
            ax.set_xlabel("날짜")
            ax.set_ylabel("종가")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("4. 추가 지표 시각화")
            
            # F&G Index Visualization
            fig_fng, ax_fng = plt.subplots(figsize=(12, 4))
            ax_fng.plot(df_final['date'], df_final['fng_index'], label='Fear & Greed Index', color='purple')
            ax_fng.set_title("Fear & Greed Index")
            ax_fng.set_xlabel("날짜")
            ax_fng.set_ylabel("지수 (0-100)")
            ax_fng.legend()
            ax_fng.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig_fng)
            
            # Google Trends Visualization
            fig_trends, ax_trends = plt.subplots(figsize=(12, 4))
            ax_trends.plot(df_final['date'], df_final['google_trends'], label='Google Trends', color='orange')
            ax_trends.set_title("Google 검색 트렌드")
            ax_trends.set_xlabel("날짜")
            ax_trends.set_ylabel("상대적 검색량")
            ax_trends.legend()
            ax_trends.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig_trends)
            

            # Feature Importance
            st.markdown("---")
            st.subheader("5. 모델 피처 중요도")
            feature_importance = pd.DataFrame({
                'Feature': lgbm_model.feature_name_,
                'Importance': lgbm_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.bar_chart(feature_importance.set_index('Feature'))
            
            st.info("💡 **모델 해석:** `RSI`, `MACD`, `볼린저밴드`와 같은 기술적 지표가 가격 예측에 가장 큰 영향을 미칩니다. 뉴스 감성, Fear & Greed Index, 그리고 구글 트렌드 값도 중요한 영향을 미치는 것을 확인할 수 있습니다.")
            
if __name__ == "__main__":
    main()
