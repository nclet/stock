import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.linear_model import LinearRegression # ⚠️ 시각화 비교를 위해 LinearRegression을 남겨둡니다.
from sklearn.metrics import mean_squared_error, r2_score
import urllib.parse
from json.decoder import JSONDecodeError
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import time

# ------------------------
# ✨ 상수 및 페이지 설정
# ------------------------
st.set_page_config(page_title="암호화폐 뉴스 감성 분석 전략", layout="wide")
st.title("🪙 암호화폐 뉴스 감성 분석 전략 (Plotly UI 강화)")

st.markdown("""
네이버 뉴스와 Fear & Greed Index를 크롤링하여 시계열 특징과 결합,
주요 암호화폐의 다음 날 **수익률**을 분석하고 예측합니다.
""")

# ------------------------
# 0. 피처 엔지니어링 함수 (시계열 특성 추가)
# ------------------------
def create_features(df_merge):
    """
    암호화폐 가격, 감성 점수 및 Fear & Greed Index에 시계열 지연(Lag) 피처를 추가합니다.
    """
    df = df_merge.copy()
    
    # 1. 타겟 변수: 다음 날의 수익률 (%)
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1) * 100

    # 2. 시계열 지연(Lag) 피처 추가
    lags = [1, 3, 5]
    
    # 2-1. 감성 점수 지연 피처
    for lag in lags:
        df[f'Sentiment_Lag_{lag}'] = df['Sentiment_Score'].shift(lag)
        
    # 2-2. 종가 수익률 지연 피처
    df['Daily_Return'] = df['Close'].pct_change() * 100
    for lag in lags:
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

    # 2-3. Fear & Greed Index 지연 피처
    for lag in lags:
        df[f'FGI_Lag_{lag}'] = df['Index'].shift(lag)

    # 3. 기술적/보조 지표 
    df['Momentum'] = df['Close'].diff()
    df['Momentum_Lag_1'] = df['Momentum'].shift(1)

    df = df.dropna()
    
    base_features = ['Close', 'Sentiment_Score', 'Index', 'Momentum', 'Momentum_Lag_1']
    lag_features = [col for col in df.columns if 'Lag' in col]
    
    features = [f for f in base_features + lag_features if f in df.columns]
    
    return df, features

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
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, device_map='auto')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        st.sidebar.success(f"✅ 감성 분석 모델 로드 완료 (장치: {device})")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ 감성 분석 모델 '{model_name}' 로드 중 오류 발생: {e}")
        st.info("Hugging Face 토큰 설정 또는 라이브러리 버전을 확인해주세요.")
        st.stop()
        return None, None, None

tokenizer, sentiment_model, device = load_sentiment_model()

def analyze_sentiment(text):
    """Calculates sentiment score for the given text."""
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
# ✨ Upbit 종목 목록 및 Fear & Greed Index 로드
# ------------------------
@st.cache_data(show_spinner="⏳ Upbit 종목 리스트 로드 중...")
def get_upbit_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        response = requests.get(url, params={'isDetails': 'false'})
        response.raise_for_status()
        markets = response.json()
        krw_markets = {m['korean_name']: m['market'] for m in markets if m['market'].startswith('KRW-')}
        return krw_markets
    except Exception as e:
        st.error(f"❌ Upbit API 오류: {e}")
        return {}

@st.cache_data(show_spinner="⏳ Fear & Greed Index 로드 중...")
def get_fear_greed_index(limit=365):
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.rename(columns={"value": "Index", "timestamp": "Date"})
        df["Date"] = df["Date"].dt.date 
        return df[["Date", "Index"]].sort_values("Date")
    except Exception as e:
        st.warning(f"⚠️ Fear & Greed Index 로드 오류: {e}. 예측에는 영향이 없을 수 있습니다.")
        return pd.DataFrame()


crypto_list = get_upbit_markets()
company_names = list(crypto_list.keys())
fg_df = get_fear_greed_index(limit=365)


# ------------------------
# ✨ UI 입력 요소
# ------------------------
col_select, col_date_start, col_date_end, col_max_news = st.columns([2, 1, 1, 1])

with col_select:
    default_crypto = "비트코인"
    if "selected_company" not in st.session_state or st.session_state.selected_company not in company_names:
        st.session_state.selected_company = default_crypto if default_crypto in company_names else company_names[0]

    company_name = st.selectbox(
        "✅ 분석할 암호화폐 선택",
        company_names,
        index=company_names.index(st.session_state.selected_company),
        key="selected_company"
    )

stock_code = crypto_list.get(company_name)

with col_date_start:
    start_date = st.date_input("뉴스 검색 시작일", datetime.now() - timedelta(days=90))
with col_date_end:
    end_date = st.date_input("뉴스 검색 종료일", datetime.now())
with col_max_news:
    max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=200, value=100, step=10)


# ------------------------
# ✨ 네이버 뉴스 API 함수
# ------------------------
def get_naver_news_api(query, display=30, start=1, sort="date"):
    """Fetches data from Naver News Search API."""
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError as e:
        st.error(f"❌ 네이버 API 키({e})가 Streamlit Secrets에 설정되어 있지 않습니다.")
        st.info("Secrets 메뉴에서 naver.client_id와 naver.client_secret을 설정해야 합니다.")
        return pd.DataFrame()

    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
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
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 실패: {e}")
        return pd.DataFrame()
    except JSONDecodeError as e:
        st.error(f"API 응답 파싱 실패: {e}")
        return pd.DataFrame()

# ------------------------
# ✨ Upbit 캔들 데이터 로드
# ------------------------
@st.cache_data(show_spinner="⏳ 암호화폐 캔들 데이터를 로드 중입니다...")
def get_upbit_candles(market, start_date, end_date):
    """
    Upbit API에서 일별 캔들 데이터를 가져옵니다. 
    Lag Feature 생성을 위해 시작일보다 30일 정도 더 이전부터 데이터를 로드합니다.
    """
    base_url = "https://api.upbit.com/v1/candles/days"
    df_list, current_date, requests_count = [], end_date, 0
    max_requests = 15 

    load_start_date = start_date - timedelta(days=30)
    
    while current_date >= load_start_date and requests_count < max_requests:
        params = {
            'market': market,
            'to': (current_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
            'count': 200
        }
        
        time.sleep(0.05) 
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data: break
            
            temp_df = pd.DataFrame(data)
            temp_df['Date'] = pd.to_datetime(temp_df['candle_date_time_kst']).dt.date
            
            temp_df = temp_df.rename(columns={'trade_price': 'Close', 
                                            'opening_price': 'Open', 
                                            'high_price': 'High', 
                                            'low_price': 'Low', 
                                            'candle_acc_trade_volume': 'Volume'}) 
            
            df_list.append(temp_df)
            current_date = temp_df['Date'].min() - timedelta(days=1)
            requests_count += 1
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Upbit API 요청 실패: {e}")
            return pd.DataFrame()
        except JSONDecodeError:
             st.error("❌ Upbit API 응답 파싱 실패.")
             return pd.DataFrame()
        
    if not df_list: return pd.DataFrame()
    df_final = pd.concat(df_list, ignore_index=True)
    df_final = df_final.sort_values('Date').drop_duplicates('Date')
    df_final.set_index('Date', inplace=True)
    
    return df_final[df_final.index >= load_start_date][['Close', 'Open', 'High', 'Low', 'Volume']]


# ------------------------
# ✨ 실행 로직
# ------------------------
st.markdown("---")
if st.button("🚀 크롤링 및 분석 시작", type="primary", use_container_width=True):
    
    # 1. 데이터 로드 및 전처리
    
    # 1-1. 뉴스 크롤링 및 감성 분석
    with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
        all_news = pd.DataFrame()
        for start_idx in range(1, max_news + 1, 100):
            count = min(100, max_news - start_idx + 1)
            df_part = get_naver_news_api(company_name, display=count, start=start_idx)
            all_news = pd.concat([all_news, df_part], ignore_index=True)
            if len(df_part) < count:
                break
            time.sleep(0.5) 

        load_start_date = start_date - timedelta(days=30) 
        all_news = all_news.dropna(subset=['Date'])
        filtered_news = all_news[(all_news['Date'] >= load_start_date) & (all_news['Date'] <= end_date)].copy()

        if filtered_news.empty:
            st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 기간이나 암호화폐명을 확인해주세요.")
            st.stop()
        
        filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
        st.success("✅ 뉴스 크롤링 및 감성 분석 완료!")
        
    # 1-2. 가격 데이터 로드
    df_asset = get_upbit_candles(stock_code, start_date, end_date)
    
    if df_asset.empty:
        st.error("❌ 암호화폐 캔들 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
        st.stop()
    else:
        st.success("✅ 암호화폐 캔들 데이터 로드 완료 (Upbit)!")
        
        # 1-3. 데이터 병합 (가격 + 감성 + Fear&Greed Index)
        filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
        df_asset.reset_index(inplace=True)
        
        df_merge = pd.merge(df_asset, filtered_news_grouped, on='Date', how='left')
        
        # Fear & Greed Index 병합
        fg_df_filtered = fg_df[(fg_df['Date'] >= load_start_date) & (fg_df['Date'] <= end_date)].copy()
        df_merge = pd.merge(df_merge, fg_df_filtered, on='Date', how='left')
        
        df_merge = df_merge.set_index('Date')
        
        df_merge['Sentiment_Score'] = df_merge['Sentiment_Score'].fillna(method='ffill').fillna(0)
        df_merge['Index'] = df_merge['Index'].fillna(method='ffill').fillna(50) 

        # 2. 피처 엔지니어링 및 데이터 준비
        df_ml, features = create_features(df_merge)
        df_ml_filtered = df_ml[(df_ml.index >= start_date) & (df_ml.index <= end_date)].copy()


        if len(df_ml_filtered) <= 50:
            st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 최소 50개 이상의 데이터가 필요합니다. 뉴스 검색 기간을 늘리거나 다른 코인을 선택해보세요.")
            st.stop()

        X = df_ml_filtered[features].values
        y = df_ml_filtered['Next_Day_Return'].values
        
        # --- 추가된 시각화를 위한 2단계 선형 회귀 모델 훈련 ---
        # Plotly 비교 차트를 위한 단순 선형 모델 (실제 예측에는 LightGBM 사용)
        X_simple = df_ml_filtered[['Sentiment_Score', 'Momentum', 'Index']].values
        y_close = df_ml_filtered['Close'].values
        
        # 선형 회귀 모델 훈련 (종가 예측)
        if len(X_simple) > 5:
            model_lr = LinearRegression().fit(X_simple, y_close)
            df_ml_filtered['Predicted_Close_LR'] = model_lr.predict(X_simple)
        else:
            df_ml_filtered['Predicted_Close_LR'] = df_ml_filtered['Close']
        # ----------------------------------------------------
            
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        test_size = max(1, int(0.2 * len(X_scaled)))
        X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # 3. 모델 훈련 (LightGBM)
        LGBM_TUNED_PARAMS = {
            'objective': 'regression', 'metric': 'rmse',
            'n_estimators': 700, 'learning_rate': 0.01, 
            'num_leaves': 21, 'max_depth': 7,
            'colsample_bytree': 0.8, 'subsample': 0.8,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }
        
        st.info("모델 훈련 중... (LGBM 파라미터 튜닝 및 시계열 특성 반영)")
        lgbm_model = lgb.LGBMRegressor(**LGBM_TUNED_PARAMS)
        
        lgbm_model.fit(X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])

        # 잔차 기반 신뢰구간 계산 및 예측 수행
        y_train_pred = lgbm_model.predict(X_train)
        residuals = y_train - y_train_pred
        residual_std = residuals.std()
        CI_FACTOR = 1.645 * residual_std 
        y_test_pred = lgbm_model.predict(X_test)
        
        # 4. 모델 성능 평가 및 다음 날 예측
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        last_data = df_ml[features].iloc[-1].values.reshape(1, -1)
        last_data_scaled = scaler.transform(last_data)
        next_day_return_pred = lgbm_model.predict(last_data_scaled)[0]
        
        low_ci = next_day_return_pred - CI_FACTOR
        high_ci = next_day_return_pred + CI_FACTOR

        st.markdown("---")
        st.subheader(f"✨ 최종 분석 및 예측 결과: {company_name} ({stock_code})")
        
        # --- A. 예측 결과 카드형 출력 ---
        col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

        def format_pred_value(value):
            return f"{value:+.2f}%"

        with col_pred1:
            st.metric(label="📈 다음 날 예측 수익률", 
                      value=format_pred_value(next_day_return_pred), 
                      delta=f"CI 범위: {low_ci:+.2f}% ~ {high_ci:+.2f}%")

        with col_pred2:
            st.metric(label="✅ 예측 신뢰도 (R²)", 
                      value=f"{r2:.2f}", 
                      help=f"MSE: {mse:.4f}. 1에 가까울수록 모델의 적합도가 높음.")
            
        with col_pred3:
            sentiment_summary = df_ml_filtered['Sentiment_Score'].iloc[-30:].mean()
            sentiment_trend = "긍정적 🟢" if sentiment_summary > 0.1 else ("부정적 🔴" if sentiment_summary < -0.1 else "중립 🟡")
            st.metric(label="📰 최근 30일 감성 점수 평균", 
                      value=f"{sentiment_summary:+.2f}", 
                      delta=sentiment_trend)

        with col_pred4:
            action = "매수 신호" if next_day_return_pred > 0.5 and low_ci > 0 else ("매도/관망" if next_day_return_pred < -0.5 else "관망")
            st.markdown(f"""
            <div style='
                padding: 10px; border-radius: 5px; text-align: center; 
                background-color: {"#D4EDDA" if action == "매수 신호" else ("#F8D7DA" if action == "매도/관망" else "#FFF3CD")}; 
                color: {"#155724" if action == "매수 신호" else ("#721C24" if action == "매도/관망" else "#856404")}; 
                font-weight: bold; margin-top: 15px;'>
                최종 액션: {action}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # --- B. Plotly: 가격 및 감성 점수 추이 ---
        st.subheader("📊 암호화폐 가격과 일일 감성/공포탐욕 지수 추이")

        df_plot = df_merge.copy()
        df_plot = df_plot[(df_plot.index >= start_date) & (df_plot.index <= end_date)]

        fig_price = go.Figure()

        # 1. 가격 (봉차트)
        fig_price.add_trace(go.Candlestick(x=df_plot.index,
                                           open=df_plot['Open'],
                                           high=df_plot['High'],
                                           low=df_plot['Low'],
                                           close=df_plot['Close'],
                                           name='가격 (OHLC)',
                                           yaxis='y1'))
        
        # 2. 감성 점수 (보조축 - y2)
        sentiment_color = df_plot['Sentiment_Score'].apply(lambda x: 'red' if x < 0 else 'green')
        fig_price.add_trace(go.Bar(x=df_plot.index, 
                                   y=df_plot['Sentiment_Score'], 
                                   name='감성 점수 평균', 
                                   yaxis='y2',
                                   marker_color=sentiment_color,
                                   opacity=0.5))
        
        # 3. Fear & Greed Index (보조축 - y3)
        fig_price.add_trace(go.Scatter(x=df_plot.index,
                                      y=df_plot['Index'],
                                      name='공포/탐욕 지수',
                                      yaxis='y3',
                                      mode='lines',
                                      line=dict(color='orange', width=1.5)))

        fig_price.update_layout(
            title=f"{company_name} 가격, 감성 점수 및 공포/탐욕 지수 비교",
            xaxis_title="날짜",
            # 가격 축 (y1)
            yaxis=dict(
                title=dict(text='가격 (종가)', font=dict(color="#1f77b4")),
                tickfont=dict(color="#1f77b4"), 
                domain=[0.35, 1] # 차트 상단 65%
            ),
            # 감성 점수 축 (y2)
            yaxis2=dict(
                title=dict(text='감성 점수', font=dict(color="#d62728")),
                tickfont=dict(color="#d62728"), 
                overlaying='y', 
                side='right', 
                domain=[0.1, 0.3] # 차트 중앙 하단 20%
            ),
            # Fear & Greed Index 축 (y3)
            yaxis3=dict(
                title=dict(text='F&G Index', font=dict(color='orange')),
                tickfont=dict(color='orange'),
                overlaying='y',
                side='left',
                position=0.05, # 왼쪽 축에 배치 
                domain=[0.05, 0.25], # 차트 최하단 20%
                showgrid=False # 그리드 숨김
            ),
            hovermode="x unified",
            height=650,
            legend=dict(x=0, y=1.05, orientation="h")
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
        
        
        st.markdown("---")
        
        # --- C. Plotly: 예측 vs. 실제 수익률 시각화 (LightGBM) ---
        st.subheader("📈 LightGBM 예측 vs. 실제 수익률 (90% 신뢰구간)")
        
        y_test_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred,
            'Low_CI': y_test_pred - CI_FACTOR,
            'High_CI': y_test_pred + CI_FACTOR
        }, index=df_ml_filtered.index[-test_size:])

        fig_pred = go.Figure()

        # 신뢰구간 (음영)
        fig_pred.add_trace(go.Scatter(
            x=y_test_df.index, y=y_test_df['High_CI'], 
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_pred.add_trace(go.Scatter(
            x=y_test_df.index, y=y_test_df['Low_CI'], 
            fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', 
            mode='lines', line=dict(width=0), name='90% 신뢰구간'
        ))
        
        # 실제 수익률 
        fig_pred.add_trace(go.Scatter(
            x=y_test_df.index, y=y_test_df['Actual'], 
            mode='markers', name='실제 수익률', marker=dict(color='blue', size=5, opacity=0.8)
        ))
        
        # 예측 수익률 
        fig_pred.add_trace(go.Scatter(
            x=y_test_df.index, y=y_test_df['Predicted'], 
            mode='lines', name='예측 수익률 (Median)', line=dict(color='red', width=2)
        ))

        fig_pred.update_layout(
            title=f"테스트 기간의 LightGBM 수익률 예측 결과",
            xaxis_title="날짜",
            yaxis_title="수익률(%)",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_pred, use_container_width=True)


        # --- D. Plotly: 감성 + 모멘텀 + 공포탐욕 (선형회귀 종가 예측) 시각화 ---
        st.markdown("---")
        st.subheader("🔍 감성 + 모멘텀 + 공포탐욕 기반 종가 예측 (단순 선형 회귀 비교)")

        fig_lr = go.Figure()
        
        # 실제 종가
        fig_lr.add_trace(go.Scatter(
            x=df_ml_filtered.index, y=df_ml_filtered['Close'], 
            mode='lines', name='실제 종가', line=dict(color='blue', width=2)
        ))
        
        # 예측 종가 (Linear Regression)
        fig_lr.add_trace(go.Scatter(
            x=df_ml_filtered.index, y=df_ml_filtered['Predicted_Close_LR'], 
            mode='lines', name='예측 종가 (LR)', line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_lr.update_layout(
            title="감성, 모멘텀, FGI를 피처로 사용한 종가 예측 비교",
            xaxis_title="날짜",
            yaxis_title="가격",
            hovermode="x unified",
            height=500,
            legend=dict(x=0, y=1.05, orientation="h")
        )

        st.plotly_chart(fig_lr, use_container_width=True)


    st.markdown("---")
    st.write("👉 **감성점수 계산 방식**: Hugging Face 모델에서 추출한 '긍정' 점수에서 '부정' 점수를 뺀 값이며, $\pm 1.0$ 범위를 가집니다.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from sklearn.linear_model import LinearRegression
# import urllib.parse
# from json.decoder import JSONDecodeError

# # ------------------------
# # ✨ 페이지 설정
# # ------------------------
# st.set_page_config(page_title="암호화폐 감성 분석 전략", layout="wide")
# st.title("암호화폐 뉴스 감성 분석 전략")

# st.markdown("""
# 네이버 뉴스를 크롤링하여 모멘텀 데이터를 결합,
# 주요 암호화폐의 가격을 분석하고 예측하는 전략입니다.
# """)

# # ------------------------
# # ✨ 감성 분석 모델 로드
# # ------------------------
# @st.cache_resource
# def load_sentiment_model():
#     """Hugging Face에서 한국어 감성 분석 모델을 로드합니다."""
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

# tokenizer, sentiment_model, device = load_sentiment_model()

# def analyze_sentiment(text):
#     """주어진 텍스트의 감성 점수를 계산합니다."""
#     if not text:
#         return 0.0
    
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = sentiment_model(**inputs)
    
#     probabilities = torch.softmax(outputs.logits, dim=1)[0]

#     neg_idx = None
#     neu_idx = None
#     pos_idx = None
#     for idx, label in sentiment_model.config.id2label.items():
#         if 'negative' in label.lower() or '부정' in label:
#             neg_idx = idx
#         elif 'neutral' in label.lower() or '중립' in label:
#             neu_idx = idx
#         elif 'positive' in label.lower() or '긍정' in label:
#             pos_idx = idx
    
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
# # ✨ 공포-탐욕 지수 데이터 로드
# # ------------------------
# @st.cache_data
# def get_fear_greed_index(limit=365):
#     """Alternative.me API에서 공포-탐욕 지수를 가져옵니다 (최대 365일)."""
#     url = f"https://api.alternative.me/fng/?limit={limit}"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json().get("data", [])
#         df = pd.DataFrame(data)
#         df["value"] = df["value"].astype(float)
#         df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
#         df = df.rename(columns={"value": "Index", "timestamp": "Date"})
#         return df[["Date", "Index"]].sort_values("Date")
#     except Exception as e:
#         st.error(f"❌ Fear & Greed Index 데이터를 가져오지 못했습니다: {e}")
#         return pd.DataFrame()

# # 실제 데이터 불러오기
# fg_df = get_fear_greed_index(limit=365)


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
# # ✨ Upbit API 함수
# # ------------------------
# @st.cache_data
# def get_upbit_candles(market, start_date, end_date):
#     """
#     Upbit API를 통해 일별 캔들 데이터를 가져와 DataFrame으로 반환합니다.
#     """
#     base_url = "https://api.upbit.com/v1/candles/days"
#     df_list = []
#     current_date = end_date
#     max_requests = 10 
#     requests_count = 0
    
#     while current_date >= start_date and requests_count < max_requests:
#         params = {
#             'market': market,
#             'to': (current_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
#             'count': 200
#         }
        
#         response = requests.get(base_url, params=params)
        
#         if response.status_code == 200:
#             data = response.json()
#             if not data:
#                 break
                
#             temp_df = pd.DataFrame(data)
#             temp_df['Date'] = pd.to_datetime(temp_df['candle_date_time_kst']).dt.date
#             temp_df = temp_df.rename(columns={'trade_price': 'Close'})
#             df_list.append(temp_df)
            
#             current_date = temp_df['Date'].min() - timedelta(days=1)
#             requests_count += 1
            
#         else:
#             st.error(f"Upbit API 요청 실패: 상태 코드 {response.status_code}")
#             return pd.DataFrame()

#     if not df_list:
#         return pd.DataFrame()

#     df_final = pd.concat(df_list, ignore_index=True)
#     df_final = df_final.sort_values('Date').drop_duplicates(subset='Date', keep='first')
#     df_final = df_final[(df_final['Date'] >= start_date) & (df_final['Date'] <= end_date)].reset_index(drop=True)
    
#     return df_final[['Date', 'Close']]

# # ------------------------
# # ✨ 실행 버튼
# # ------------------------
# max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=100, value=30, step=10)

# if st.button("🚀 크롤링 및 분석 시작"):
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
#     else:
#         filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)
#         st.success("✅ 뉴스 감성 분석 완료!")
#         st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False))

#         st.info(f"📈 {company_name} 가격 데이터를 Upbit API로 로드 중입니다...")
#         df_asset = get_upbit_candles(stock_code, start_date, end_date)
            
#         if df_asset.empty:
#             st.error("❌ 암호화폐 가격 데이터를 가져오지 못했습니다. 종목 코드나 날짜 범위를 확인해주세요.")
#             st.stop()
#         else:
#             df_asset['Date'] = pd.to_datetime(df_asset['Date'])
#             st.success("✅ 암호화폐 가격 데이터 로드 완료 (Upbit API)!")

#             df_asset['Momentum'] = df_asset['Close'].diff()
#             df_asset['Date'] = pd.to_datetime(df_asset['Date'])
#             filtered_news['Date'] = pd.to_datetime(filtered_news['Date'])
            
#             filtered_news_grouped = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
#             df_merge = pd.merge(df_asset, filtered_news_grouped, on='Date', how='left').fillna(0)
            
#             X = df_merge[['Sentiment_Score', 'Momentum']].fillna(0).values
#             y = df_merge['Close'].values

#             if len(X) > 5:
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 y_pred = model.predict(X)
#                 df_merge['Predicted_Close'] = y_pred

#                 st.subheader("📊 예측 결과")
#                 fig, ax = plt.subplots(figsize=(12, 6))
#                 ax.plot(df_merge['Date'], df_merge['Close'], label='실제 가격 (Actual)', color='blue')
#                 ax.plot(df_merge['Date'], df_merge['Predicted_Close'], label='예측 가격 (Predicted)', linestyle='--', color='red')
#                 ax.set_title(f"{company_name} 가격 예측 (뉴스 감성 + 모멘텀)")
#                 ax.set_xlabel("날짜")
#                 ax.set_ylabel("종가")
#                 ax.legend()
#                 ax.grid(True)
#                 plt.xticks(rotation=45)
#                 st.pyplot(fig)

#                 st.subheader("📈 회귀 모델 계수")
#                 st.metric("감성 점수 회귀계수", f"{model.coef_[0]:.2f}")
#                 st.metric("모멘텀 회귀계수", f"{model.coef_[1]:.2f}")
#             else:
#                 st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")

#         st.markdown("---")
#         st.write("👉 감성점수는 부정 뉴스에 -1, 긍정 뉴스에 1 점수를 대입합니다. 즉, -1(부정)~1(긍정)으로 점수가 계산됩니다.")


# # ------------------------
# # ✨ 예측 모델 (감성 + 모멘텀 + 공포탐욕 지수)
# # ------------------------
# if not fg_df.empty:
#     # 날짜 기준으로 병합
#     df_merge = pd.merge(df_merge, fg_df, on="Date", how="left")
#     df_merge["Index"] = df_merge["Index"].fillna(method="ffill").fillna(method="bfill")
# else:
#     df_merge["Index"] = 50  # 데이터 없으면 중립값(50)으로 채움

# X = df_merge[["Sentiment_Score", "Momentum", "Index"]].fillna(0).values
# y = df_merge["Close"].values

# if len(X) > 5:
#     model = LinearRegression()
#     model.fit(X, y)
#     y_pred = model.predict(X)
#     df_merge["Predicted_Close"] = y_pred

#     st.subheader("📊 예측 결과 (감성 + 모멘텀 + 공포탐욕)")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(df_merge["Date"], df_merge["Close"], label="실제 가격 (Actual)", color="blue")
#     ax.plot(df_merge["Date"], df_merge["Predicted_Close"], label="예측 가격 (Predicted)", linestyle="--", color="red")
#     ax.set_title(f"{company_name} 가격 예측 (뉴스 감성 + 모멘텀 + 공포탐욕 지수)")
#     ax.set_xlabel("날짜")
#     ax.set_ylabel("종가")
#     ax.legend()
#     ax.grid(True)
#     plt.xticks(rotation=45)
#     st.pyplot(fig)

#     st.subheader("📈 회귀 모델 계수")
#     st.metric("감성 점수 회귀계수", f"{model.coef_[0]:.2f}")
#     st.metric("모멘텀 회귀계수", f"{model.coef_[1]:.2f}")
#     st.metric("공포탐욕 지수 회귀계수", f"{model.coef_[2]:.2f}")
# else:
#     st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")
