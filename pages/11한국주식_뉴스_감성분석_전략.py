import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import FinanceDataReader as fdr

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide")

st.title("📰 뉴스 감성 분석 기반 주가 예측 모델")
st.markdown("뉴스 기사 감성 점수와 기술적 지표를 결합하여 다음 날 주가 수익률을 예측합니다.")

# --- 데이터 로드 함수 (FinanceDataReader) ---
@st.cache_data
def load_and_preprocess_data(code):
    """FinanceDataReader를 사용하여 데이터를 로드하고 전처리합니다."""
    try:
        # 예측에 필요한 충분한 과거 데이터를 확보하기 위해 3년치 데이터 로드
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        df = fdr.DataReader(code, start=start_date, end=end_date)
        
        if 'Close' not in df.columns or df['Close'].isnull().all():
            st.error(f"❌ '{code}' 종목에 대한 가격 데이터가 없습니다. 다른 종목을 선택해주세요.")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df.columns = df.columns.str.strip()
        df.drop_duplicates(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # 종가(Close)를 제외한 모든 컬럼의 결측값을 0으로 채움
        df.loc[:, df.columns != 'Close'] = df.loc[:, df.columns != 'Close'].fillna(0)
        
        st.success(f"✅ '{code}' 종목의 데이터를 성공적으로 로드했습니다.")
        return df
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# --- 뉴스 감성 분석 함수 (간소화된 예제) ---
def analyze_sentiment(text):
    """
    매우 간소화된 감성 분석 함수.
    실제 사용 시에는 더 복잡한 NLP 모델(KoNLPy, BERT 등)이 필요합니다.
    """
    positive_words = ['성장', '상승', '확대', '기대', '강화', '돌파', '수혜', '혁신', '긍정']
    negative_words = ['하락', '감소', '위험', '경고', '부담', '우려', '부정', '손실', '악재']
    
    score = 0
    text_lower = text.lower()
    
    for word in positive_words:
        if word in text_lower:
            score += 1
    
    for word in negative_words:
        if word in text_lower:
            score -= 1
            
    return score

def fetch_sentiment_scores(stock_name, dates):
    """
    특정 종목에 대한 가상의 뉴스 감성 데이터를 생성합니다.
    실제 뉴스 크롤링은 웹사이트의 정책, 기술적 제약 등으로 인해 복잡하며,
    이 코드는 예제 목적으로만 사용해야 합니다.
    """
    # 가상의 뉴스 데이터. 실제로는 웹 크롤링을 통해 수집해야 함
    dummy_news = {
        '삼성전자': {
            '2024-01-01': ['삼성전자, AI 반도체 시장 성장 기대감에 주가 상승', '반도체 업황 개선으로 삼성전자 실적 강화'],
            '2024-01-02': ['삼성전자, 신규 스마트폰 출시로 시장 점유율 확대', '갤럭시S 시리즈 혁신으로 긍정적 평가'],
            '2024-01-03': ['글로벌 경기 침체 우려로 삼성전자 주가 하락', '반도체 재고 부담에 따른 손실 위험 경고']
        },
        'SK하이닉스': {
            '2024-01-01': ['SK하이닉스, HBM 시장 선점으로 성장 기대', 'DDR5 수요 증가로 실적 강화'],
            '2024-01-02': ['SK하이닉스, 경쟁 심화로 인한 수익성 악화 우려', '낸드 가격 하락으로 실적 감소'],
            '2024-01-03': ['SK하이닉스, AI 반도체 수혜주로 부상', '기술 혁신을 통한 시장 돌파']
        },
        '카카오': {
            '2024-01-01': ['카카오, 플랫폼 규제 강화로 주가 하락', '자회사 상장 부담에 따른 부정적 전망'],
            '2024-01-02': ['카카오, 신규 서비스 출시로 성장 동력 확보', '카카오톡 비즈니스 기능 강화'],
            '2024-01-03': ['카카오, 미래 먹거리 투자 확대로 기대감 상승', '글로벌 시장 진출 전략 긍정적 평가']
        }
    }

    sentiment_df = pd.DataFrame(index=dates)
    sentiment_df['Sentiment_Score'] = 0.0

    if stock_name in dummy_news:
        for date_str, headlines in dummy_news[stock_name].items():
            news_date = pd.to_datetime(date_str)
            if news_date in sentiment_df.index:
                total_score = sum(analyze_sentiment(h) for h in headlines)
                sentiment_df.loc[news_date, 'Sentiment_Score'] = total_score
    
    # 누락된 날짜의 감성 점수는 이전 값으로 채움
    sentiment_df['Sentiment_Score'] = sentiment_df['Sentiment_Score'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return sentiment_df

# --- 머신러닝 (LightGBM) 관련 함수 ---
@st.cache_resource
def train_and_predict_lgbm(df_data, ml_features):
    """LightGBM 모델을 학습하고 다음 날 수익률을 예측합니다."""
    df_data['Next_Day_Return'] = df_data['Close'].pct_change().shift(-1) * 100
    df_ml = df_data[ml_features + ['Next_Day_Return']].dropna()

    if len(df_ml) < 20:
        st.warning(f"⚠️ 데이터가 부족하여 수익률 예측을 할 수 없습니다. (현재 {len(df_ml)}일)")
        return None, None, None, None, None, None

    X_ml = df_ml[ml_features].values
    y_ml = df_ml['Next_Day_Return'].values
    
    scaler_ml = MinMaxScaler()
    X_ml_scaled = scaler_ml.fit_transform(X_ml)
    
    # 마지막 날 데이터를 예측에 사용
    last_data_ml_scaled = X_ml_scaled[-1].reshape(1, -1)
    
    test_size_ml = max(1, int(0.2 * len(X_ml_scaled)))
    X_train_ml, X_test_ml = X_ml_scaled[:-test_size_ml], X_ml_scaled[-test_size_ml:]
    y_train_ml, y_test_ml = y_ml[:-test_size_ml], y_ml[-test_size_ml:]
    
    # 모델 학습
    lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=500,
                                   learning_rate=0.05, num_leaves=31, max_depth=-1,
                                   random_state=42, n_jobs=-1, verbose=-1)
    
    lgbm_model.fit(X_train_ml, y_train_ml,
                   eval_set=[(X_test_ml, y_test_ml)],
                   callbacks=[lgb.early_stopping(100, verbose=False)])

    y_pred_ml = lgbm_model.predict(X_test_ml)
    next_day_return_pred_ml = lgbm_model.predict(last_data_ml_scaled)[0]

    return lgbm_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml, df_ml

# --- Streamlit UI 시작 ---
# FinanceDataReader에서 상장 종목 리스트를 가져와 종목 코드를 매핑
@st.cache_data
def load_stock_codes():
    try:
        df_krx = fdr.StockListing('KRX')
        return df_krx.set_index('Code')['Name'].to_dict()
    except Exception as e:
        st.error(f"❌ 종목 리스트를 불러오는 중 오류가 발생했습니다: {e}")
        st.warning("일시적인 오류일 수 있습니다. 삼성전자, SK하이닉스 등 대표 종목만으로 진행합니다.")
        return {'005930': '삼성전자', '000660': 'SK하이닉스', '005380': '현대차', '035720': '카카오', '035420': '네이버'}

try:
    name_code_dict = {v: k for k, v in load_stock_codes().items()}
except Exception as e:
    st.error(f"종목 리스트를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()
    
selected_name = st.selectbox("🔮 **예측할 종목을 선택하세요**", sorted(name_code_dict.keys()))
selected_code = name_code_dict[selected_name]

if st.button("🚀 **예측 시작**"):
    st.session_state.df_stock = load_and_preprocess_data(selected_code)
    
    if st.session_state.df_stock.empty:
        st.stop()
        
    df_stock = st.session_state.df_stock.copy()
    
    # --- 기술적 지표 계산 (간소화) ---
    df_stock['Volatility'] = df_stock['Change'].rolling(window=20).std()
    
    # --- 뉴스 감성 분석 데이터 결합 ---
    # 실제 데이터의 인덱스를 기준으로 감성 점수를 가져옴
    sentiment_df = fetch_sentiment_scores(selected_name, df_stock.index)
    df_stock = df_stock.join(sentiment_df, how='left')
    df_stock['Sentiment_Score'] = df_stock['Sentiment_Score'].fillna(0) # 결측치는 0으로 채움
    
    # --- LightGBM 모델 예측 섹션 ---
    st.header("📊 LightGBM 모델: 단기 수익률 예측")
    
    # 예측에 사용할 특징 목록 (기술적 지표 + 감성 점수)
    ml_features = ['Close', 'Volume', 'Open', 'High', 'Low', 'Volatility', 'Sentiment_Score']
    ml_features = [col for col in ml_features if col in df_stock.columns]
        
    lgbm_model, y_pred_ml, next_day_return_pred_ml, y_test_ml, X_test_ml, df_ml = train_and_predict_lgbm(df_stock, ml_features)

    if lgbm_model is not None:
        st.subheader("📊 모델 성능 평가")
        st.write(f"**평균 제곱 오차 (MSE)**: {mean_squared_error(y_test_ml, y_pred_ml):.2f}")
        st.write(f"**결정 계수 (R² Score)**: {r2_score(y_test_ml, y_pred_ml):.2f}")
        st.write(f"테스트 데이터의 **평균 실제 수익률**: {np.mean(y_test_ml):.2f}%")
        st.write(f"테스트 데이터의 **평균 예측 수익률**: {np.mean(y_pred_ml):.2f}%")

        st.subheader("📈 예측 결과 시각화")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test_ml, label='실제 수익률', color='blue', marker='o', linestyle='None', alpha=0.6)
        ax.plot(y_pred_ml, label='예측 수익률', color='red', marker='x', linestyle='None', alpha=0.6)
        ax.set_title(f"{selected_name} ({selected_code}) LightGBM 예측 vs. 실제 수익률")
        ax.set_xlabel("데이터 포인트 인덱스")
        ax.set_ylabel("수익률(%)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("💡 다음 날 주가 예측")
        st.write(f"다음 영업일의 주가 수익률은 **{next_day_return_pred_ml:.2f}%**로 예측됩니다.")
        
        # 긍부정 판단
        if next_day_return_pred_ml > 0:
            st.success("예측 수익률이 긍정적입니다. 매수 신호로 고려해볼 수 있습니다.")
        else:
            st.warning("예측 수익률이 부정적입니다. 매도 또는 관망 신호로 고려해볼 수 있습니다.")
