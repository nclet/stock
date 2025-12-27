import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

import FinanceDataReader as fdr
import pyupbit

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ======================================================
# 1. 캔들 패턴 탐지
# ======================================================
def find_candle_patterns(df):
    df = df.copy()
    df['body'] = abs(df['Close'] - df['Open'])
    df['range'] = df['High'] - df['Low']
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    df['is_hammer'] = (df['lower_shadow'] > 2 * df['body']) & (df['upper_shadow'] < df['body'])
    df['is_doji'] = (df['body'] / df['range'] < 0.05)

    df['is_bullish_engulfing'] = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1))
    )

    return df

PATTERN_COLS = [
    'is_hammer',
    'is_doji',
    'is_bullish_engulfing'
]

# ======================================================
# 2. 수치적 특징
# ======================================================
def add_numeric_features(df):
    df = df.copy()
    df['body_ratio'] = df['body'] / df['range']
    df['upper_ratio'] = df['upper_shadow'] / df['range']
    df['lower_ratio'] = df['lower_shadow'] / df['range']
    df['volatility'] = df['range'] / df['Close']
    return df

# ======================================================
# 3. 추세 & 맥락
# ======================================================
def add_context_features(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['trend'] = (df['MA20'] > df['MA60']).astype(int)

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# ======================================================
# 4. 패턴 강도
# ======================================================
def add_pattern_strength(df):
    df = df.copy()
    df['pattern_strength'] = (
        df['body_ratio'] +
        df['lower_ratio'] -
        df['upper_ratio']
    )
    return df

# ======================================================
# 5. 타겟 변수
# ======================================================
def add_targets(df):
    df = df.copy()
    df['ret_3'] = df['Close'].shift(-3) / df['Close'] - 1
    df['ret_5'] = df['Close'].shift(-5) / df['Close'] - 1
    df['ret_10'] = df['Close'].shift(-10) / df['Close'] - 1
    return df

# ======================================================
# 6. LightGBM 학습
# ======================================================
def train_lgbm(df, target):
    feature_cols = (
        PATTERN_COLS +
        ['body_ratio', 'upper_ratio', 'lower_ratio',
         'volatility', 'trend', 'RSI', 'pattern_strength']
    )

    df = df.dropna()
    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return model, rmse, X, y

# ======================================================
# 7. 패턴 승률 리포트
# ======================================================
def pattern_report(df):
    rows = []
    for p in PATTERN_COLS:
        trades = df[df[p]]
        if len(trades) > 10:
            winrate = (trades['ret_5'] > 0).mean() * 100
            rows.append({
                'Pattern': p,
                'Count': len(trades),
                'WinRate(%)': winrate
            })
    return pd.DataFrame(rows).sort_values('WinRate(%)', ascending=False)

# ======================================================
# 8. Streamlit UI
# ======================================================
st.set_page_config(layout="wide")
st.title("📊 AI 캔들 패턴 수익률 예측기 (LightGBM)")

market = st.radio("시장 선택", ["KRX 주식", "미국 주식", "코인(Upbit)"], horizontal=True)
ticker = st.text_input("티커", "AAPL")

target_map = {
    "3봉 수익률": "ret_3",
    "5봉 수익률": "ret_5",
    "10봉 수익률": "ret_10"
}
target_label = st.selectbox("예측 수익률 기준", list(target_map.keys()))
target_col = target_map[target_label]

if st.button("🚀 분석 실행"):

    with st.spinner("데이터 로딩 중..."):
        if market == "코인(Upbit)":
            df = pyupbit.get_ohlcv(ticker, count=500)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Value']
        else:
            df = fdr.DataReader(ticker)

    df = find_candle_patterns(df)
    df = add_numeric_features(df)
    df = add_context_features(df)
    df = add_pattern_strength(df)
    df = add_targets(df)

    model, rmse, X, y = train_lgbm(df, target_col)

    st.success(f"모델 학습 완료 | RMSE: {rmse:.4f}")

    # ======================================================
    # 예측
    # ======================================================
    latest_X = X.iloc[[-1]]
    pred = model.predict(latest_X)[0]

    st.metric("📈 예상 수익률", f"{pred*100:.2f}%")

    # ======================================================
    # 수익률 분포
    # ======================================================
    st.subheader("📊 수익률 분포")
    fig, ax = plt.subplots()
    ax.hist(y * 100, bins=30)
    ax.axvline(pred * 100, color='red', linestyle='--', label='Prediction')
    ax.legend()
    st.pyplot(fig)

    # ======================================================
    # 패턴 리포트
    # ======================================================
    st.subheader("🏆 패턴별 승률 리포트")
    report = pattern_report(df)
    st.dataframe(report)

    # ======================================================
    # SHAP
    # ======================================================
    st.subheader("🔍 SHAP 기여도")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[[-1]])

    shap_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP': shap_values[0]
    }).sort_values('SHAP', key=abs, ascending=False)

    st.dataframe(shap_df)
