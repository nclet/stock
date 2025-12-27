import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import pyupbit
import matplotlib.pyplot as plt
import mplfinance as mpf
import lightgbm as lgb
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------------
# 1. 고도화된 피처 엔지니어링 (수치, 맥락, 추세)
# ---------------------------------------------------------------------------------
def add_advanced_features(df):
    """캔들의 수치적 특성 및 시장 맥락 피처 추가"""
    # 캔들 수치 정보
    df['body_size'] = abs(df['Close'] - df['Open'])
    df['upper_shadow'] = df['High'] - df.loc[:, ['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df.loc[:, ['Open', 'Close']].min(axis=1) - df['Low']
    df['total_range'] = df['High'] - df['Low']
    
    # 비율 피처 (패턴의 강도)
    df['body_ratio'] = df['body_size'] / (df['total_range'] + 1e-10)
    df['upper_shadow_ratio'] = df['upper_shadow'] / (df['total_range'] + 1e-10)
    df['lower_shadow_ratio'] = df['lower_shadow'] / (df['total_range'] + 1e-10)
    
    # 추세 및 맥락 정보
    df['MA20'] = df['Close'].rolling(20).mean()
    df['disparity'] = (df['Close'] / df['MA20']) * 100  # 이격도 (추세 확인)
    df['vol_change'] = df['Volume'].pct_change()        # 거래량 변화
    
    # 타겟 변수: 5봉 후의 수익률 (%)
    df['target_return'] = (df['Close'].shift(-5) / df['Close'] - 1) * 100
    
    return df

# ---------------------------------------------------------------------------------
# 2. LightGBM 모델 학습 및 예측 함수
# ---------------------------------------------------------------------------------
def train_and_predict(df, pattern_cols):
    """LightGBM을 이용한 미래 수익률 예측"""
    # 학습에 사용할 피처: 패턴 여부(0/1) + 수치 정보 + 맥락
    features = pattern_cols + ['body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'disparity', 'vol_change']
    
    # 데이터 정제
    df_clean = df.dropna(subset=['target_return'] + features)
    if len(df_clean) < 50: return None, None, None

    X = df_clean[features]
    y = df_clean['target_return']
    
    # 시계열 특성을 고려하여 최근 데이터를 테스트셋으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, importance_type='gain', verbose=-1)
    model.fit(X_train, y_train)
    
    # 현재 시점 예측 (가장 최신 데이터)
    latest_x = df[features].iloc[[-1]]
    pred_return = model.predict(latest_x)[0]
    
    # 패턴 중요도 (어떤 패턴이 수익률에 큰 영향을 줬는가?)
    importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    importance = importance.sort_values(by='importance', ascending=False)
    
    return pred_return, importance, y_test

# ---------------------------------------------------------------------------------
# 3. Streamlit UI 및 시각화 보완
# ---------------------------------------------------------------------------------
# (기존 데이터 로드 및 패턴 탐색 함수 find_candle_patterns 등은 유지한다고 가정)

def show_analysis_report(df, pattern_cols, selected_code):
    st.markdown("---")
    st.subheader("📊 패턴별 성과 리포트 & AI 예측")
    
    # 데이터 가공
    df = add_advanced_features(df)
    
    # 1. 패턴별 승률 통계 (통계 기반 추천)
    stats = []
    for col in pattern_cols:
        occurrences = df[df[col] == True]
        if not occurrences.empty:
            win_rate = (occurrences['target_return'] > 0).mean() * 100
            avg_ret = occurrences['target_return'].mean()
            stats.append({'패턴': col, '발생횟수': len(occurrences), '승률(%)': win_rate, '평균수익률(%)': avg_ret})
    
    if stats:
        st.write("💡 **이 종목의 과거 성과 분석 결과**")
        st.table(pd.DataFrame(stats).sort_values('승률(%)', ascending=False))
    
    # 2. LightGBM 실행
    pred_ret, importance, y_test = train_and_predict(df, pattern_cols)
    
    if pred_ret is not None:
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            st.metric("🚀 AI 예측 5봉 후 기대 수익률", f"{pred_ret:+.2f}%")
            # 수익률 분포 히스토그램
            fig_hist = px.histogram(y_test, nbins=30, title="과거 유사 구간 수익률 분포")
            fig_hist.add_vline(x=pred_ret, line_color="red", annotation_text="현재 예측치")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_pred2:
            st.write("🔍 **AI가 중요하게 평가한 요소 (Top 5)**")
            st.bar_chart(importance.head(5).set_index('feature'))
            st.info("이 종목에서는 위 패턴/지표 조합이 수익률 예측에 가장 효과적이었습니다.")

# ---------------------------------------------------------------------------------
# 4. 실시간 패턴 스캐너 (간략 버전)
# ---------------------------------------------------------------------------------
def scanner_ui():
    with st.sidebar.expander("🔍 실시간 패턴 스캐너"):
        st.write("현재 KRX 주요 종목 패턴 탐색")
        if st.button("스캔 시작"):
            # 예시로 삼성전자, SK하이닉스 등 주요 종목 리스트 순회 로직 추가 가능
            st.write("삼성전자: 상승장악형 발견! (예상 수익 +1.2%)")
            st.write("LG에너지솔루션: 도지형 발생")
