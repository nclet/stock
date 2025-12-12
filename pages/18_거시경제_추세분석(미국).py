# 미국 증시 거시경제 분석 Streamlit 앱

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import requests
import xmltodict
import time # 시간 지연을 위해 추가
import random # 무작위 시간 지연을 위해 추가

# FinanceDataReader 라이브러리 추가 (S&P 500 데이터용)
try:
    import FinanceDataReader as fdr
except ImportError:
    st.error("""
    FinanceDataReader 라이브러리가 설치되지 않았습니다!
    `pip install FinanceDataReader` 명령어를 실행해주세요.
    """)
    st.stop()


# FRED API 키 로드 (secrets.toml에서)
# ECOS 관련 코드는 모두 삭제되었습니다.
try:
    FRED_API_KEY = st.secrets['fred']["FRED_API_KEY"]
    import pandas_datareader.data as web # pandas_datareader는 FRED용으로 유지
except ImportError:
    st.error("""
    **필수 라이브러리가 설치되지 않았거나 API 키가 설정되지 않았습니다!**
    `pip install pandas_datareader requests matplotlib seaborn` 명령어를 실행하고,
    `.streamlit/secrets.toml` 파일에 FRED_API_KEY를 설정해주세요.
    """)
    st.stop()
except KeyError:
    st.error("""
    **FRED API 키가 Streamlit Secrets에 설정되지 않았습니다!**
    `.streamlit/secrets.toml` 파일에 아래 내용을 추가해주세요:
    FRED_API_KEY = "YOUR_FRED_API_KEY"
    """)
    st.stop()

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide")

st.title("🌐 미국 거시경제 지표 기반 시장 추세 분석")
st.markdown("FRED 데이터를 활용하여 미국 시장의 거시경제 국면을 분석하고, S&P 500 지수 추세를 예측합니다.")

# --- 데이터 수집 함수 ---

@st.cache_data(ttl=3600 * 24 * 7) # 1주일 캐시 유지
def get_fred_data(api_key):
    """FRED에서 주요 미국 거시경제 지표를 가져옵니다. KRW/USD 환율은 제외됩니다."""
    st.info("🔄 FRED 데이터 수집 중...")
    start_date = datetime(2010, 1, 1) # 충분한 과거 데이터
    end_date = datetime.now()

    # 한국 관련 지표(KRW/USD ExcRate)는 제거하고 미국 지표만 남깁니다.
    fred_codes = {
        'US_CPI_YoY': 'CPIAUCSL', # 미국 CPI, 월별 (전년 동기 대비 변화율 계산)
        'US_FFR': 'FEDFUNDS', # 미국 기준금리, 월별
        'US_10Y_Treasury': 'DGS10', # 미국 10년 국채금리, 일별 (월말 값 사용)
    }

    df_fred = pd.DataFrame()
    max_fred_retries = 3
    initial_fred_delay = 1 # 초

    for name, code in fred_codes.items():
        for attempt in range(max_fred_retries):
            try:
                temp_df = web.DataReader(code, 'fred', start_date, end_date, api_key=api_key)
                df_fred = pd.concat([df_fred, temp_df], axis=1)
                st.info(f"✅ FRED 데이터 로드 성공: {name}")
                break # 성공하면 재시도 루프 탈출
            except Exception as e:
                if attempt < max_fred_retries - 1:
                    sleep_time = initial_fred_delay * (2 ** attempt) + random.uniform(0, 1)
                    st.warning(f"FRED 데이터 로드 오류 ({name}, {code}): {e}. {sleep_time:.2f}초 후 재시도... ({attempt + 1}/{max_fred_retries})")
                    time.sleep(sleep_time)
                else:
                    st.error(f"❌ FRED 데이터 로드 최종 실패: {name}, {code}. 오류: {e}")
                    # 최종 실패 시 해당 컬럼은 NaN으로 남을 수 있음
                    
        time.sleep(random.uniform(0.3, 0.8)) # 각 지표 호출 후 무작위 지연
    
    df_fred.columns = fred_codes.keys()
    df_fred = df_fred.resample('ME').last().ffill() # 모든 지표를 월말 기준으로 리샘플링하고, 결측치는 이전 값으로 채움
    st.success("✅ FRED 데이터 수집 완료!")
    return df_fred

# ECOS 데이터를 가져오는 함수는 완전히 제거되었습니다.

@st.cache_data(ttl=3600 * 24 * 7) # 1주일 캐시 유지
def get_stock_data():
    """S&P 500 ETF (SPY) 데이터를 FinanceDataReader로 가져옵니다. KOSPI 데이터는 제외됩니다."""
    st.info("🔄 S&P 500 ETF (SPY) 데이터 수집 중...")
    start_date = datetime(2010, 1, 1)
    end_date = datetime.now()
    
    df_stocks = pd.DataFrame()

    max_stock_retries = 5 # 주식 데이터는 재시도 설정
    initial_stock_delay = 2 # 초

    # KOSPI 데이터 로드 코드는 제거되었습니다. S&P 500 ETF (SPY) 데이터만 로드합니다.
    for attempt in range(max_stock_retries):
        try:
            df_spy = fdr.DataReader('SPY', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            if 'Close' in df_spy.columns:
                df_spy_monthly = df_spy['Close'].resample('ME').last().ffill().to_frame(name='US_Stock_Close')
                df_stocks = pd.concat([df_stocks, df_spy_monthly], axis=1)
                st.success("✅ S&P 500 ETF (SPY) 데이터 수집 완료!")
                break # 성공하면 재시도 루프 탈출
            else:
                st.warning("S&P 500 데이터에 'Close' 컬럼이 없습니다. 재시도... (FinanceDataReader 컬럼 문제일 수 있음)")
                if attempt < max_stock_retries - 1:
                    sleep_time = initial_stock_delay * (2 ** attempt) + random.uniform(0, 2)
                    st.info(f"{sleep_time:.2f}초 후 재시도... ({attempt + 1}/{max_stock_retries})")
                    time.sleep(sleep_time)
                else:
                    st.error("❌ S&P 500 ETF (SPY) 데이터 로드 최종 실패: 'Close' 컬럼 없음.")
        except Exception as e:
            if attempt < max_stock_retries - 1:
                sleep_time = initial_stock_delay * (2 ** attempt) + random.uniform(0, 2)
                st.warning(f"❌ S&P 500 ETF (SPY) 데이터 로드 중 오류 발생: {e}. {sleep_time:.2f}초 후 재시도... ({attempt + 1}/{max_stock_retries})")
                time.sleep(sleep_time)
            else:
                st.error(f"❌ S&P 500 ETF (SPY) 데이터 로드 최종 오류 발생: {e}. FinanceDataReader 문제일 수 있습니다.")
    
    if df_stocks.empty:
        st.error("주식 지수 데이터를 전혀 가져오지 못했습니다.")

    return df_stocks


# --- 데이터 전처리 및 파생 변수 생성 ---
@st.cache_data
def preprocess_and_engineer_features(df_fred, df_stocks):
    st.info("🔄 데이터 전처리 및 팩터 생성 중...")
    
    # ECOS 데이터프레임 제거
    valid_dfs = [df for df in [df_fred, df_stocks] if not df.empty]
    
    if not valid_dfs:
        st.error("모든 데이터프레임이 비어있어 데이터를 병합할 수 없습니다.")
        return pd.DataFrame(), []

    # 모든 데이터프레임의 인덱스(날짜)를 가져와서 가장 이른 시작 날짜와 가장 늦은 종료 날짜를 찾음
    all_indices = [df.index for df in valid_dfs]
    min_date = min(idx.min() for idx in all_indices)
    max_date = max(idx.max() for idx in all_indices)

    # 전체 기간에 해당하는 월말 날짜 범위 생성
    full_month_range = pd.date_range(start=min_date, end=max_date, freq='ME')
    
    df_merged = pd.DataFrame(index=full_month_range)
    
    if not df_fred.empty:
        df_merged = pd.merge(df_merged, df_fred, left_index=True, right_index=True, how='left')
    # ECOS 데이터 병합 코드 제거
    if not df_stocks.empty:
        df_merged = pd.merge(df_merged, df_stocks, left_index=True, right_index=True, how='left')

    df_merged.ffill(inplace=True)
    df_merged.bfill(inplace=True)

    # 주요 팩터 생성 (한국 관련 팩터는 모두 제거)
    if 'US_CPI_YoY' in df_merged.columns:
        df_merged['US_CPI_YoY_Change'] = df_merged['US_CPI_YoY'].diff(12) # 전년 동기 대비 변화율 계산
    else: df_merged['US_CPI_YoY_Change'] = np.nan

    if 'US_FFR' in df_merged.columns:
        df_merged['US_FFR_Change'] = df_merged['US_FFR'].diff()
    else: df_merged['US_FFR_Change'] = np.nan

    if 'US_10Y_Treasury' in df_merged.columns:
        df_merged['US_10Y_Treasury_Change'] = df_merged['US_10Y_Treasury'].diff()
    else:
        df_merged['US_10Y_Treasury_Change'] = np.nan

    # KOSPI 다음 달 수익률 코드 제거. 미국 주식 시장 (S&P 500) 다음 달 수익률만 남김
    if 'US_Stock_Close' in df_merged.columns:
        df_merged['US_Next_Month_Return'] = df_merged['US_Stock_Close'].pct_change(1).shift(-1) * 100
    else: df_merged['US_Next_Month_Return'] = np.nan


    # 한국 관련 지표를 제거하고 미국 지표만 사용
    features = [
        'US_CPI_YoY', # CPI 값 자체 추가
        'US_CPI_YoY_Change',
        'US_FFR',
        'US_FFR_Change',
        'US_10Y_Treasury',
        'US_10Y_Treasury_Change'
    ]
    
    actual_features = [f for f in features if f in df_merged.columns]

    # 미국 주식 시장의 다음 달 수익률만 최종 데이터에 포함
    target_returns = ['US_Next_Month_Return']
    actual_targets = [t for t in target_returns if t in df_merged.columns]

    df_final = df_merged[actual_features + actual_targets].dropna()

    st.success("✅ 데이터 전처리 및 팩터 생성 완료!")
    return df_final, actual_features, actual_targets

# --- 시장 국면 정의 및 예측 모델 ---
@st.cache_data
def define_market_regime(df):
    st.info("🔄 시장 국면 정의 중...")
    df_regime = df.copy()

    # 한국 CPI가 제거되었으므로, 시장 국면은 오직 미국 CPI 추세에만 의존하여 분류합니다.
    if 'US_CPI_YoY_Change' in df_regime.columns:
        # 전년 동기 대비 CPI의 6개월 이동평균이 0보다 크면 인플레이션, 아니면 디스인플레이션으로 간주
        df_regime['Inflation_Trend'] = (df_regime['US_CPI_YoY_Change'].rolling(window=6).mean() > 0)
    else:
        df_regime['Inflation_Trend'] = False

    def classify_regime(row):
        if row['Inflation_Trend']:
            return "Inflationary Period"
        else:
            return "Disinflationary Period"

    df_regime['Market_Regime'] = df_regime.apply(classify_regime, axis=1)
    
    st.success("✅ 시장 국면 정의 완료!")
    return df_regime

# --- Streamlit UI 및 실행 로직 ---

st.markdown("---")
st.subheader("데이터 수집 및 분석 시작")
st.write("아래 버튼을 클릭하여 FRED 및 S&P 500 데이터를 수집하고 시장 국면 분석을 시작하세요.")

if st.button("🚀 **데이터 수집 및 분석 시작!**", key="start_analysis_button"):
    with st.spinner("데이터를 수집하고 분석 중입니다. 잠시만 기다려 주세요..."):
        # -----------------------------------------------------------
        # 💡 [핵심 수정] FRED_API_KEY 변수를 인수로 명시적으로 전달 💡
        df_fred = get_fred_data(FRED_API_KEY) 
        # -----------------------------------------------------------
        
        df_stocks = get_stock_data() # SPY 데이터만 호출

        # 데이터가 하나라도 비어있으면 경고 후 중단
        if df_fred.empty or df_stocks.empty:
            st.error("⚠️ 필수 데이터(FRED, S&P 500) 중 일부 또는 전부를 성공적으로 로드하지 못했습니다. 위의 경고 메시지(API 키, 인터넷 연결 등)를 확인하세요.")
            st.stop()

        # 데이터가 하나라도 비어있으면 경고 후 중단
        # ECOS 데이터프레임이 없어졌으므로 체크 로직 수정
        if df_fred.empty or df_stocks.empty:
            st.error("⚠️ 필수 데이터(FRED, S&P 500) 중 일부 또는 전부를 성공적으로 로드하지 못했습니다. 위의 경고 메시지(API 키, 인터넷 연결 등)를 확인하세요.")
            st.stop()
        
        # 2. 데이터 전처리 및 팩터 생성 (ECOS 데이터프레임 제거)
        df_final, features, targets = preprocess_and_engineer_features(df_fred, df_stocks)

        if df_final.empty:
            st.error("데이터 전처리 후 유효한 데이터가 없습니다. 원본 데이터의 날짜 범위 또는 결측치를 확인하세요.")
            st.stop()

        # 3. 시장 국면 정의
        df_regime_classified = define_market_regime(df_final)

        st.subheader("📚 **분석된 거시경제 지표 및 시장 국면**")
        st.dataframe(df_regime_classified.tail(15))

        latest_regime = df_regime_classified['Market_Regime'].iloc[-1]
        st.markdown(f"### ➡️ 현재 시장 국면: **{latest_regime}**")
        
        st.subheader("📈 **시장 국면별 주식 시장 월별 평균 수익률 분석**")
        
        # KOSPI 관련 플롯과 통계는 제거하고 S&P 500만 남김
        if 'US_Next_Month_Return' in df_regime_classified.columns:
            us_regime_performance = df_regime_classified.groupby('Market_Regime')['US_Next_Month_Return'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
            
            fig_us_regime, ax_us_regime = plt.subplots(figsize=(10, 6))
            sns.barplot(x=us_regime_performance.index, y=us_regime_performance['mean'], ax=ax_us_regime, palette='plasma')
            ax_us_regime.set_title("S&P 500 Monthly returns by market phase (%)")
            ax_us_regime.set_xlabel("market phase")
            ax_us_regime.set_ylabel("Average Monthly Return (%)")
            ax_us_regime.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_us_regime)
            st.write("---")
            st.write("**S&P 500 국면별 수익률 통계:**")
            st.dataframe(us_regime_performance)
        else:
            st.warning("S&P 500 시장 국면별 수익률 데이터를 분석할 수 없습니다.")


        st.markdown("---")
        st.subheader("📊 **주요 거시경제 지표 추세 (최근 5년)**")
        
        plot_df = df_regime_classified.last('5Y')
        
        cols = st.columns(3)
        # 한국 관련 지표를 제거하고 미국 지표만 반복
        us_features_to_plot = ['US_CPI_YoY', 'US_CPI_YoY_Change', 'US_FFR', 'US_FFR_Change', 'US_10Y_Treasury', 'US_10Y_Treasury_Change']
        
        for i, feature in enumerate(us_features_to_plot):
            with cols[i % 3]:
                if feature in plot_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(plot_df.index, plot_df[feature])
                    ax.set_title(feature)
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.write(f"⚠️ **'{feature}'** 지표 데이터가 존재하지 않아 시각화할 수 없습니다.")
        
        st.markdown("---")
        st.subheader("Correlation Heatmap: Macroeconomic Factors vs. S&P 500 Returns")
        st.write("거시경제 지표 변화와 S&P 500의 다음 달 수익률 간의 상관관계를 시각화합니다.")

        corr_df = df_final[features + targets].corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
        # KOSPI 관련 데이터가 제거되었으므로, 미국 지표와 S&P 500 수익률 간의 상관관계만 표시
        sns.heatmap(corr_df.loc[features, targets], annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Relationship between macroeconomic indices and S&P 500 returns")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_corr)

        st.markdown("---")
        st.subheader("🔮 **시장 추세 분석 결론 및 제안**")
        st.write(f"현재 거시경제 지표를 분석한 결과, 시장은 **'{latest_regime}'** 국면에 있는 것으로 보입니다.")
        
        # S&P 500 예상 수익률만 계산
        if 'US_Next_Month_Return' in df_regime_classified.columns and latest_regime in us_regime_performance.index:
            us_expected_return = us_regime_performance.loc[latest_regime, 'mean']
            st.write(f"과거 데이터에 기반할 때, 현재 **'{latest_regime}'** 국면에서 S&P 500의 월 평균 수익률은 **{us_expected_return:.2f}%**였습니다.")
            if us_expected_return > 0:
                st.success("✅ 현재 국면은 S&P 500에 긍정적인 경향을 보입니다.")
            else:
                st.warning("⚠️ 현재 국면은 S&P 500에 부정적인 경향을 보입니다. 신중한 접근이 필요합니다.")
        else:
            st.warning("S&P 500 시장의 현재 국면 예상 수익률을 계산할 수 없습니다.")
        
        st.markdown("이 분석은 과거 데이터에 기반한 통계적 경향을 보여주며, 미래 성과를 보장하지 않습니다. 실제 투자 결정 시에는 추가적인 분석과 전문가의 조언을 구하십시오.")
