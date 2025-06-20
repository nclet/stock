import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import requests
import xmltodict
# import financedatareader as fdr
FinanceDataReader 라이브러리 추가 (KOSPI 데이터용)
try:
    import FinanceDataReader as fdr
except ImportError:
    st.error("""
    FinanceDataReader 라이브러리가 설치되지 않았습니다!
    `pip install FinanceDataReader` 명령어를 실행해주세요.
    """)
    st.stop()


# FRED 및 ECOS API 키 로드 (secrets.toml에서)
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    ECOS_API_KEY = st.secrets["ECOS_API_KEY"]
    import pandas_datareader.data as web # pandas_datareader는 FRED용으로 유지
except ImportError:
    st.error("""
    **필수 라이브러리가 설치되지 않았거나 API 키가 설정되지 않았습니다!**
    `pip install pandas_datareader requests xmltodict matplotlib seaborn` 명령어를 실행하고,
    `.streamlit/secrets.toml` 파일에 FRED_API_KEY와 ECOS_API_KEY를 설정해주세요.
    """)
    st.stop()
except KeyError:
    st.error("""
    **FRED 또는 ECOS API 키가 Streamlit Secrets에 설정되지 않았습니다!**
    `.streamlit/secrets.toml` 파일에 아래 내용을 추가해주세요:
    FRED_API_KEY = "YOUR_FRED_API_KEY"
    ECOS_API_KEY = "YOUR_ECOS_API_KEY"
    """)
    st.stop()

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide")

st.title("🌐 거시경제 지표 기반 시장 추세 분석")
st.markdown("FRED와 ECOS 데이터를 활용하여 시장의 거시경제 국면을 분석하고, 한국 주식 시장의 추세를 예측합니다.")

# --- 데이터 수집 함수 ---

@st.cache_data(ttl=3600 * 24 * 7) # 1주일 캐시 유지
def get_fred_data(api_key):
    """FRED에서 주요 거시경제 지표를 가져옵니다. GDP는 제외됩니다."""
    st.info("🔄 FRED 데이터 수집 중...")
    start_date = datetime(2010, 1, 1) # 충분한 과거 데이터
    end_date = datetime.now()

    fred_codes = {
        'US_CPI_YoY': 'CPIAUCSL', # 미국 CPI, 월별 (전년 동기 대비 변화율 계산)
        'US_FFR': 'FEDFUNDS', # 미국 기준금리, 월별
        'US_10Y_Treasury': 'DGS10', # 미국 10년 국채금리, 일별 (월말 값 사용)
        'KRW_USD_ExcRate': 'DEXKOUS' # 원/달러 환율, 일별 (월말 값 사용)
    }

    df_fred = pd.DataFrame()
    for name, code in fred_codes.items():
        try:
            temp_df = web.DataReader(code, 'fred', start_date, end_date, api_key=api_key)
            df_fred = pd.concat([df_fred, temp_df], axis=1)
        except Exception as e:
            st.warning(f"FRED 데이터 로드 오류 ({name}, {code}): {e}")
            continue
    
    df_fred.columns = fred_codes.keys()
    df_fred = df_fred.resample('ME').last().ffill() # 모든 지표를 월말 기준으로 리샘플링하고, 결측치는 이전 값으로 채움
    st.success("✅ FRED 데이터 수집 완료!")
    return df_fred

@st.cache_data(ttl=3600 * 24 * 7) # 1주일 캐시 유지
def get_ecos_data(api_key):
    """ECOS에서 주요 한국 거시경제 지표를 가져옵니다. GDP는 포함되지 않으며 실업률이 추가됩니다."""
    st.info("🔄 ECOS 데이터 수집 중...")
    base_url = "http://ecos.bok.or.kr/api/StatisticSearch/"
    
    # ECOS 주요 지표 코드 및 주기 정보
    ecos_codes_info = {
        'KR_CPI_YoY': {'code': '901Y009/0', 'freq': 'M', 'name_for_url': 'KPIC'}, # 소비자물가지수 (총지수)
        'KR_BaseRate': {'code': '722Y001/0101000', 'freq': 'M', 'name_for_url': 'KR_BaseRate'}, # 기준금리
        'KR_Unemployment_Rate': {'code': '901Y027/I61BB', 'freq': 'M', 'name_for_url': 'KR_Unemployment_Rate'} # 실업률
    }

    df_ecos = pd.DataFrame()
    
    # 데이터 시작일과 종료일 설정 (충분한 과거 데이터 확보를 위해 2010년부터 시작)
    # 현재 날짜를 기준으로 ECOS API 기간을 설정
    now = datetime.now()
    
    # 월별 데이터 포맷:YYYYMM
    start_date_str_month = datetime(2010, 1, 1).strftime('%Y%m') 
    end_date_str_month = (now + timedelta(days=30)).strftime('%Y%m') # 넉넉하게 현재 날짜의 다음 달까지

    for name, info in ecos_codes_info.items():
        try:
            stat_code, item_code = info['code'].split('/')
            freq = info['freq']
            
            # 주기에 따라 start_date와 end_date 형식 조정
            if freq == 'M':
                current_start_date_str = start_date_str_month
                current_end_date_str = end_date_str_month
            elif freq == 'D': # ECOS에 일별 데이터가 있다면
                current_start_date_str = datetime(2010, 1, 1).strftime('%Y%m%d')
                current_end_date_str = (now + timedelta(days=30)).strftime('%Y%m%d')
            elif freq == 'A': # ECOS에 연간 데이터가 있다면
                current_start_date_str = "2010"
                current_end_date_str = str(now.year)
            else: # 현재 코드에서는 M, D, A 외의 주기는 없으므로 추가 필요 시 확장
                st.warning(f"ECOS API 호출에서 지원하지 않는 주기 형식입니다: {freq} for {name}")
                continue
                
            # ECOS API는 한 번에 최대 1000건의 데이터를 반환
            url = f"{base_url}{api_key}/json/kr/1/1000/{stat_code}/{freq}/{current_start_date_str}/{current_end_date_str}/{item_code}"
            
            response = requests.get(url)
            data = response.json()
            
            if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                rows = data['StatisticSearch']['row']
                temp_df = pd.DataFrame(rows)
                
                if 'TIME' in temp_df.columns and 'DATA_VALUE' in temp_df.columns:
                    # ECOS 주기별 날짜 포맷 변환 및 월말 날짜 설정
                    if freq == 'M': # 월별 데이터 (YYYYMM -> YYYY-MM-DD 월말)
                        temp_df['TIME'] = pd.to_datetime(temp_df['TIME'], format='%Y%m', errors='coerce') + pd.offsets.MonthEnd(0) # 월말 날짜로 변경
                    elif freq == 'D': # 일별 데이터 (YYYYMMDD -> YYYY-MM-DD)
                        temp_df['TIME'] = pd.to_datetime(temp_df['TIME'], format='%Y%m%d', errors='coerce')
                    elif freq == 'A': # 연간 데이터 (YYYY -> YYYY-12-31)
                        temp_df['TIME'] = pd.to_datetime(temp_df['TIME'], format='%Y', errors='coerce') + pd.offsets.YearEnd(0)
                    else: # 이 부분은 위에 처리되겠지만, 혹시 모를 경우를 대비
                        st.warning(f"알 수 없는 ECOS 주기 형식: {freq} for {name}")
                        continue
                    
                    temp_df.dropna(subset=['TIME'], inplace=True) # 날짜 변환 실패한 행 제거
                    temp_df.drop_duplicates(subset=['TIME'], inplace=True) # 중복 날짜 제거
                    temp_df.set_index('TIME', inplace=True)
                    
                    temp_df[name] = pd.to_numeric(temp_df['DATA_VALUE'], errors='coerce') # 데이터 값 숫자로 변환
                    
                    if df_ecos.empty:
                        df_ecos = temp_df[[name]]
                    else:
                        df_ecos = pd.merge(df_ecos, temp_df[[name]], left_index=True, right_index=True, how='outer')
                else:
                    st.warning(f"ECOS 응답 형식 오류: '{name}'에 예상 컬럼 'TIME' 또는 'DATA_VALUE' 없음. 실제 응답 컬럼: {temp_df.columns.tolist()}")
            elif 'RESULT' in data and data['RESULT']['CODE'] != 'INFO-000':
                    # ECOS API에서 오류 메시지가 왔을 때 출력
                    st.warning(f"ECOS API 호출 오류 ({name}, {info['code']}): {data['RESULT']['CODE']} - {data['RESULT']['MESSAGE']}")
            else:
                st.warning(f"ECOS 데이터 오류 또는 응답 없음: {name}, {info['code']}. 응답: {data}")
                
        except Exception as e:
            st.warning(f"ECOS 데이터 로드 중 예상치 못한 오류 ({name}, {info['code']}): {e}")
            continue
    
    if not df_ecos.empty:
        df_ecos = df_ecos.resample('ME').last().ffill() # 월말 기준으로 리샘플링
    else:
        st.error("ECOS에서 유효한 데이터를 전혀 가져오지 못했습니다. ECOS API 키, 통계표 코드 및 네트워크 상태를 다시 확인해주세요.")
        
    st.success("✅ ECOS 데이터 수집 완료!")
    return df_ecos

@st.cache_data(ttl=3600 * 24 * 7) # 1주일 캐시 유지
def get_stock_data():
    """KOSPI 지수 및 S&P 500 ETF (SPY) 데이터를 FinanceDataReader로 가져옵니다."""
    st.info("🔄 주식 지수 데이터 (KOSPI, S&P 500) 수집 중...")
    start_date = datetime(2010, 1, 1)
    end_date = datetime.now()
    
    df_stocks = pd.DataFrame()

    # KOSPI 데이터 로드
    try:
        df_kospi = fdr.DataReader('KS11', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if 'Close' in df_kospi.columns:
            df_kospi_monthly = df_kospi['Close'].resample('ME').last().ffill().to_frame(name='KOSPI_Close')
            df_stocks = pd.concat([df_stocks, df_kospi_monthly], axis=1)
            st.success("✅ KOSPI 지수 데이터 수집 완료!")
        else:
            st.warning("KOSPI 데이터에 'Close' 컬럼이 없습니다.")

    except Exception as e:
        st.error(f"❌ KOSPI 지수 데이터 로드 중 오류 발생: {e}. FinanceDataReader 문제일 수 있습니다.")

    # S&P 500 ETF (SPY) 데이터 로드
    try:
        df_spy = fdr.DataReader('SPY', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if 'Close' in df_spy.columns:
            df_spy_monthly = df_spy['Close'].resample('ME').last().ffill().to_frame(name='US_Stock_Close')
            df_stocks = pd.concat([df_stocks, df_spy_monthly], axis=1)
            st.success("✅ S&P 500 ETF (SPY) 데이터 수집 완료!")
        else:
            st.warning("S&P 500 데이터에 'Close' 컬럼이 없습니다.")
    except Exception as e:
        st.error(f"❌ S&P 500 ETF (SPY) 데이터 로드 중 오류 발생: {e}. FinanceDataReader 문제일 수 있습니다.")
    
    if df_stocks.empty:
        st.error("주식 지수 데이터를 전혀 가져오지 못했습니다.")

    return df_stocks


# --- 데이터 전처리 및 파생 변수 생성 ---
@st.cache_data
def preprocess_and_engineer_features(df_fred, df_ecos, df_stocks):
    st.info("🔄 데이터 전처리 및 팩터 생성 중...")
    
    valid_dfs = [df for df in [df_fred, df_ecos, df_stocks] if not df.empty]
    
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
    if not df_ecos.empty:
        df_merged = pd.merge(df_merged, df_ecos, left_index=True, right_index=True, how='left')
    if not df_stocks.empty:
        df_merged = pd.merge(df_merged, df_stocks, left_index=True, right_index=True, how='left')

    df_merged.ffill(inplace=True)
    df_merged.bfill(inplace=True)

    # 주요 팩터 생성
    if 'US_CPI_YoY' in df_merged.columns:
        df_merged['US_CPI_YoY_Change'] = df_merged['US_CPI_YoY'].pct_change(12) * 100
    else: df_merged['US_CPI_YoY_Change'] = np.nan

    if 'KR_CPI_YoY' in df_merged.columns:
        df_merged['KR_CPI_YoY_Change'] = df_merged['KR_CPI_YoY'].pct_change(12) * 100
    else: df_merged['KR_CPI_YoY_Change'] = np.nan

    if 'KR_Unemployment_Rate' in df_merged.columns:
        df_merged['KR_Unemployment_Rate_YoY_Change'] = df_merged['KR_Unemployment_Rate'].diff(12) # 전년 동월 대비 변화 (절대값)
    else: df_merged['KR_Unemployment_Rate_YoY_Change'] = np.nan

    if 'US_FFR' in df_merged.columns:
        df_merged['US_FFR_Change'] = df_merged['US_FFR'].diff()
    else: df_merged['US_FFR_Change'] = np.nan

    if 'KR_BaseRate' in df_merged.columns:
        df_merged['KR_BaseRate_Change'] = df_merged['KR_BaseRate'].diff()
    else: df_merged['KR_BaseRate_Change'] = np.nan

    if 'KRW_USD_ExcRate' in df_merged.columns:
        df_merged['KRW_USD_ExcRate_Change'] = df_merged['KRW_USD_ExcRate'].pct_change(1) * 100
    else: df_merged['KRW_USD_ExcRate_Change'] = np.nan

    if 'US_10Y_Treasury' in df_merged.columns:
        df_merged['US_10Y_Treasury_Change'] = df_merged['US_10Y_Treasury'].diff()
    else:
        df_merged['US_10Y_Treasury_Change'] = np.nan

    # KOSPI 다음 달 수익률
    if 'KOSPI_Close' in df_merged.columns:
        df_merged['KOSPI_Next_Month_Return'] = df_merged['KOSPI_Close'].pct_change(1).shift(-1) * 100
    else: df_merged['KOSPI_Next_Month_Return'] = np.nan

    # 미국 주식 시장 (S&P 500) 다음 달 수익률
    if 'US_Stock_Close' in df_merged.columns:
        df_merged['US_Next_Month_Return'] = df_merged['US_Stock_Close'].pct_change(1).shift(-1) * 100
    else: df_merged['US_Next_Month_Return'] = np.nan


    features = [
        'US_CPI_YoY_Change', 
        'KR_CPI_YoY_Change', 
        'KR_Unemployment_Rate_YoY_Change',
        'US_FFR', 'US_FFR_Change', 'KR_BaseRate', 'KR_BaseRate_Change',
        'KRW_USD_ExcRate_Change',
        'US_10Y_Treasury', 'US_10Y_Treasury_Change'
    ]
    
    actual_features = [f for f in features if f in df_merged.columns]

    # KOSPI와 미국 주식 시장의 다음 달 수익률도 최종 데이터에 포함
    target_returns = ['KOSPI_Next_Month_Return', 'US_Next_Month_Return']
    actual_targets = [t for t in target_returns if t in df_merged.columns]

    df_final = df_merged[actual_features + actual_targets].dropna()

    st.success("✅ 데이터 전처리 및 팩터 생성 완료!")
    return df_final, actual_features, actual_targets

# --- 시장 국면 정의 및 예측 모델 ---
@st.cache_data
def define_market_regime(df):
    st.info("🔄 시장 국면 정의 중...")
    df_regime = df.copy()

    df_regime['Growth_Trend'] = True # 이 값은 현재 시장 국면 분류 로직에 영향을 미치지 않습니다.

    # CPI 지표(미국, 한국) 모두 사용
    if 'US_CPI_YoY_Change' in df_regime.columns and 'KR_CPI_YoY_Change' in df_regime.columns:
        df_regime['Inflation_Trend'] = (df_regime['US_CPI_YoY_Change'].rolling(window=6).mean() > 0) & \
                                         (df_regime['KR_CPI_YoY_Change'].rolling(window=6).mean() > 0)
    elif 'US_CPI_YoY_Change' in df_regime.columns: # 한국 CPI가 없을 경우 미국 CPI만 사용
        df_regime['Inflation_Trend'] = (df_regime['US_CPI_YoY_Change'].rolling(window=6).mean() > 0)
    else:
        df_regime['Inflation_Trend'] = False

    def classify_regime(row):
        # GDP가 제거되었으므로, 시장 국면은 물가 추세에만 의존하여 분류합니다.
        if row['Inflation_Trend']:
            return "물가 상승 국면 (Inflationary Period)"
        else:
            return "물가 하락 국면 (Disinflationary Period)"

    df_regime['Market_Regime'] = df_regime.apply(classify_regime, axis=1)
    
    st.success("✅ 시장 국면 정의 완료!")
    return df_regime

# --- Streamlit UI 및 실행 로직 ---

st.markdown("---")
st.subheader("데이터 수집 및 분석 시작")
st.write("아래 버튼을 클릭하여 FRED, ECOS, KOSPI 데이터를 수집하고 시장 국면 분석을 시작하세요.")

if st.button("🚀 **데이터 수집 및 분석 시작!**", key="start_analysis_button"):
    with st.spinner("데이터를 수집하고 분석 중입니다. 잠시만 기다려 주세요..."):
        # 1. 데이터 수집
        df_fred = get_fred_data(FRED_API_KEY)
        df_ecos = get_ecos_data(ECOS_API_KEY) # CPI, 기준금리, 실업률 포함된 ECOS 데이터 호출
        df_stocks = get_stock_data() # KOSPI 및 SPY 데이터 호출

        # 데이터가 하나라도 비어있으면 경고 후 중단
        if df_fred.empty or df_ecos.empty or df_stocks.empty:
            st.error("⚠️ 필수 데이터(FRED, ECOS, 주식 지수) 중 일부 또는 전부를 성공적으로 로드하지 못했습니다. 위의 경고 메시지(API 키, 인터넷 연결 등)를 확인하세요.")
            st.stop()
        
        # 2. 데이터 전처리 및 팩터 생성
        df_final, features, targets = preprocess_and_engineer_features(df_fred, df_ecos, df_stocks)

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
        
        col1, col2 = st.columns(2)

        # KOSPI 국면별 수익률
        with col1:
            if 'KOSPI_Next_Month_Return' in df_regime_classified.columns:
                kospi_regime_performance = df_regime_classified.groupby('Market_Regime')['KOSPI_Next_Month_Return'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
                
                fig_kospi_regime, ax_kospi_regime = plt.subplots(figsize=(10, 6))
                sns.barplot(x=kospi_regime_performance.index, y=kospi_regime_performance['mean'], ax=ax_kospi_regime, palette='viridis')
                ax_kospi_regime.set_title("KOSPI 시장 국면별 월별 평균 수익률 (%)")
                ax_kospi_regime.set_xlabel("시장 국면")
                ax_kospi_regime.set_ylabel("평균 월별 수익률 (%)")
                ax_kospi_regime.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig_kospi_regime)
                st.write("---")
                st.write("**KOSPI 국면별 수익률 통계:**")
                st.dataframe(kospi_regime_performance)
            else:
                st.warning("KOSPI 시장 국면별 수익률 데이터를 분석할 수 없습니다.")

        # 미국 주식 시장 (S&P 500) 국면별 수익률
        with col2:
            if 'US_Next_Month_Return' in df_regime_classified.columns:
                us_regime_performance = df_regime_classified.groupby('Market_Regime')['US_Next_Month_Return'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
                
                fig_us_regime, ax_us_regime = plt.subplots(figsize=(10, 6))
                sns.barplot(x=us_regime_performance.index, y=us_regime_performance['mean'], ax=ax_us_regime, palette='plasma')
                ax_us_regime.set_title("S&P 500 시장 국면별 월별 평균 수익률 (%)")
                ax_us_regime.set_xlabel("시장 국면")
                ax_us_regime.set_ylabel("평균 월별 수익률 (%)")
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
        for i, feature in enumerate(features):
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
        st.subheader("Correlation Heatmap: Macroeconomic Factors vs. Stock Returns")
        st.write("거시경제 지표 변화와 한국/미국 주식 시장의 다음 달 수익률 간의 상관관계를 시각화합니다.")

        corr_df = df_final[features + targets].corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_df.loc[features, targets], annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title("거시경제 지표와 주식 시장 다음 달 수익률 간의 상관관계")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_corr)

        st.markdown("---")
        st.subheader("🔮 **시장 추세 분석 결론 및 제안**")
        st.write(f"현재 거시경제 지표를 분석한 결과, 시장은 **'{latest_regime}'** 국면에 있는 것으로 보입니다.")
        
        # 현재 국면별 KOSPI 예상 수익률
        if 'KOSPI_Next_Month_Return' in df_regime_classified.columns and latest_regime in kospi_regime_performance.index:
            kospi_expected_return = kospi_regime_performance.loc[latest_regime, 'mean']
            st.write(f"과거 데이터에 기반할 때, 현재 **'{latest_regime}'** 국면에서 KOSPI의 월 평균 수익률은 **{kospi_expected_return:.2f}%**였습니다.")
            if kospi_expected_return > 0:
                st.success("✅ 현재 국면은 KOSPI에 긍정적인 경향을 보입니다.")
            else:
                st.warning("⚠️ 현재 국면은 KOSPI에 부정적인 경향을 보입니다. 신중한 접근이 필요합니다.")
        else:
            st.warning("KOSPI 시장의 현재 국면 예상 수익률을 계산할 수 없습니다.")

        # 현재 국면별 S&P 500 예상 수익률
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
