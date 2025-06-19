import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import requests
import xmltodict

# FinanceDataReader 라이브러리 추가 (KOSPI 데이터용)
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
        # 'US_GDP_QoQ': 'GDPPOT', # 잠재 GDP, 분기 (추세용) - GDP 제거됨
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
    """ECOS에서 주요 한국 거시경제 지표를 가져옵니다. GDP는 포함되지 않습니다."""
    st.info("🔄 ECOS 데이터 수집 중...")
    base_url = "http://ecos.bok.or.kr/api/StatisticSearch/"
    
    # ECOS 주요 지표 코드 및 주기 정보
    # GDP는 제외하고 CPI와 기준금리만 포함
    ecos_codes_info = {
        'KR_CPI_YoY': {'code': '901Y009/0', 'freq': 'M', 'name_for_url': 'KPIC'}, # 소비자물가지수 (총지수)
        'KR_BaseRate': {'code': '722Y001/0101000', 'freq': 'M', 'name_for_url': 'KR_BaseRate'}, # 기준금리
    }

    df_ecos = pd.DataFrame()
    
    # 데이터 시작일과 종료일 설정 (충분한 과거 데이터 확보를 위해 2010년부터 시작)
    # 현재 날짜를 기준으로 ECOS API 기간을 설정
    now = datetime.now()
    
    # 월별 데이터 포맷: YYYYMM
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
def get_kospi_data():
    """KOSPI 지수 데이터를 FinanceDataReader로 가져옵니다."""
    st.info("🔄 KOSPI 지수 데이터 수집 중...")
    start_date = datetime(2010, 1, 1)
    end_date = datetime.now()
    try:
        df_kospi = fdr.DataReader('KS11', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if 'Close' in df_kospi.columns:
            df_kospi = df_kospi['Close'].resample('ME').last().ffill().to_frame(name='KOSPI_Close')
            st.success("✅ KOSPI 지수 데이터 수집 완료!")
            return df_kospi
        else:
            st.error("KOSPI 데이터에 'Close' 컬럼이 없습니다. FinanceDataReader의 컬럼명을 확인해주세요.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ KOSPI 지수 데이터 로드 중 오류 발생: {e}. FinanceDataReader 문제일 수 있습니다.")
        return pd.DataFrame()


# --- 데이터 전처리 및 파생 변수 생성 ---
@st.cache_data
def preprocess_and_engineer_features(df_fred, df_ecos, df_kospi):
    st.info("🔄 데이터 전처리 및 팩터 생성 중...")
    
    valid_dfs = [df for df in [df_fred, df_ecos, df_kospi] if not df.empty]
    
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
    if not df_kospi.empty:
        df_merged = pd.merge(df_merged, df_kospi, left_index=True, right_index=True, how='left')

    df_merged.ffill(inplace=True)
    df_merged.bfill(inplace=True)

    # 주요 팩터 생성
    if 'US_CPI_YoY' in df_merged.columns:
        df_merged['US_CPI_YoY_Change'] = df_merged['US_CPI_YoY'].pct_change(12) * 100
    else: df_merged['US_CPI_YoY_Change'] = np.nan

    if 'KR_CPI_YoY' in df_merged.columns: # 한국 CPI가 다시 포함됨
        df_merged['KR_CPI_YoY_Change'] = df_merged['KR_CPI_YoY'].pct_change(12) * 100
    else: df_merged['KR_CPI_YoY_Change'] = np.nan

    # GDP 관련 항목 모두 제거
    # df_merged['US_GDP_QoQ_Change'] = df_merged['US_GDP_QoQ'].pct_change(4) * 100 if 'US_GDP_QoQ' in df_merged.columns else np.nan
    # df_merged['KR_GDP_QoQ_Change'] = np.nan 

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


    if 'KOSPI_Close' in df_merged.columns:
        df_merged['KOSPI_Next_Month_Return'] = df_merged['KOSPI_Close'].pct_change(1).shift(-1) * 100
    else: df_merged['KOSPI_Next_Month_Return'] = np.nan


    features = [
        'US_CPI_YoY_Change', 
        'KR_CPI_YoY_Change', # 한국 CPI 포함
        # 'US_GDP_QoQ_Change', # FRED GDP 제거됨
        'US_FFR', 'US_FFR_Change', 'KR_BaseRate', 'KR_BaseRate_Change',
        'KRW_USD_ExcRate_Change',
        'US_10Y_Treasury', 'US_10Y_Treasury_Change'
    ]
    
    actual_features = [f for f in features if f in df_merged.columns]

    df_final = df_merged[actual_features + ['KOSPI_Next_Month_Return']].dropna()

    st.success("✅ 데이터 전처리 및 팩터 생성 완료!")
    return df_final, actual_features

# --- 시장 국면 정의 및 예측 모델 ---
@st.cache_data
def define_market_regime(df):
    st.info("🔄 시장 국면 정의 중...")
    df_regime = df.copy()

    # 성장 추세 정의 (GDP 제거되었으므로 성장 추세는 사용하지 않음)
    df_regime['Growth_Trend'] = True # 모든 기간을 '성장'으로 간주하거나, 이 부분을 아예 제거

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
        df_ecos = get_ecos_data(ECOS_API_KEY) # CPI와 기준금리 포함된 ECOS 데이터 호출
        df_kospi = get_kospi_data()

        # 데이터가 하나라도 비어있으면 경고 후 중단
        if df_fred.empty or df_ecos.empty or df_kospi.empty:
            st.error("⚠️ 필수 데이터(FRED, ECOS, KOSPI) 중 일부 또는 전부를 성공적으로 로드하지 못했습니다. 위의 경고 메시지(API 키, 인터넷 연결 등)를 확인하세요.")
            st.stop()
        
        # 2. 데이터 전처리 및 팩터 생성
        df_final, features = preprocess_and_engineer_features(df_fred, df_ecos, df_kospi)

        if df_final.empty:
            st.error("데이터 전처리 후 유효한 데이터가 없습니다. 원본 데이터의 날짜 범위 또는 결측치를 확인하세요.")
            st.stop()

        # 3. 시장 국면 정의
        df_regime_classified = define_market_regime(df_final)

        st.subheader("📚 **분석된 거시경제 지표 및 시장 국면**")
        st.dataframe(df_regime_classified.tail(15))

        latest_regime = df_regime_classified['Market_Regime'].iloc[-1]
        st.markdown(f"### ➡️ 현재 시장 국면: **{latest_regime}**")
        
        st.subheader("📈 **시장 국면별 KOSPI 월별 평균 수익률 분석**")
        
        # GDP 지표가 없으므로 2가지 국면만 분류될 수 있음
        # if df_regime_classified['Market_Regime'].nunique() < 4: # 이 경고는 더 이상 필요 없음
        #     st.warning("⚠️ 한국 GDP 데이터가 부족하여 시장 국면 분류 시 한국 성장 추세가 반영되지 않습니다. 분석 결과가 제한적일 수 있습니다.")

        regime_performance = df_regime_classified.groupby('Market_Regime')['KOSPI_Next_Month_Return'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
        
        fig_regime, ax_regime = plt.subplots(figsize=(10, 6))
        sns.barplot(x=regime_performance.index, y=regime_performance['mean'], ax=ax_regime, palette='viridis')
        ax_regime.set_title("Average monthly return on KOSPI by market phase(%)")
        ax_regime.set_xlabel("the state of the market")
        ax_regime.set_ylabel("average monthly rate of return(%)")
        ax_regime.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_regime)

        st.write("---")
        st.write(regime_performance)
        
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
        st.subheader("🔮 **시장 추세 분석 결론 및 제안**")
        st.write(f"현재 거시경제 지표를 분석한 결과, 시장은 **'{latest_regime}'** 국면에 있는 것으로 보입니다.")
        
        if not regime_performance.empty:
            best_regime = regime_performance.index[0]
            best_mean_return = regime_performance['mean'].iloc[0]

            st.write(f"과거 데이터에 따르면, '{best_regime}' 국면에서 KOSPI가 월 평균 **{best_mean_return:.2f}%**로 가장 좋은 성과를 보였습니다.")
            
            if latest_regime == best_regime:
                st.success("✅ 현재 시장 국면은 KOSPI가 가장 좋은 성과를 보였던 국면과 일치합니다. 긍정적인 시장 흐름이 예상됩니다.")
            else:
                st.info(f"💡 현재 시장 국면은 '{latest_regime}'이며, 과거 데이터에 기반할 때 '{best_regime}' 국면만큼의 긍정적인 성과는 아닐 수 있습니다. 거시경제 지표의 변화를 주시하며 신중한 접근이 필요합니다.")
        else:
            st.warning("시장 국면별 KOSPI 수익률 데이터를 분석할 수 없습니다. 데이터 부족 또는 오류를 확인하세요.")
        
        st.markdown("이 분석은 과거 데이터에 기반한 통계적 경향을 보여주며, 미래 성과를 보장하지 않습니다. 실제 투자 결정 시에는 추가적인 분석과 전문가의 조언을 구하십시오.")
