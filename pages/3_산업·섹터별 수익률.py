import streamlit as st # 웹 앱을 만들기 위한 라이브러리
import pandas as pd    # 데이터 처리를 위한 라이브러리
import numpy as np     # 숫자 계산을 위한 라이브러리
import FinanceDataReader as fdr # 주가 데이터를 쉽게 가져오기 위한 라이브러리
import datetime        # 날짜와 시간 처리를 위한 라이브러리
import warnings        # 경고 메시지를 제어하기 위한 라이브러리

# 불필요한 경고 메시지를 숨깁니다.
warnings.filterwarnings('ignore')

# --- 1. 주가 데이터를 가져오는 함수 (ETF에 맞춰 재사용) ---
@st.cache_data(ttl=3600*24) # 24시간 동안 캐싱
def get_etf_data(ticker, start_date, end_date):
    """
    FinanceDataReader를 사용하여 특정 ETF의 주가 데이터를 가져옵니다.
    Args:
        ticker (str): ETF 종목 코드 (예: '305720' KODEX 2차전지산업)
        start_date (datetime.date): 데이터 시작일
        end_date (datetime.date): 데이터 종료일
    Returns:
        pd.DataFrame: ETF 주가 데이터 (날짜, 시가, 고가, 저가, 종가, 거래량 등)
    """
    try:
        df = fdr.DataReader(ticker, start=start_date, end=end_date)
        return df
    except Exception as e:
        # 오류 메시지 대신 경고만 출력하고 빈 데이터프레임 반환
        st.warning(f"🚨 ETF '{ticker}' 데이터를 가져오는 중 문제가 발생했습니다: {e}. 해당 ETF는 상장일 이후 데이터부터 표시됩니다.")
        return pd.DataFrame() 

# --- Streamlit 웹 앱의 메인 레이아웃 구성 시작 ---
st.set_page_config(layout="wide", page_title="ETF 기반 섹터 로테이션 분석") # 페이지 제목 설정
st.title("📈 ETF 기반 산업/섹터 로테이션 백테스팅")

st.markdown("""
해당 페이지는 **ETF(상장지수펀드) 데이터를 활용**하여 주요 산업/섹터의 과거 성과를 비교하고, 
장기적인 관점에서 어떤 섹터가 강세를 보였는지 분석합니다.
**선택된 기간 내 상장된 ETF는 상장일 이후부터 데이터가 표시됩니다.**
""")

# --- 분석할 산업/섹터별 ETF 정의 ---
# 실제 시장에서 유동성이 높고 대표성을 띠는 ETF 티커를 선정했습니다.
# (2025년 6월 현재 기준 예시, 실제 투자 시 최신 정보 확인 필수)#TIGER Fn반도체TOP10396500
sector_etfs = {
    "반도체": {"티커": "396500", "이름": "TIGER Fn반도체"}, # 2019-09-09 상장
    "2차전지": {"티커": "305720", "이름": "KODEX 2차전지산업"}, # 2018-09-11 상장
    "자동차": {"티커": "102110", "이름": "KODEX 자동차"}, # 2008-04-22 상장
    "바이오": {"티커": "244790", "이름": "KODEX 바이오"}, # 2016-09-06 상장
    "인터넷": {"티커": "369370", "이름": "TIGER KRX인터넷K뉴딜"}, # 2020-08-07 상장
    "은행": {"티커": "117700", "이름": "KODEX 은행"}, # 2008-11-25 상장
    "철강": {"티커": "139240", "이름": "TIGER 철강"}, # 2011-08-25 상장
    "음식료": {"티커": "139250", "이름": "TIGER 음식료"}, # 2011-08-25 상장
    "게임": {"티커": "364980", "이름": "KODEX 게임산업"}, # 2020-08-07 상장
    "건설": {"티커": "139260", "이름": "TIGER 건설"}, # 2011-08-25 상장
    "방산": {"티커": "449450", "이름": "PLUS K방산"},
    "조선": {"티커": "494670", "이름": "TIGER 조선TOP10"}
}

st.subheader("📊 분석 설정")

# --- 사용자 입력: 분석할 섹터와 기간 선택 ---
# 모든 섹터를 기본으로 선택하도록 변경
all_sector_names = list(sector_etfs.keys())
selected_sectors = st.multiselect(
    "📊 **분석할 산업/섹터를 선택하세요 (다중 선택 가능):**",
    options=all_sector_names,
    default=all_sector_names # 모든 섹터를 기본으로 선택
)

# 지난 10년간의 시작일로 기본 설정 (오늘 날짜 - 10년)
today = datetime.date.today()
# Current time is Monday, June 16, 2025 at 3:05:32 PM KST.
# 2025년 6월 16일 기준 10년 전은 2015년 6월 16일 (대략적인 계산)
start_of_10_years_ago = today - datetime.timedelta(days=365 * 10 + 2) 


col1, col2 = st.columns(2)
with col1:
    start_date_sector = st.date_input("분석 시작일", value=start_of_10_years_ago, key='sector_start_date')
with col2:
    end_date_sector = st.date_input("분석 종료일", value=today, key='sector_end_date')


# 섹터 성과 분석 시작 버튼
if st.button("섹터 성과 분석 시작", key='analyze_sectors_button'):
    if not selected_sectors:
        st.warning("분석할 산업/섹터를 하나 이상 선택해주세요.")
    else:
        st.subheader("📈 선택된 ETF 섹터별 누적 수익률 비교")
        
        # 모든 ETF의 수익률을 담을 빈 DataFrame을 생성 (날짜 인덱스 미리 생성)
        # 선택된 기간 내 모든 날짜를 포함하는 DatetimeIndex를 만듭니다.
        full_date_range = pd.date_range(start=start_date_sector, end=end_date_sector, freq='B') # 'B'는 영업일 기준
        all_etf_cumulative_returns = pd.DataFrame(index=full_date_range)
        
        # 실제로 데이터가 로드된 섹터들의 이름을 저장할 리스트
        loaded_sectors = []
        
        # 진행 상황을 보여주는 바 (데이터 로딩이 시간이 걸릴 수 있습니다.)
        progress_text = "ETF 데이터 로딩 중입니다. 잠시만 기다려 주세요..."
        my_bar = st.progress(0, text=progress_text)
        
        # 선택된 각 섹터에 대해 반복하며 데이터를 가져오고 수익률을 계산합니다.
        for i, sector_name in enumerate(selected_sectors):
            # 진행바 업데이트
            my_bar.progress((i + 1) / len(selected_sectors), text=f"'{sector_name}' ETF 데이터 로딩 중...")
            
            etf_ticker = sector_etfs[sector_name]["티커"]
            etf_name = sector_etfs[sector_name]["이름"]

            # ETF 주가 데이터를 가져옵니다.
            etf_data = get_etf_data(etf_ticker, start_date_sector, end_date_sector)
            
            if not etf_data.empty: # ETF 데이터가 성공적으로 있다면
                # 종가 기준으로 일별 누적 수익률을 계산합니다.
                # 첫 날의 종가를 1로 기준하여 이후 날짜의 수익률을 상대적으로 보여줍니다.
                etf_cumulative_return = etf_data['Close'] / etf_data['Close'].iloc[0]
                
                # 전체 날짜 범위 데이터프레임에 해당 ETF의 누적 수익률을 추가합니다.
                # join()을 사용하여 날짜 인덱스를 기준으로 데이터를 합칩니다.
                all_etf_cumulative_returns = all_etf_cumulative_returns.join(etf_cumulative_return.rename(sector_name))
                loaded_sectors.append(sector_name) # 성공적으로 로드된 섹터만 리스트에 추가
            else:
                # 데이터가 없는 ETF는 경고 메시지만 출력하고 분석은 계속 진행합니다.
                pass 

        my_bar.empty() # 모든 로딩이 끝나면 진행바를 숨깁니다.

        # 모든 ETF 데이터가 합쳐진 후에 NaN 값을 처리합니다.
        # 상장일 이전의 NaN 값은 1.0으로 채워 차트 시작점을 통일하고,
        # 이후의 NaN 값(예: 특정일 휴장 등)은 이전 유효한 값으로 채웁니다.
        returns_df = all_etf_cumulative_returns.fillna(1.0).ffill().bfill()
        
        # --- 이전 수정 부분 (KeyError 해결) ---
        if loaded_sectors: # 로드된 섹터가 하나라도 있다면
            returns_df = returns_df[loaded_sectors]
        else: # 로드된 섹터가 하나도 없다면
            st.error("선택된 기간 내 유효한 ETF 데이터를 가져올 수 없습니다. 기간 또는 섹터 설정을 확인해주세요.")
            st.stop() # 더 이상 진행하지 않고 앱을 중단합니다.
        # --- 이전 수정 부분 끝 ---


        if not returns_df.empty and not returns_df.columns.empty: # 최종 데이터프레임이 비어있지 않고 컬럼이 있다면
            st.write("선택된 ETF 섹터별 누적 수익률 추이 (초기값 1.0 기준):")
            st.dataframe(returns_df.tail()) # 최근 데이터 5개 보여주기

            st.line_chart(returns_df) # 누적 수익률 차트 그리기

            st.markdown("""
            **차트 해석:**
            위 차트는 선택된 기간 동안 각 산업/섹터 ETF의 투자 시작 시점(1.0) 대비 누적된 수익률 변화를 보여줍니다.
            * **곡선이 위로 가파르게 상승할수록** 해당 기간 동안 높은 수익률을 기록했음을 의미합니다.
            * 여러 곡선을 비교하여 어떤 섹터가 특정 시점에 강세를 보였는지 파악할 수 있습니다.
            * **차트 시작 지점에서 1.0으로 평평하게 유지되다가 상승하는 곡선**은 해당 ETF가 분석 시작일 이후에 상장되었음을 나타냅니다.
            * 기간 중 **상대적으로 높은 위치에 있는 섹터**가 해당 시기 시장을 주도한 섹터일 가능성이 높습니다.
            """)

            # 기간별 최종 수익률 요약 및 막대 차트
            # Charting-specific data (pure numeric for bar chart)
            final_returns_for_chart = returns_df.iloc[-1] - 1 
            final_returns_for_chart_percent = (final_returns_for_chart * 100).dropna().sort_values(ascending=False) 
            
            # --- 새로운 시도: Series의 이름을 명시하고, 데이터프레임으로 변환하여 전달 ---
            # Altair가 Series를 처리하는 방식이 까다로울 수 있으므로,
            # 명확한 컬럼명을 가진 DataFrame 형태로 변환하여 전달합니다.
            if not final_returns_for_chart_percent.empty:
                # Series를 DataFrame으로 변환. 인덱스는 섹터 이름, 값은 수익률
                chart_data = pd.DataFrame({
                    'Sector': final_returns_for_chart_percent.index.astype(str), # 인덱스를 문자열로 명시적 변환
                    'Return_Percent': final_returns_for_chart_percent.values.astype(float) # 값을 실수형으로 명시적 변환
                })
                
                # Streamlit의 bar_chart는 Altair를 래핑한 것이므로,
                # Altair 차트를 직접 구성하여 데이터 타입과 인코딩을 명시해줄 수 있습니다.
                # 이를 위해 altair 라이브러리를 import 합니다.
                import altair as alt # 여기에 추가 (코드 상단에 import altair as alt 추가 필요)

                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Sector:N', sort='-y', title='섹터'), # N: Nominal (명목형)
                    y=alt.Y('Return_Percent:Q', title='수익률 (%)') # Q: Quantitative (정량형)
                ).properties(
                    title='분석 기간 동안 최종 수익률'
                )
                st.altair_chart(chart, use_container_width=True) # Altair 차트 렌더링
                
            else:
                st.warning("선택된 기간 내 유효한 ETF 수익률 데이터가 없어 최종 수익률을 계산할 수 없습니다.")
            # --- 수정 끝 ---

            # Display data (formatted string for dataframe)
            st.write("📈 **분석 기간 동안 최종 수익률 (%):** (데이터프레임 표)")
            st.dataframe(final_returns_for_chart_percent.apply(lambda x: f"{x:.2f}%")) # % 형식으로 표시


        #     # --- ETF 목록 확인 기능 ---
        #     st.markdown("---")
        #     st.subheader("🔍 사용 가능한 한국 ETF 목록 확인 (FinanceDataReader)")
        #     st.info("FinanceDataReader가 제공하는 모든 한국 ETF 목록을 확인할 수 있습니다. "
        #             "이를 참고하여 '분석할 산업/섹터를 선택하세요' 부분의 ETF를 직접 변경할 수 있습니다.")
            
        #     if st.button("한국 ETF 목록 불러오기", key='load_etf_list'):
        #         try:
        #             etf_list_df = fdr.StockListing('ETF/KR')
        #             st.dataframe(etf_list_df[['Symbol', 'Name', 'Sector', 'ListingDate']].head(200)) # 상위 200개만
        #             st.download_button(
        #                 label="전체 ETF 목록 CSV 다운로드",
        #                 data=etf_list_df.to_csv(index=False).encode('utf-8'),
        #                 file_name="korean_etf_list.csv",
        #                 mime="text/csv",
        #             )
        #             st.markdown("---")
        #             st.write("`Symbol` 컬럼의 값을 복사하여 위에 정의된 `sector_etfs` 딕셔너리에 추가하거나 수정할 수 있습니다.")
        #             st.write("예시: `\"새로운섹터\": {\"티커\": \"새로운티커\", \"이름\": \"새로운ETF이름\"}`")

        #         except Exception as e:
        #             st.error(f"🚨 한국 ETF 목록을 불러오는 중 오류 발생: {e}")
        # else:
        #     st.error("선택된 섹터에 대한 유효한 ETF 데이터가 없어 분석을 수행할 수 없습니다. 섹터 정의와 기간 설정을 확인해주세요.")

st.markdown("---")
st.caption("면책 조항: 이 도구는 정보 제공 목적으로만 사용되며 투자 조언을 제공하지 않습니다. 과거의 성과는 미래의 결과를 보장하지 않습니다. ETF의 상장일이 선택된 분석 시작일보다 늦을 경우 해당 ETF는 분석 기간 동안의 전체 데이터를 포함하지 못할 수 있습니다.")
