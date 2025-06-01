import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import traceback

st.set_page_config(layout="wide")
st.title("📊 PER / PBR 기반 수익률 분석")
st.markdown("특정 PER/PBR 범위에 해당하는 종목들의 과거 수익률을 분석합니다.")

# --------------------------------------------
# 함수 정의 (필요한 경우 여기에 배치, 또는 utils.py 등으로 분리 가능)
# 현재 코드에서는 파일 내에 직접 포함
# --------------------------------------------
current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, '..')
per_pbr_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')
# --------------------------------------------

try:
    df_fundamental = pd.read_csv(per_pbr_file_path)
    df_fundamental.columns = df_fundamental.columns.str.strip()
    df_fundamental['Date'] = pd.to_datetime(df_fundamental['Date'])
    df_fundamental = df_fundamental.dropna(subset=['PER', 'PBR', 'Close'])
    st.success(f"✅ PER/PBR 데이터를 성공적으로 불러왔습니다. 기간을 설정한 뒤, '전략 분석 시작'을 눌러주세요.")
    
    # 날짜 선택 (기존 코드와 동일)
    min_date_data = df_fundamental['Date'].min().date()
    max_date_data = df_fundamental['Date'].max().date()

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        per_pbr_start = st.date_input("시작일", min_value=min_date_data, max_value=max_date_data, value=min_date_data)
    with col_date2:
        per_pbr_end = st.date_input("종료일", min_value=per_pbr_start, max_value=max_date_data, value=max_date_data)

    if per_pbr_start >= per_pbr_end:
        st.error("종료 날짜는 시작 날짜보다 미래여야 합니다.")
        st.stop()

    # --- PER 범위 선택 (수정 부분) ---
    st.write("### 📈 PER 범위 선택")

    # 세션 상태 초기화 (PER)
    if 'per_min_value' not in st.session_state:
        st.session_state.per_min_value = 0.0
    if 'per_max_value' not in st.session_state:
        st.session_state.per_max_value = 15.0

    # PER 슬라이더의 on_change 콜백 함수
    def update_per_from_slider():
        st.session_state.per_min_value = st.session_state.per_slider[0]
        st.session_state.per_max_value = st.session_state.per_slider[1]

    # PER 숫자 입력의 on_change 콜백 함수 (min)
    def update_per_min_from_number():
        # 숫자 입력이 슬라이더 범위를 벗어나지 않도록 조정
        if st.session_state.per_min_input > st.session_state.per_max_value:
            st.session_state.per_max_value = st.session_state.per_min_input
        st.session_state.per_slider = (st.session_state.per_min_input, st.session_state.per_max_value)

    # PER 숫자 입력의 on_change 콜백 함수 (max)
    def update_per_max_from_number():
        # 숫자 입력이 슬라이더 범위를 벗어나지 않도록 조정
        if st.session_state.per_max_input < st.session_state.per_min_value:
            st.session_state.per_min_value = st.session_state.per_max_input
        st.session_state.per_slider = (st.session_state.per_min_value, st.session_state.per_max_input)


    col1, col2 = st.columns(2)
    with col1:
        per_min = st.number_input(
            "최소 PER", 
            min_value=0.0, 
            max_value=100.0, 
            value=st.session_state.per_min_value, 
            step=0.1, 
            key="per_min_input", 
            on_change=update_per_min_from_number
        )
    with col2:
        per_max = st.number_input(
            "최대 PER", 
            min_value=0.0, 
            max_value=100.0, 
            value=st.session_state.per_max_value, 
            step=0.1, 
            key="per_max_input",
            on_change=update_per_max_from_number
        )

    # 슬라이더의 value는 튜플 (min, max) 형태로 전달
    per_slider_range = st.slider(
        "PER 범위 슬라이더", 
        0.0, 100.0, 
        value=(st.session_state.per_min_value, st.session_state.per_max_value), 
        step=0.1, 
        key="per_slider", 
        on_change=update_per_from_slider
    )

    # 최종적으로 사용할 PER 값은 세션 상태에서 가져옵니다.
    per_min_final = st.session_state.per_min_value
    per_max_final = st.session_state.per_max_value

    # --- PBR 범위 선택 (수정 부분) ---
    st.write("### 📉 PBR 범위 선택")

    # 세션 상태 초기화 (PBR)
    if 'pbr_min_value' not in st.session_state:
        st.session_state.pbr_min_value = 0.0
    if 'pbr_max_value' not in st.session_state:
        st.session_state.pbr_max_value = 2.0

    # PBR 슬라이더의 on_change 콜백 함수
    def update_pbr_from_slider():
        st.session_state.pbr_min_value = st.session_state.pbr_slider[0]
        st.session_state.pbr_max_value = st.session_state.pbr_slider[1]

    # PBR 숫자 입력의 on_change 콜백 함수 (min)
    def update_pbr_min_from_number():
        if st.session_state.pbr_min_input > st.session_state.pbr_max_value:
            st.session_state.pbr_max_value = st.session_state.pbr_min_input
        st.session_state.pbr_slider = (st.session_state.pbr_min_input, st.session_state.pbr_max_value)

    # PBR 숫자 입력의 on_change 콜백 함수 (max)
    def update_pbr_max_from_number():
        if st.session_state.pbr_max_input < st.session_state.pbr_min_value:
            st.session_state.pbr_min_value = st.session_state.pbr_max_input
        st.session_state.pbr_slider = (st.session_state.pbr_min_value, st.session_state.pbr_max_value)


    col3, col4 = st.columns(2)
    with col3:
        pbr_min = st.number_input(
            "최소 PBR", 
            min_value=0.0, 
            max_value=10.0, 
            value=st.session_state.pbr_min_value, 
            step=0.1, 
            key="pbr_min_input",
            on_change=update_pbr_min_from_number
        )
    with col4:
        pbr_max = st.number_input(
            "최대 PBR", 
            min_value=0.0, 
            max_value=10.0, 
            value=st.session_state.pbr_max_value, 
            step=0.1, 
            key="pbr_max_input",
            on_change=update_pbr_max_from_number
        )

    pbr_slider_range = st.slider(
        "PBR 범위 슬라이더", 
        0.0, 10.0, 
        value=(st.session_state.pbr_min_value, st.session_state.pbr_max_value), 
        step=0.1, 
        key="pbr_slider",
        on_change=update_pbr_from_slider
    )

    # 최종적으로 사용할 PBR 값은 세션 상태에서 가져옵니다.
    pbr_min_final = st.session_state.pbr_min_value
    pbr_max_final = st.session_state.pbr_max_value
    
    if st.button("📊 전략 분석 시작"):
        # 필터링
        # 이제 per_min, per_max 대신 per_min_final, per_max_final을 사용합니다.
        df_filtered = df_fundamental[
            (df_fundamental['PER'] >= per_min_final) & (df_fundamental['PER'] <= per_max_final) &
            (df_fundamental['PBR'] >= pbr_min_final) & (df_fundamental['PBR'] <= pbr_max_final) &
            (df_fundamental['Date'] >= pd.to_datetime(per_pbr_start)) &
            (df_fundamental['Date'] <= pd.to_datetime(per_pbr_end))
        ]

        if df_filtered.empty:
            st.warning("선택한 조건에 해당하는 종목이 없습니다. 조건을 다시 설정해주세요.")
        else:
            # pivot_table을 사용하여 날짜별 종목별 종가 데이터프레임 생성
            df_pivot = df_filtered.pivot_table(index='Date', columns='Code', values='Close')
            
            # 일간 수익률 계산
            # PER/PBR 데이터가 월간이라면 월간 수익률 계산으로 변경 필요
            # 여기서는 일간 종가를 기준으로 일간 수익률을 계산합니다.
            df_return = df_pivot.pct_change().fillna(0)
            
            # 누적 수익률 계산 (1 + 일간 수익률)의 누적 곱
            cumulative_return = (1 + df_return).cumprod() - 1 # 초기 100% 수익률을 0%로 맞추기 위해 -1

            # 최종 수익률
            final_return = cumulative_return.iloc[-1]
            
            # 수익률 상위 10개 종목 추출
            top_codes = final_return.sort_values(ascending=False).head(10).index
            
            # 종목 코드와 이름을 매핑
            code_name_map = df_fundamental.drop_duplicates('Code').set_index('Code')['Name'].to_dict()
            top_names = [code_name_map.get(code, code) for code in top_codes]

            st.subheader("🏆 수익률 상위 10개 종목")
            st.dataframe(pd.DataFrame({
                '종목코드': top_codes,
                '종목명': top_names,
                '수익률(%)': (final_return[top_codes] * 100).round(2).values
            }).reset_index(drop=True))

            st.subheader("📈 상위 10개 종목 누적 수익률 차트")
            
            # matplotlib으로 차트 생성 (Streamlit의 st.line_chart는 범례가 제한적일 수 있음)
            fig, ax = plt.subplots(figsize=(12, 6))
            for code in top_codes:
                ax.plot(cumulative_return.index, cumulative_return[code], label=code_name_map.get(code, code))
            
            ax.set_title(f"PER/PBR 전략 누적 수익률 ({per_pbr_start} ~ {per_pbr_end})")
            ax.set_xlabel("날짜")
            ax.set_ylabel("누적 수익률")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 범례를 차트 밖에 배치
            ax.grid(True)
            plt.tight_layout() # 그래프 요소가 겹치지 않도록 조정
            st.pyplot(fig)


except FileNotFoundError:
    st.error(f"❌ PER/PBR 데이터 파일 '{per_pbr_file_path}'이(가) 현재 디렉토리에 존재하지 않습니다. 파일을 확인해주세요.")
except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
    st.error(traceback.format_exc()) # 오류 상세 내용을 출력하여 디버깅에 도움을 줍니다.

st.markdown("---")
st.write("### 참고")
st.write("""
- **PER/PBR:** 기업의 주가수익비율(PER)과 주가순자산비율(PBR)을 기준으로 저평가된 종목을 선별합니다.
- **백테스팅 모델의 한계:** 거래 수수료, 슬리피지 등을 고려하지 않은 단순 시뮬레이션입니다.
- **※추후 PSR, PCR, ROE, F-스코어 등의 팩터를 추가할 예정입니다.
""")


############################################
# # streamlit_test/pages/1_PER_PBR_전략.py
# import streamlit as st
# import pandas as pd
# import datetime
# import matplotlib.pyplot as plt
# import os
# import traceback

# st.set_page_config(layout="wide")
# st.title("📊 PER / PBR 기반 수익률 분석")
# st.markdown("특정 PER/PBR 범위에 해당하는 종목들의 과거 수익률을 분석합니다.")

# # --------------------------------------------
# # 함수 정의 (필요한 경우 여기에 배치, 또는 utils.py 등으로 분리 가능)
# # 현재 코드에서는 파일 내에 직접 포함
# # --------------------------------------------
# # per_pbr_file = 'merged_data_monthly_per_pbr.csv' # 이 파일은 streamlit_test 폴더에 있어야 함

# # 현재 스크립트 파일(1_PER_PBR_전략.py)의 디렉토리 경로를 가져옵니다.
# current_dir = os.path.dirname(__file__)

# # 현재 디렉토리에서 상위 디렉토리(stock/ 루트 폴더)로 이동합니다.
# # '..'는 상위 디렉토리를 의미합니다.
# root_dir = os.path.join(current_dir, '..')

# # 루트 디렉토리 안에 있는 CSV 파일의 전체 경로를 만듭니다.
# per_pbr_file_path = os.path.join(root_dir, 'merged_data_monthly_per_pbr.csv')
# # --------------------------------------------

# try:
#     # 수정된 경로를 사용하여 파일을 읽습니다.
#     df_fundamental = pd.read_csv(per_pbr_file_path) # <-- 여기에 per_pbr_file_path를 사용합니다.
#     df_fundamental.columns = df_fundamental.columns.str.strip()
#     df_fundamental['Date'] = pd.to_datetime(df_fundamental['Date'])
#     df_fundamental = df_fundamental.dropna(subset=['PER', 'PBR', 'Close'])
#     # 성공 메시지에도 수정된 경로를 사용하도록 변경합니다.
#     st.success(f"✅ PER/PBR 데이터 로드에 성공했습니다. ")
    
#     # 날짜 선택
#     min_date_data = df_fundamental['Date'].min().date()
#     max_date_data = df_fundamental['Date'].max().date()

#     col_date1, col_date2 = st.columns(2)
#     with col_date1:
#         per_pbr_start = st.date_input("시작일", min_value=min_date_data, max_value=max_date_data, value=min_date_data)
#     with col_date2:
#         per_pbr_end = st.date_input("종료일", min_value=per_pbr_start, max_value=max_date_data, value=max_date_data)

#     if per_pbr_start >= per_pbr_end:
#         st.error("종료 날짜는 시작 날짜보다 미래여야 합니다.")
#         st.stop()

#     # PER 입력
#     st.write("### 📈 PER 범위 선택")
#     col1, col2 = st.columns(2)
#     with col1:
#         per_min = st.number_input("최소 PER", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="per_min_input")
#     with col2:
#         per_max = st.number_input("최대 PER", min_value=0.0, max_value=100.0, value=15.0, step=0.1, key="per_max_input")
#     st.slider("PER 범위 슬라이더", 0.0, 100.0, (per_min, per_max), disabled=True, key="per_slider") # 슬라이더는 인풋 값을 반영만

#     # PBR 입력
#     st.write("### 📉 PBR 범위 선택")
#     col3, col4 = st.columns(2)
#     with col3:
#         pbr_min = st.number_input("최소 PBR", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="pbr_min_input")
#     with col4:
#         pbr_max = st.number_input("최대 PBR", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="pbr_max_input")
#     st.slider("PBR 범위 슬라이더", 0.0, 10.0, (pbr_min, pbr_max), disabled=True, key="pbr_slider") # 슬라이더는 인풋 값을 반영만
    
#     if st.button("📊 전략 분석 시작"):
#         # 필터링
#         df_filtered = df_fundamental[
#             (df_fundamental['PER'] >= per_min) & (df_fundamental['PER'] <= per_max) &
#             (df_fundamental['PBR'] >= pbr_min) & (df_fundamental['PBR'] <= pbr_max) &
#             (df_fundamental['Date'] >= pd.to_datetime(per_pbr_start)) &
#             (df_fundamental['Date'] <= pd.to_datetime(per_pbr_end))
#         ]

#         if df_filtered.empty:
#             st.warning("선택한 조건에 해당하는 종목이 없습니다. 조건을 다시 설정해주세요.")
#         else:
#             # pivot_table을 사용하여 날짜별 종목별 종가 데이터프레임 생성
#             df_pivot = df_filtered.pivot_table(index='Date', columns='Code', values='Close')
            
#             # 일간 수익률 계산
#             # PER/PBR 데이터가 월간이라면 월간 수익률 계산으로 변경 필요
#             # 여기서는 일간 종가를 기준으로 일간 수익률을 계산합니다.
#             df_return = df_pivot.pct_change().fillna(0)
            
#             # 누적 수익률 계산 (1 + 일간 수익률)의 누적 곱
#             cumulative_return = (1 + df_return).cumprod() - 1 # 초기 100% 수익률을 0%로 맞추기 위해 -1

#             # 최종 수익률
#             final_return = cumulative_return.iloc[-1]
            
#             # 수익률 상위 10개 종목 추출
#             top_codes = final_return.sort_values(ascending=False).head(10).index
            
#             # 종목 코드와 이름을 매핑
#             code_name_map = df_fundamental.drop_duplicates('Code').set_index('Code')['Name'].to_dict()
#             top_names = [code_name_map.get(code, code) for code in top_codes]

#             st.subheader("🏆 수익률 상위 10개 종목")
#             st.dataframe(pd.DataFrame({
#                 '종목코드': top_codes,
#                 '종목명': top_names,
#                 '수익률(%)': (final_return[top_codes] * 100).round(2).values
#             }).reset_index(drop=True))

#             st.subheader("📈 상위 10개 종목 누적 수익률 차트")
            
#             # matplotlib으로 차트 생성 (Streamlit의 st.line_chart는 범례가 제한적일 수 있음)
#             fig, ax = plt.subplots(figsize=(12, 6))
#             for code in top_codes:
#                 ax.plot(cumulative_return.index, cumulative_return[code], label=code_name_map.get(code, code))
            
#             ax.set_title(f"PER/PBR 전략 누적 수익률 ({per_pbr_start} ~ {per_pbr_end})")
#             ax.set_xlabel("날짜")
#             ax.set_ylabel("누적 수익률")
#             ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 범례를 차트 밖에 배치
#             ax.grid(True)
#             plt.tight_layout() # 그래프 요소가 겹치지 않도록 조정
#             st.pyplot(fig)


# except FileNotFoundError:
#     st.error(f"❌ PER/PBR 데이터 파일 '{per_pbr_file}'이(가) 현재 디렉토리에 존재하지 않습니다. 파일을 확인해주세요.")
# except Exception as e:
#     st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")

# st.markdown("---")
# st.write("### 참고")
# st.write("""

# - **PER/PBR:** 기업의 주가수익비율(PER)과 주가순자산비율(PBR)을 기준으로 저평가된 종목을 선별합니다.
# - **백테스팅 모델의 한계:** 거래 수수료, 슬리피지 등을 고려하지 않은 단순 시뮬레이션입니다.
# """)
