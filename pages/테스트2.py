import streamlit as st

# 페이지 제목 설정
st.title("메뉴 분류를 이용한 Streamlit 웹페이지")
st.write("사이드바에서 대분류와 소분류를 선택하여 페이지를 탐색해 보세요.")

# --- 수평선 ---
st.markdown("---")

# 사이드바에 대분류 메뉴 추가
with st.sidebar:
    st.header("메인 메뉴")
    main_category = st.selectbox(
        "대분류 선택",
        ["소개", "데이터 분석", "시각화", "설정"]
    )

# 선택된 대분류에 따라 다른 소분류 메뉴와 콘텐츠를 표시
if main_category == "소개":
    st.header("소개 페이지")
    st.write("이 페이지는 애플리케이션의 기본적인 정보를 소개합니다.")
    st.write("사이드바에서 다른 메뉴를 선택하여 기능들을 확인해 보세요.")

elif main_category == "데이터 분석":
    with st.sidebar:
        st.header("데이터 분석 메뉴")
        sub_category = st.selectbox(
            "소분류 선택",
            ["요약 통계", "데이터 필터링"]
        )

    if sub_category == "요약 통계":
        st.header("요약 통계")
        st.write("데이터의 기본적인 통계 정보를 보여주는 페이지입니다.")
        st.write("예: 평균, 중앙값, 표준편차 등")

    elif sub_category == "데이터 필터링":
        st.header("데이터 필터링")
        st.write("조건에 따라 데이터를 필터링하는 기능을 제공합니다.")
        st.write("예: 특정 날짜 범위, 값 범위 등")
        
elif main_category == "시각화":
    with st.sidebar:
        st.header("시각화 메뉴")
        sub_category = st.selectbox(
            "소분류 선택",
            ["막대 그래프", "산점도", "선 그래프"]
        )
    
    if sub_category == "막대 그래프":
        st.header("막대 그래프")
        st.write("데이터를 막대 그래프로 시각화하는 페이지입니다.")

    elif sub_category == "산점도":
        st.header("산점도")
        st.write("데이터 포인트 간의 관계를 산점도로 보여줍니다.")

    elif sub_category == "선 그래프":
        st.header("선 그래프")
        st.write("시간에 따른 데이터의 변화를 선 그래프로 시각화합니다.")

elif main_category == "설정":
    st.header("설정 페이지")
    st.write("애플리케이션의 다양한 설정을 변경할 수 있는 페이지입니다.")
    st.write("예: 테마 변경, 사용자 프로필 업데이트 등")

# --- 수평선 ---
st.markdown("---")

# 페이지 하단 정보
st.markdown("© 2024 Streamlit 메뉴 분류 예제")
