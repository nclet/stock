import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import shap
import plotly.express as px
import re
import urllib.parse
import time
from requests.exceptions import ConnectionError, Timeout
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ======================================================
# 페이지 설정
# ======================================================
st.set_page_config(page_title="🏘️ 지역별 부동산 가격 예측", layout="wide")
st.title("🏘️ 지역별 아파트 가격 추세 예측 (3개월)")
st.markdown("**국토교통부 실거래 + 네이버 뉴스 감성 + LightGBM**")
##=============
#데이터 기간설정
#==============
def make_yyyymm_list(start_ym: str, end_ym: str):
    """
    start_ym, end_ym: 'YYYY-MM' 형식
    return: ['YYYYMM', 'YYYYMM', ...]
    """
    dates = pd.date_range(
        start=f"{start_ym}-01",
        end=f"{end_ym}-01",
        freq="MS"  # Month Start
    )
    return [d.strftime("%Y%m") for d in dates]
# ======================================================
# 감성 분석 모델
# ======================================================
@st.cache_resource
def load_sentiment_model():
    model_name = "snunlp/KR-FinBert-SC"

    # ✅ HuggingFace 토큰 (최상위 key: HF_TOKEN)
    hf_token = st.secrets.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=hf_token
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def analyze_sentiment(text):
    if not text:
        return 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    return probs[2].item() - probs[0].item()   # 긍정 - 부정
#
#세션생성
#
def create_session():
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


# ======================================================
# 네이버 뉴스 API
# ======================================================
def get_naver_news(query):
    cid = st.secrets["naver"]["client_id"]
    cs = st.secrets["naver"]["client_secret"]

    enc_query = urllib.parse.quote(query)
    url = f"http://openapi.naver.com/v1/search/news.json?query={enc_query}&display=100&sort=date"
    headers = {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": cs
    }

    r = requests.get(url, headers=headers)
    items = r.json().get("items", [])

    rows = []
    for it in items:
        title = re.sub("<.*?>", "", it["title"])
        pub_date = datetime.strptime(
            it["pubDate"], "%a, %d %b %Y %H:%M:%S %z"
        ).date()
        rows.append({"date": pub_date, "title": title})

    return pd.DataFrame(rows)

# ======================================================
# 국토교통부 실거래 API (MOLIT_KEY 적용)
# =====================================================

# @st.cache_data
# def load_real_estate_data(lawd_cd, start_ym, end_ym):

#     # 1️⃣ 키는 절대 quote 하지 마세요
#     service_key = st.secrets.get("MOLIT_KEY", None)
#     if service_key is None:
#         st.error("❌ MOLIT_KEY가 secrets에 없습니다.")
#         return pd.DataFrame()
        
#     decoded_key = requests.utils.unquote(service_key)
    
#     months = make_yyyymm_list(start_ym, end_ym)
#     rows = []

#     BASE_URL = (
#         "https://openapi.molit.go.kr/"
#         "OpenAPI_ToolInstallPackage/service/rest/"
#         "RTMSOBJSvc/getRTMSDataSvcAptTrade"
#     )

#     session = create_session()

#     for ym in months:
#         url = (
#             f"{BASE_URL}"
#             f"?serviceKey={service_key}"
#             f"&LAWD_CD={lawd_cd}"
#             f"&DEAL_YMD={ym}"
#             f"&numOfRows=1000"
#         )

#         try:
#             r = session.get(url, timeout=20)

#             soup = BeautifulSoup(r.text, "xml")

#             # 🔍 첫 달만 상태 출력
#             if ym == months[0]:
#                 header = soup.find("header")
#                 if header:
#                     st.subheader("🧪 국토부 API 상태")
#                     st.write(
#                         "resultCode:", header.find("resultCode").text,
#                         "resultMsg:", header.find("resultMsg").text
#                     )

#             items = soup.find_all("item")
#             if not items:
#                 continue

#             for it in items:
#                 try:
#                     rows.append({
#                         "price": int(it.find("거래금액").text.replace(",", "").strip()),
#                         "year": int(it.find("년").text.strip()),
#                         "month": int(it.find("월").text.strip())
#                     })
#                 except Exception:
#                     continue

#         except Exception as e:
#             st.warning(f"⚠️ {ym} 로드 중 오류: {e}")
#             continue

#         time.sleep(0.5)

#     if not rows:
#         return pd.DataFrame()

#     df = pd.DataFrame(rows)
#     df["date"] = pd.to_datetime(
#         df["year"].astype(str)
#         + "-"
#         + df["month"].astype(str).str.zfill(2)
#         + "-01"
#     )

#     return df
@st.cache_data
def load_real_estate_data(lawd_cd, start_ym, end_ym):
    clean_lawd_cd = str(lawd_cd)[:5]
    service_key = st.secrets.get("MOLIT_KEY", None)
    decoded_key = requests.utils.unquote(service_key)
    months = make_yyyymm_list(start_ym, end_ym)
    rows = []

    # ✅ 새로운 엔드포인트 주소 (최신 버전)
    # 기존: http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade
    # 신규 시도: 아래 주소는 국토부에서 제공하는 공식 최신 가이드 기준입니다.
    BASE_URL = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade"
    # 만약 위 주소도 안된다면 :8081을 빼고 시도하세요.
    
    for ym in months:
        # API 가이드에 맞춘 정확한 파라미터 전달
        params = {
            'serviceKey': decoded_key,
            'LAWD_CD': clean_lawd_cd,
            'DEAL_YMD': ym
        }

        try:
            # 브라우저인 것처럼 속이기 위한 User-Agent 추가
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            r = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
            
            # 응답 확인
            if "Connection refused" in r.text or r.status_code != 200:
                # 만약 8081 포트가 막혔다면 일반 포트로 재시도
                ALT_URL = BASE_URL.replace(":8081", "")
                r = requests.get(ALT_URL, params=params, headers=headers, timeout=15)

            soup = BeautifulSoup(r.text, "xml")
            items = soup.find_all("item")
            
            if not items:
                # 데이터가 없는 것인지, 키 에러인지 출력
                header = soup.find("header")
                if header and header.find("resultCode").text != "00":
                    st.error(f"❌ {ym} 에러: {header.find('resultMsg').text}")
                continue

            for it in items:
                rows.append({
                    "price": int(it.find("거래금액").text.replace(",", "").strip()),
                    "year": int(it.find("년").text.strip()),
                    "month": int(it.find("월").text.strip())
                })
            
            # 한 달 성공 시 짧은 휴식
            time.sleep(0.5)

        except Exception as e:
            st.warning(f"⚠️ {ym} 호출 실패. 네트워크 환경을 확인하세요: {e}")
            break # 계속 실패하면 중단

    return pd.DataFrame(rows)
# ======================================================
# UI
# ======================================================
col1, col2, col3 = st.columns(3)

with col1:
    lawd_cd = st.text_input("📍 법정동 코드 (예: 강남구 11680)", "11680")
with col2:
    start_ym = st.text_input("시작 월 (YYYY-MM)", "2020-01")
with col3:
    end_ym = st.text_input("종료 월 (YYYY-MM)", "2024-12")

news_query = st.text_input(
    "📰 뉴스 키워드",
    "강남 아파트|서울 집값|금리 인상|부동산 규제"
)

# ======================================================
# 실행
# ======================================================
if st.button("🚀 예측 실행", use_container_width=True):

    with st.spinner("📦 국토교통부 실거래 데이터 수집 중..."):
        df_raw = load_real_estate_data(lawd_cd, start_ym, end_ym)

    if df_raw.empty:
        st.error("실거래 데이터가 없습니다.")
        st.stop()

    # 월별 집계
    df_month = df_raw.groupby("date").agg(
        price_mean=("price", "mean"),
        volume=("price", "count")
    )

    # 타겟 생성 (3개월 변화율)
    df_month["price_change_3m"] = df_month["price_mean"].pct_change(3) * 100

    # 뉴스 감성
    news_df = get_naver_news(news_query)

    if not news_df.empty:
        news_df["sentiment"] = news_df["title"].apply(analyze_sentiment)
        news_daily = news_df.groupby("date").agg(
            Sentiment_Score=("sentiment", "mean"),
            News_Count=("title", "count")
        )
        df_month = df_month.merge(
            news_daily, left_index=True, right_index=True, how="left"
        )

    df_month = df_month.fillna(0).dropna()

    # 모델 입력
    FEATURES = [
        "price_mean",
        "volume",
        "Sentiment_Score",
        "News_Count"
    ]
    TARGET = "price_change_3m"

    X = df_month[FEATURES]
    y = df_month[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31,
        random_state=42
    )
    model.fit(X, y)

    # 예측
    pred = model.predict(X.iloc[[-1]])[0]

    st.metric(
        "📈 향후 3개월 예상 가격 변화율",
        f"{pred:+.2f}%"
    )

    # 시각화
    fig_price = px.line(
        df_month,
        y="price_mean",
        title="월별 평균 아파트 실거래가"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # SHAP 해석
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[[-1]])

    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP Value": shap_values[0]
    }).sort_values("SHAP Value", key=abs, ascending=False)

    st.subheader("🔍 예측 기여 요인 (SHAP)")
    st.dataframe(shap_df, use_container_width=True)
