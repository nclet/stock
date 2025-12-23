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

# ======================================================
# 페이지 설정
# ======================================================
st.set_page_config(page_title="🏘️ 지역별 부동산 가격 예측", layout="wide")
st.title("🏘️ 지역별 아파트 가격 추세 예측 (3개월)")
st.markdown("**국토교통부 실거래 + 네이버 뉴스 감성 + LightGBM**")

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

# ======================================================
# 네이버 뉴스 API
# ======================================================
def get_naver_news(query):
    cid = st.secrets["naver"]["client_id"]
    cs = st.secrets["naver"]["client_secret"]

    enc_query = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display=100&sort=date"
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
# ======================================================
@st.cache_data
def load_real_estate_data(lawd_cd, start_ym, end_ym):

    service_key = st.secrets.get("MOLIT_KEY", None)
    if service_key is None:
        st.error("❌ MOLIT_KEY가 secrets에 없습니다.")
        return pd.DataFrame()

    months = pd.period_range(start=start_ym, end=end_ym, freq="M").astype(str)
    rows = []

    BASE_URL = (
        "https://openapi.molit.go.kr/"
        "OpenAPI_ToolInstallPackage/service/rest/"
        "RTMSOBJSvc/getRTMSDataSvcAptTrade"
    )

    for ym in months:
        params = {
            "serviceKey": service_key,
            "LAWD_CD": lawd_cd,
            "DEAL_YMD": ym.replace("-", ""),
            "numOfRows": 1000
        }

        success = False

        for attempt in range(3):  # ✅ 최대 3회 재시도
            try:
                r = requests.get(
                    BASE_URL,
                    params=params,
                    timeout=10
                )

                if r.status_code != 200:
                    time.sleep(1)
                    continue

                soup = BeautifulSoup(r.text, "xml")

                for it in soup.find_all("item"):
                    try:
                        rows.append({
                            "price": int(it.거래금액.text.replace(",", "")),
                            "year": int(it.년.text),
                            "month": int(it.월.text)
                        })
                    except:
                        continue

                success = True
                break  # 성공하면 retry 탈출

            except (ConnectionError, Timeout):
                time.sleep(2)  # 서버 쉬게 해줌

        if not success:
            # ❗ 이 달 데이터만 스킵
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str)
    )

    return df
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
