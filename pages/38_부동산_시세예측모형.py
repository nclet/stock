import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ===============================
# 기본 설정
# ===============================
st.set_page_config(page_title="📊 국내 부동산 가격 예측", layout="wide")

TARGET_MONTH = 3

# ===============================
# API KEY
# ===============================
MOLIT_KEY = st.secrets["MOLIT_KEY"]
NAVER_ID = st.secrets["naver"]["client_id"]
NAVER_SECRET = st.secrets["naver"]["client_secret"]

# ===============================
# 1. 국토교통부 실거래가
# ===============================
@st.cache_data(ttl=60*60*24)
def load_real_estate_data(region_code, start_year=2018):
    rows = []
    today = datetime.date.today()

    for year in range(start_year, today.year + 1):
        for month in range(1, 13):
            ym = f"{year}{month:02d}"
            url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade"
            params = {
                "serviceKey": MOLIT_KEY,
                "LAWD_CD": region_code,
                "DEAL_YMD": ym,
                "numOfRows": 1000,
                "pageNo": 1
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    continue

                from xml.etree import ElementTree
                root = ElementTree.fromstring(r.text)
                for item in root.iter("item"):
                    price = int(item.findtext("거래금액").replace(",", ""))
                    year_ = int(item.findtext("년"))
                    month_ = int(item.findtext("월"))
                    rows.append({
                        "date": pd.to_datetime(f"{year_}-{month_:02d}-01"),
                        "price": price
                    })
            except:
                continue

    df = pd.DataFrame(rows)
    df = df.groupby("date")["price"].mean().reset_index()
    return df

# ===============================
# 2. 네이버 뉴스 감성 점수
# ===============================
def get_news_sentiment(query):
    headers = {
        "X-Naver-Client-Id": NAVER_ID,
        "X-Naver-Client-Secret": NAVER_SECRET
    }
    url = "https://openapi.naver.com/v1/search/news.json"
    params = {
        "query": query,
        "display": 100,
        "sort": "date"
    }

    try:
        r = requests.get(url, headers=headers, params=params)
        items = r.json().get("items", [])
    except:
        return 0

    positive_words = ["상승", "호재", "완화", "개선", "회복"]
    negative_words = ["하락", "규제", "침체", "악화", "부진"]

    score = 0
    for item in items:
        title = item["title"]
        for p in positive_words:
            if p in title:
                score += 1
        for n in negative_words:
            if n in title:
                score -= 1

    return score / max(len(items), 1)

# ===============================
# 3. Feature Engineering
# ===============================
def create_features(df, sentiment):
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["ma_3"] = df["price"].rolling(3).mean()
    df["ma_6"] = df["price"].rolling(6).mean()
    df["vol"] = df["return"].rolling(3).std()
    df["sentiment"] = sentiment
    df["target"] = df["price"].shift(-TARGET_MONTH)
    return df.dropna()

# ===============================
# 4. 모델 학습
# ===============================
def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_scaled, y)
    return model, scaler

# ===============================
# 5. 예측
# ===============================
def predict_future(model, scaler, df):
    last = df.drop("target", axis=1).iloc[-1:]
    last_scaled = scaler.transform(last)
    return model.predict(last_scaled)[0]

# ===============================
# 6. Streamlit UI
# ===============================
st.title("🏘️ 국내 부동산 가격 예측 시스템")

region = st.selectbox(
    "지역 선택",
    {
        "서울 강남구": "11680",
        "서울 송파구": "11710",
        "서울 마포구": "11440"
    }
)

run = st.button("📈 예측 실행")

if run:
    with st.spinner("📡 데이터 수집 중..."):
        df_price = load_real_estate_data(region)
        sentiment = get_news_sentiment("부동산 가격")

    df_feat = create_features(df_price, sentiment)
    model, scaler = train_model(df_feat)
    prediction = predict_future(model, scaler, df_feat)

    st.subheader("📊 결과")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_price["date"],
        y=df_price["price"],
        name="실제 가격"
    ))

    fig.add_trace(go.Scatter(
        x=[df_price["date"].iloc[-1] + pd.DateOffset(months=TARGET_MONTH)],
        y=[prediction],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="예측 가격"
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"🔮 {TARGET_MONTH}개월 후 예상 가격: {prediction:,.0f} 만원")
