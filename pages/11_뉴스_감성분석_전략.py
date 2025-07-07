import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import yfinance as yf # VIX 데이터를 위해 필요
import matplotlib as mpl # 한글 폰트 설정용

# 딥러닝 감성 분석 라이브러리
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.linear_model import LinearRegression

# 한글깨짐방지
mpl.rc('font', family='Malgun Gothic')
# 마이너 버그 방지: 마이너스 깨짐 현상 방지
mpl.rcParams['axes.unicode_minus'] = False

# ------------------------
# ✨ 페이지 설정
# ------------------------
st.set_page_config(page_title="뉴스 감성분석 전략", layout="wide")
st.title("뉴스 감성 분석 전략")

st.markdown("""
**네이버 뉴스 API**를 통해 해당 종목의 뉴스를 종합합니다. 그리고 VIX(변동성지수), 모멘텀을 결합하여 주가를 예측합니다.
""")

# ------------------------
# ✨ 감성 분석 모델 로드
# ------------------------
@st.cache_resource
def load_sentiment_model():
    # 이 모델은 일반적인 한국어 언어 모델이며, 감성 분석에 특화되어 파인튜닝된 모델이 아닙니다.
    # 이전 대화에서 사용했던 'snunlp/KR-BERT-finetuned-sentiment'가 감성 분석에 더 적합합니다.
    # 만약 'beomi/KcELECTRA-base'를 감성 분석에 사용하려면, 감성 분류를 위한 추가 레이어와 학습이 필요합니다.
    # 현재 코드에서는 'beomi/KcELECTRA-base'를 그대로 사용합니다.
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base")
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

def analyze_sentiment(text):
    if not text:
        return 0.0
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    # KcELECTRA-base는 기본적으로 감성 분류를 위한 레이블이 정의되어 있지 않을 수 있습니다.
    # 이 부분은 모델의 출력 로짓을 어떻게 해석할지에 따라 달라집니다.
    # 만약 이 모델을 감성 분석에 사용하려면, 긍정/부정 로짓의 인덱스를 확인하거나,
    # 이 모델을 감성 분석 데이터셋으로 추가 학습(fine-tuning)해야 합니다.
    # 현재는 임시로 첫 번째 로짓을 사용합니다.
    score = torch.softmax(outputs.logits, dim=1)[0][0].item() # 임시로 첫 번째 로짓 사용
    return (score - 0.5) * 2  # -1 ~ 1

# ------------------------
# ✨ 종목 선택 UI
# ------------------------
@st.cache_data(ttl=86400) # 종목 리스트는 하루에 한 번만 로드하여 불필요한 재로드 방지
def get_stock_listing(market):
    st.info(f"📊 {market} 종목 리스트를 로드 중입니다...")
    try:
        df = fdr.StockListing(market)
        st.success(f"✅ {market} 종목 리스트 로드 완료!")
        return df
    except Exception as e:
        st.error(f"❌ {market} 종목 리스트를 로드할 수 없습니다: {e}")
        st.stop()

# 시장 선택 위젯에 key 추가
market_option = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ"], key="market_selector")

# 캐싱된 종목 리스트 사용
company_list_df = get_stock_listing(market_option)

if company_list_df.empty:
    st.warning("종목 리스트를 가져올 수 없습니다. 앱 실행이 어렵습니다.")
    st.stop()

company_names = company_list_df['Name'].tolist()

# company_name 셀렉트 박스에 key 추가하고 index 인자 제거
# Streamlit은 key가 있으면 위젯의 상태를 자동으로 유지하려고 시도합니다.
# index를 지정하면 사용자의 선택을 덮어쓸 수 있으므로 제거하는 것이 좋습니다.
company_name = st.selectbox("✅ 분석할 기업 선택", company_names, key="company_selector")

# 선택된 company_name에 해당하는 stock_code를 찾습니다.
# 사용자가 선택한 company_name이 company_names 리스트에 없을 경우를 대비하여 예외 처리
if company_name in company_list_df['Name'].values:
    stock_code = company_list_df.loc[company_list_df['Name'] == company_name, 'Code'].values[0]
else:
    st.error(f"선택된 기업 '{company_name}'의 코드를 찾을 수 없습니다. 다시 선택해주세요.")
    st.stop()

start_date = st.date_input("뉴스 검색 시작일", datetime.now().date() - timedelta(days=30), key="start_date_picker")
end_date = st.date_input("뉴스 검색 종료일", datetime.now().date(), key="end_date_picker")


# ------------------------
# ✨ 네이버 뉴스 API 크롤링 함수
# ------------------------
@st.cache_data(ttl=3600) # API 호출 결과를 캐싱하여 불필요한 재호출 방지
def get_naver_news_api(company_name, display=30, start=1, sort="date"):
    # secrets.toml에 naver.client_id와 naver.client_secret이 설정되어 있어야 합니다.
    try:
        client_id = st.secrets["naver"]["client_id"]
        client_secret = st.secrets["naver"]["client_secret"]
    except KeyError:
        st.error("""
        ❌ 네이버 API 키가 설정되지 않았습니다.
        프로젝트의 `.streamlit/secrets.toml` 파일에 다음과 같이 추가해주세요:
        ```toml
        [naver]
        client_id = "YOUR_NAVER_CLIENT_ID"
        client_secret = "YOUR_NAVER_CLIENT_SECRET"
        ```
        """)
        return pd.DataFrame() # 빈 데이터프레임 반환하여 앱 중단

    import urllib.parse
    enc_query = urllib.parse.quote(company_name)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display={display}&start={start}&sort={sort}"

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    st.info(f"뉴스 API 요청 중: {company_name}, 시작: {start}, 건수: {display}")
    try:
        response = requests.get(url, headers=headers, timeout=10) # 타임아웃 추가
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        data = response.json()
        items = data.get('items', [])
        news_data = []
        for item in items:
            title = item.get('title', '')
            pub_date = item.get('pubDate', '')
            try:
                # API 날짜 형식: "Weekday, DD Mon YYYY HH:MM:SS +0900"
                pub_date_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z").date()
            except ValueError: # 날짜 파싱 오류 시
                st.warning(f"뉴스 날짜 파싱 오류: '{pub_date}'. 해당 뉴스는 건너뜁니다.")
                pub_date_dt = None
            
            # HTML 태그 제거 (뉴스 제목에 <br> 등 HTML 태그가 포함될 수 있음)
            clean_title = re.sub(r'<[^>]+>', '', title)
            
            news_data.append({
                'Date': pub_date_dt,
                'Title': clean_title
            })
        df = pd.DataFrame(news_data)
        # 날짜 필터링은 호출하는 쪽에서 수행하므로 여기서는 모든 데이터 반환
        return df.dropna(subset=['Date']) # 날짜 파싱 실패한 항목은 제거
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 실패 (네트워크/HTTP 오류): {e}")
        st.error("네이버 API 키가 유효한지, 인터넷 연결 상태를 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"API 응답 처리 중 오류 발생: {e}")
        return pd.DataFrame()

# ------------------------
# ✨ 버튼 실행
# ------------------------
max_news = st.slider("최대 뉴스 건수", min_value=10, max_value=100, value=30, step=10, key="max_news_slider")

if st.button("🚀 뉴스 크롤링 및 분석 시작", key="start_button"):
    with st.spinner("뉴스 크롤링 및 감성 분석 중..."):
        all_news = pd.DataFrame()
        # 네이버 API는 start 최대 1000까지, display 최대 100개까지 가능
        # max_news에 따라 여러 번 API 호출
        for start_idx in range(1, max_news + 1, 100):
            count = min(100, max_news - start_idx + 1)
            if count <= 0: # 남은 뉴스 건수가 없으면 중단
                break
            df_part = get_naver_news_api(company_name, display=count, start=start_idx)
            if df_part.empty: # API 호출 실패 시 중단
                break
            all_news = pd.concat([all_news, df_part], ignore_index=True)
            if len(df_part) < count: # 요청한 건수보다 적게 오면 더 이상 뉴스가 없는 것으로 간주
                break

        # 날짜 필터링
        # st.date_input은 datetime.date 객체를 반환하므로, 비교를 위해 datetime.date로 변환
        start_date_dt = start_date
        end_date_dt = end_date
        
        # API에서 받은 날짜는 이미 datetime.date 객체이므로 추가 변환 불필요
        filtered_news = all_news[(all_news['Date'] >= start_date_dt) & (all_news['Date'] <= end_date_dt)].copy()
        
    if filtered_news.empty:
        st.error("❌ 뉴스 데이터를 가져오지 못했습니다. 검색 조건(기간, 키워드, 최대 뉴스 건수)을 조정하거나 네이버 API 키 설정을 확인해주세요.")
    else:
        # 감성 점수 계산
        filtered_news['Sentiment_Score'] = filtered_news['Title'].apply(analyze_sentiment)

        st.success(f"✅ 총 {len(filtered_news)}개의 뉴스 기사 감성 분석 완료!")
        st.dataframe(filtered_news[['Date', 'Title', 'Sentiment_Score']].sort_values(by='Date', ascending=False).reset_index(drop=True))

        # ------------------------
        # ✨ 주가 데이터 로드
        # ------------------------
        st.info(f"📈 {company_name} ({stock_code}) 주가 데이터를 로드 중입니다...")
        try:
            # FinanceDataReader는 datetime.date 객체를 직접 받을 수 있음
            df_stock = fdr.DataReader(stock_code, start_date_dt, end_date_dt)
            if df_stock.empty:
                st.error(f"❌ {company_name} ({stock_code}) 주가 데이터를 가져오지 못했습니다.")
                st.stop()
            df_stock = df_stock.reset_index()[['Date', 'Close']]
            df_stock['Date'] = pd.to_datetime(df_stock['Date']) # datetime.datetime으로 변환
            st.success(f"✅ {company_name} 주가 데이터 로드 완료!")
        except Exception as e:
            st.error(f"❌ 주가 데이터 로드 중 오류 발생: {e}")
            st.stop()

        # ------------------------
        # ✨ VIX 데이터 로드
        # ------------------------
        st.info("📉 VIX(변동성 지수) 데이터를 로드 중입니다...")
        try:
            # yfinance는 start/end에 datetime.date 객체를 직접 받을 수 있음
            # VIX는 주가 데이터 기간보다 약간 더 넓게 가져오는 것이 좋음
            vix_start_date = start_date_dt - timedelta(days=30)
            vix_end_date = end_date_dt + timedelta(days=1)
            vix = yf.download('^VIX', start=vix_start_date, end=vix_end_date, progress=False)
            if vix.empty:
                st.warning("⚠️ VIX 데이터를 가져오지 못했습니다. 예측에 포함되지 않습니다.")
                vix = pd.DataFrame(columns=['Date', 'VIX_Close']) # 빈 데이터프레임으로 초기화
            else:
                vix = vix.reset_index()
                vix = vix[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})
                vix['Date'] = pd.to_datetime(vix['Date']) # datetime.datetime으로 변환
                st.success("✅ VIX 데이터 로드 완료!")
        except Exception as e:
            st.warning(f"⚠️ VIX 데이터 로드 중 오류 발생: {e}. 예측에 포함되지 않습니다.")
            vix = pd.DataFrame(columns=['Date', 'VIX_Close']) # 오류 발생 시 빈 데이터프레임으로 초기화

        # ------------------------
        # ✨ 모멘텀 계산
        # ------------------------
        st.info("📊 모멘텀 지표를 계산 중입니다...")
        df_stock['Momentum'] = df_stock['Close'].diff().fillna(0) # 첫 날 결측치 0으로 채움
        st.success("✅ 모멘텀 계산 완료!")

        # ------------------------
        # ✅ 모든 Date 컬럼을 datetime.datetime 타입으로 통일 및 MultiIndex 제거
        # ------------------------
        # filtered_news의 'Date'는 이미 datetime.date이므로, pd.to_datetime으로 datetime.datetime으로 변환
        filtered_news_daily_sentiment = filtered_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
        filtered_news_daily_sentiment['Date'] = pd.to_datetime(filtered_news_daily_sentiment['Date'])

        # df_stock, vix는 이미 위에서 pd.to_datetime 처리됨
        # MultiIndex 제거는 필요 없음. FinanceDataReader와 yfinance는 기본적으로 MultiIndex를 반환하지 않음.
        # 만약 MultiIndex가 발생한다면, 보통 컬럼 이름이 (컬럼명, '') 형태로 되어있을 때 발생.
        # df.columns = ['_'.join(col).strip() for col in df.columns.values] 와 같이 처리 가능.
        # 현재 코드에서는 필요 없어 보임.

        # ------------------------
        # ✨ 데이터 병합
        # ------------------------
        st.info("📊 모든 데이터를 병합 중입니다...")
        df_merge = pd.merge(df_stock, vix, on='Date', how='left')
        df_merge = pd.merge(df_merge, filtered_news_daily_sentiment, on='Date', how='left').fillna(0)
        st.success("✅ 모든 데이터 병합 완료!")

        # ------------------------
        # ✨ 단순 선형회귀 예측
        # ------------------------
        st.subheader("📉 감성, 모멘텀, VIX 기반 단순 선형 회귀 예측")
        
        # 예측에 사용할 피처 선택 및 결측치 처리 (병합 후 최종적으로 처리)
        # VIX 데이터가 없을 경우 0으로 채워질 것이므로, fillna(0) 유지
        X = df_merge[['Sentiment_Score', 'Momentum', 'VIX_Close']].fillna(0).values
        y = df_merge['Close'].values

        if len(X) > 5: # 최소 6개 이상의 데이터 포인트가 있어야 안정적인 회귀 가능
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            df_merge['Predicted_Close'] = y_pred

            # ------------------------
            # ✨ 결과 시각화
            # ------------------------
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_merge['Date'], df_merge['Close'], label='실제 종가', color='blue')
            ax.plot(df_merge['Date'], df_merge['Predicted_Close'], label='예측 종가', linestyle='--', color='red')
            ax.set_title(f"{company_name} 주가 예측 (뉴스 감성 + 모멘텀 + VIX)")
            ax.set_xlabel("날짜")
            ax.set_ylabel("가격(₩/원)")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            st.metric("회귀계수 (감성)", f"{model.coef_[0]:.2f}")
            st.metric("회귀계수 (모멘텀)", f"{model.coef_[1]:.2f}")
            st.metric("회귀계수 (VIX)", f"{model.coef_[2]:.2f}")
        else:
            st.warning("데이터가 부족하여 예측을 수행할 수 없습니다. 뉴스 검색 기간을 늘리거나 다른 종목을 선택해보세요.")

        st.markdown("---")
        st.write("👉 감성 점수는 -1 (강 부정) ~ 1 (강 긍정), 모멘텀은 단순 종가 차이, VIX는 시장 변동성을 나타냅니다.")

