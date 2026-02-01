import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURATION & MODELS ---
st.set_page_config(page_title="FinPulse", layout="wide")

# HARDCODED KEY (Use for local testing only!)
GROQ_API_KEY = "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"

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_resource
def load_llm():
    # Streamlit looks for GROQ_API_KEY in secrets automatically
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("Secret 'GROQ_API_KEY' not found. Check your secrets.toml or Cloud settings.")
        st.stop()

    return ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key
    )
sentiment_pipe = load_sentiment_model()
llm = load_llm()

# --- 2. STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
st.sidebar.title("🏦 FinPulse Intelligence")
ticker = st.sidebar.text_input("Ticker Symbol", value="").upper()
period = st.sidebar.selectbox("Price History", ["1mo", "3mo", "6mo", "1y"])

# --- 4. MAIN INTERFACE ---
st.title("Financial Sentiment & Market Intelligence Dashboard")
st.caption(f"Analyzing real-time data for: {ticker}")

if st.sidebar.button("Run Analysis"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    news = stock.news[:12]

    if news:
        # Safe extraction to avoid KeyError: 'title'
        headlines = [n.get('title', 'No Title Available') for n in news if isinstance(n, dict)]
        sentiments = sentiment_pipe(headlines)

        df = pd.DataFrame({
            "Headline": headlines,
            "Label": [s['label'] for s in sentiments],
            "Confidence": [s['score'] for s in sentiments]
        })

        # Metrics
        pos_pct = (df['Label'] == 'positive').sum() / len(df) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Bullish Sentiment", f"{pos_pct:.1f}%")
        col2.metric("Last Close", f"${hist['Close'].iloc[-1]:.2f}")
        col3.metric("Volatility (Std)", f"{hist['Close'].std():.2f}")

        # AI Summary
        st.subheader("🤖 AI Market Summary")
        with st.spinner("Synthesizing news..."):
            context = "\n".join(headlines)
            template = """
            As a financial analyst, summarize the sentiment for {ticker} based on these headlines:
            {context}
            Identify the top 2 risks and top 2 opportunities. Keep it concise.
            """
            prompt = PromptTemplate.from_template(template)
            chain = prompt | llm
            summary = chain.invoke({"ticker": ticker, "context": context})
            st.info(summary.content)

        # Tabs for Visuals
        t1, t2 = st.tabs(["Charts", "Data Table"])
        with t1:
            st.plotly_chart(px.line(hist, y="Close", title=f"{ticker} Price Trend"), use_container_width=True)
            st.plotly_chart(px.pie(df, names='Label', color='Label',
                                   color_discrete_map={'positive': '#00cc96', 'negative': '#ef553b', 'neutral': '#636efa'}))
        with t2:
            st.table(df)
    else:
        st.error("Could not retrieve news data. Check ticker symbol.")
else:
    st.write("Enter a ticker and click 'Run Analysis' to begin.")