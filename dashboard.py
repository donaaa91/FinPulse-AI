import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="FinPulse AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .fraud-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #F44336;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD & PROCESS DATA
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    df.rename(columns={
        'Class': 'is_fraud',
        'Time': 'time_seconds',
        'Amount': 'amount'
    }, inplace=True)
    df['hour_of_day'] = (df['time_seconds'] / 3600 % 24).astype(int)
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    df['is_outlier'] = df['amount_zscore'].abs() > 3
    df['invalid_amount'] = df['amount'] <= 0

    def classify_amount(amt):
        if amt <= 0: return 'INVALID'
        elif amt < 10: return 'MICRO'
        elif amt < 100: return 'LOW'
        elif amt < 1000: return 'MEDIUM'
        elif amt < 10000: return 'HIGH'
        else: return 'VERY_HIGH'

    df['amount_bucket'] = df['amount'].apply(classify_amount)
    return df

# ============================================
# SQL ANOMALY DETECTION
# ============================================
@st.cache_data
def run_anomaly_detection(df):
    conn = sqlite3.connect(':memory:')
    df.to_sql('transactions', conn, index=False, if_exists='replace')
    query = """
        WITH base AS (
            SELECT *,
                amount - LAG(amount,1) OVER (ORDER BY time_seconds) AS amount_spike,
                AVG(amount) OVER (PARTITION BY hour_of_day) AS avg_amount_by_hour,
                RANK() OVER (ORDER BY amount DESC) AS amount_rank
            FROM transactions
        )
        SELECT
            CASE
                WHEN amount_spike > 5000 THEN 'SUDDEN_SPIKE'
                WHEN amount > avg_amount_by_hour * 10 THEN 'HOURLY_OUTLIER'
                WHEN amount_rank <= 100 THEN 'TOP_100_HIGHEST'
                WHEN is_outlier = 1 THEN 'STATISTICAL_OUTLIER'
                ELSE 'NORMAL'
            END AS anomaly_type,
            amount,
            is_fraud,
            hour_of_day,
            amount_bucket
        FROM base
    """
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result

# ============================================
# MAIN APP
# ============================================
with st.spinner("🔄 Running FinPulse Pipeline..."):
    df = load_data()
    anomaly_df = run_anomaly_detection(df)

# ============================================
# HEADER
# ============================================
st.title("💳 FinPulse AI")
st.markdown("#### Real-Time Financial Transaction Intelligence & Anomaly Detection")
st.markdown("---")

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.title("🔧 Pipeline Controls")
st.sidebar.markdown("---")

hour_filter = st.sidebar.slider(
    "Filter by Hour of Day",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

bucket_filter = st.sidebar.multiselect(
    "Filter by Amount Bucket",
    options=['MICRO', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH', 'INVALID'],
    default=['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
)

show_fraud_only = st.sidebar.checkbox("Show Fraudulent Transactions Only", value=False)

# Apply filters
filtered_df = df[
    (df['hour_of_day'] >= hour_filter[0]) &
    (df['hour_of_day'] <= hour_filter[1]) &
    (df['amount_bucket'].isin(bucket_filter))
]
if show_fraud_only:
    filtered_df = filtered_df[filtered_df['is_fraud'] == 1]

st.sidebar.markdown("---")
st.sidebar.markdown(f"📊 **Showing:** {len(filtered_df):,} records")

# ============================================
# KPI METRICS ROW
# ============================================
col1, col2, col3, col4, col5 = st.columns(5)

total = len(filtered_df)
frauds = filtered_df['is_fraud'].sum()
outliers = filtered_df['is_outlier'].sum()
invalid = filtered_df['invalid_amount'].sum()
consistency = ((total - invalid) / total * 100) if total > 0 else 0

col1.metric("💳 Total Transactions", f"{total:,}")
col2.metric("🚨 Fraudulent", f"{frauds:,}", delta=f"{frauds/total*100:.2f}%", delta_color="inverse")
col3.metric("⚠️ Outliers", f"{outliers:,}")
col4.metric("❌ Invalid Records", f"{invalid:,}", delta_color="inverse")
col5.metric("✅ Consistency Rate", f"{consistency:.2f}%")

st.markdown("---")

# ============================================
# CHARTS ROW 1
# ============================================
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🕐 Transaction Volume by Hour")
    hourly = filtered_df.groupby(['hour_of_day', 'is_fraud'])['amount'].count().reset_index()
    hourly.columns = ['hour', 'is_fraud', 'count']
    hourly['type'] = hourly['is_fraud'].map({0: 'Normal', 1: 'Fraud'})
    fig1 = px.bar(
        hourly, x='hour', y='count', color='type',
        color_discrete_map={'Normal': '#2196F3', 'Fraud': '#F44336'},
        barmode='overlay', template='plotly_dark'
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("💰 Amount Distribution")
    plot_df = filtered_df[filtered_df['amount'] < 2000].copy()
    plot_df['type'] = plot_df['is_fraud'].map({0: 'Normal', 1: 'Fraud'})
    fig2 = px.histogram(
        plot_df, x='amount', color='type', nbins=100,
        color_discrete_map={'Normal': '#2196F3', 'Fraud': '#F44336'},
        opacity=0.7, template='plotly_dark'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================
# CHARTS ROW 2
# ============================================
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🚨 Anomaly Type Breakdown")
    anomaly_counts = anomaly_df.groupby(['anomaly_type', 'is_fraud']).size().reset_index()
    anomaly_counts.columns = ['anomaly_type', 'is_fraud', 'count']
    anomaly_counts['type'] = anomaly_counts['is_fraud'].map({0: 'Normal', 1: 'Fraud'})
    fig3 = px.bar(
        anomaly_counts, x='anomaly_type', y='count', color='type',
        color_discrete_map={'Normal': '#2196F3', 'Fraud': '#F44336'},
        barmode='group', template='plotly_dark'
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("📈 Avg Transaction Amount by Hour")
    rolling = filtered_df.groupby('hour_of_day')['amount'].mean().reset_index()
    rolling.columns = ['hour', 'avg_amount']
    fig4 = px.line(
        rolling, x='hour', y='avg_amount',
        markers=True, line_shape='spline', template='plotly_dark'
    )
    fig4.update_traces(line_color='#4CAF50', line_width=3)
    st.plotly_chart(fig4, use_container_width=True)

# ============================================
# ANOMALY DETECTION TABLE
# ============================================
st.markdown("---")
st.subheader("🔍 SQL Window Function — Anomaly Detection Results")

summary = anomaly_df.groupby('anomaly_type').agg(
    count=('amount', 'count'),
    avg_amount=('amount', 'mean'),
    max_amount=('amount', 'max'),
    frauds_caught=('is_fraud', 'sum')
).reset_index()
summary['avg_amount'] = summary['avg_amount'].round(2)
summary['max_amount'] = summary['max_amount'].round(2)
summary = summary.sort_values('frauds_caught', ascending=False)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "**FinPulse AI** | Built with Python, SQL Window Functions & Streamlit | "
    "Data: UCI Credit Card Fraud Detection Dataset"
)