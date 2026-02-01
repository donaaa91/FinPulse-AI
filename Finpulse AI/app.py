import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="JPMC Risk Engine | Debug Mode", layout="wide")

st.title("🛡️ Sentinel: Risk Engine (Debug Mode)")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
max_sector_pct = st.sidebar.slider("Max Sector (%)", 10, 100, 30)
restricted_sectors = st.sidebar.multiselect("Restricted", ["Gambling", "Crypto", "Defense"], ["Gambling", "Crypto"])
var_conf = st.sidebar.slider("VaR Confidence", 90, 99, 95)


# --- HELPER FUNCTIONS ---
def clean_currency(x):
    """Aggressive cleaner for currency strings"""
    if pd.isna(x) or x == '': return 0.0
    if isinstance(x, (int, float)): return float(x)

    x = str(x).strip()
    # Remove '$', ',', ' ' (spaces)
    clean_str = x.replace('$', '').replace(',', '').replace(' ', '')
    try:
        return float(clean_str)
    except:
        return 0.0


# --- MAIN APP ---
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("Waiting for file...")
    st.stop()

# 1. READ FILE
try:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Debug: Raw Columns Found")
    st.write(list(df.columns))  # THIS SHOWS YOU WHAT PYTHON SEES
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# 2. SMART RENAME (Forceful)
# We look for specific keywords to identify the money column
df.columns = df.columns.str.strip()  # Remove invisible spaces

rename_map = {}
for col in df.columns:
    c_lower = col.lower()
    if "value" in c_lower or "amount" in c_lower or "position" in c_lower or "$" in c_lower:
        rename_map[col] = "Market_Value"
    elif "symbol" in c_lower or "ticker" in c_lower:
        rename_map[col] = "Asset_ID"
    elif "company" in c_lower or "name" in c_lower:
        rename_map[col] = "Asset_Name"
    elif "sector" in c_lower or "industry" in c_lower:
        rename_map[col] = "Sector"
    elif "asset" in c_lower and "class" in c_lower:
        rename_map[col] = "Asset_Type"
    elif "rating" in c_lower:
        rename_map[col] = "Credit_Rating"

if rename_map:
    st.success(f"Mapped Columns: {rename_map}")
    df = df.rename(columns=rename_map)

# 3. VALIDATE REQUIRED COLUMN
if 'Market_Value' not in df.columns:
    st.error("❌ Critical Error: Could not identify a 'Market_Value' column.")
    st.write("Please rename your CSV column header to 'Market_Value' or 'Amount' manually and re-upload.")
    st.stop()

# 4. CLEAN DATA (With Visual Check)
st.write("### 🧹 Data Cleaning Check")
# Show 'Before' state
st.write("Before Cleaning (First 5 rows):")
st.dataframe(df['Market_Value'].head())

# Apply Cleaning
df['Market_Value'] = df['Market_Value'].apply(clean_currency)

# Show 'After' state
st.write("After Cleaning (First 5 rows):")
st.dataframe(df['Market_Value'].head())

# Check for Zeroes
total_val = df['Market_Value'].sum()
if total_val == 0:
    st.error("⚠️ All values are $0.0! The cleaner failed.")
    st.stop()

# 5. FILL DEFAULTS
if 'Asset_Type' not in df.columns: df['Asset_Type'] = 'Equity'
if 'Sector' not in df.columns: df['Sector'] = 'Unknown'
if 'Credit_Rating' not in df.columns: df['Credit_Rating'] = 'NR'
if 'Volatility_Score' not in df.columns: df['Volatility_Score'] = 0.20

# 6. RUN RISK ENGINE
df['Weight (%)'] = (df['Market_Value'] / total_val) * 100
audit_log = []

# Logic Gates
sector_weights = df.groupby("Sector")['Weight (%)'].sum()
for sector, weight in sector_weights.items():
    if weight > max_sector_pct:
        audit_log.append({"Rule": "Concentration", "Detail": f"{sector} is {weight:.1f}%", "Status": "FAIL"})

for i, row in df.iterrows():
    if row['Sector'] in restricted_sectors:
        audit_log.append({"Rule": "Restricted Asset", "Detail": f"Found {row.get('Asset_Name')}", "Status": "FAIL"})

# 7. DASHBOARD
col1, col2 = st.columns(2)
with col1:
    if audit_log:
        st.error(f"Violations: {len(audit_log)}")
        st.table(pd.DataFrame(audit_log))
    else:
        st.success("✅ Compliant")

with col2:
    fig = px.pie(df, values='Market_Value', names='Sector', title='Allocation')
    st.plotly_chart(fig, use_container_width=True)

# 8. AI ANALYST
st.markdown("---")
st.subheader("🤖 AI Analyst")
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = st.text_input("Groq API Key", type="password")

if st.button("Generate Report") and api_key and audit_log:
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        violations = "\n".join([f"- {v['Rule']}: {v['Detail']}" for v in audit_log])
        msg = f"Fix these risk violations:\n{violations}"
        resp = client.chat.completions.create(messages=[{"role": "user", "content": msg}], model="llama3-8b-8192")
        st.info(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"AI Error: {e}")