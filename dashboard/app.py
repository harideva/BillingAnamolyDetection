"""
Streamlit Dashboard for Billing Anomaly Detection
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from io import StringIO

st.set_page_config(
    page_title="Billing Anomaly Detection Dashboard",
    page_icon="⚡",
    layout="wide"
)

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.title("⚡ Billing Anomaly Detection System")
st.markdown("*Oracle CC&B-style billing data anomaly detection using Isolation Forest*")

with st.sidebar:
    st.header("📊 Controls")
    mode = st.radio("Select Mode", ["📁 Upload CSV", "📝 Single Record"])
    
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Not Available")
        st.info("Start API with: uvicorn api.main:app --reload")

if mode == "📁 Upload CSV":
    st.header("Upload Billing Data CSV")
    st.code("consumption_kwh, rate_per_kwh, total_amount, expected_amount")
    
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        
        if st.button("🔍 Run Anomaly Detection"):
            with st.spinner("Processing..."):
                files = {"file": uploaded}
                response = requests.post(f"{API_URL}/predict_batch", files=files)
                if response.status_code == 200:
                    result = response.json()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Records", result['total_records'])
                    c2.metric("Anomalies Detected", result['anomalies_detected'])
                    c3.metric("Anomaly %", f"{result['anomaly_percentage']:.1f}%")
                    
                    results_df = pd.DataFrame(result['results'])
                    st.dataframe(results_df)
                    
                    fig = px.pie(results_df, names='is_anomaly', title='Anomaly Distribution')
                    st.plotly_chart(fig)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "anomaly_results.csv")

elif mode == "📝 Single Record":
    st.header("Single Record Prediction")
    c1, c2 = st.columns(2)
    with c1:
        consumption = st.number_input("Consumption (kWh)", 0.0, value=450.5)
        rate = st.number_input("Rate per kWh", 0.0, 1.0, value=0.125)
    with c2:
        total = st.number_input("Total Amount", 0.0, value=56.31)
        expected = st.number_input("Expected Amount", 0.0, value=56.31)
    
    if st.button("🔍 Check"):
        payload = {
            "consumption_kwh": consumption,
            "rate_per_kwh": rate,
            "total_amount": total,
            "expected_amount": expected
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            if result['is_anomaly']:
                st.error("🚨 ANOMALY DETECTED!")
            else:
                st.success("✅ Normal Record")
            st.info(f"Score: {result['anomaly_score']:.4f}")
