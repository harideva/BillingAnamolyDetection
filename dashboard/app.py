"""
Streamlit Dashboard for Billing Anomaly Detection
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
from io import StringIO

st.set_page_config(
    page_title="Billing Anomaly Detection Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Use environment variable or fallback to local
# API_URL = os.getenv("API_URL", "https://billing-anomaly-api.onrender.com")
API_URL = "https://billing-anomaly-api.onrender.com"

st.title("⚡ Billing Anomaly Detection System")
st.markdown("*Oracle CC&B-style billing data anomaly detection using Isolation Forest*")

# Sidebar
with st.sidebar:
    st.header("📊 Controls")
    mode = st.radio("Select Mode", ["📁 Upload CSV", "📝 Single Record"])
    
    st.divider()
    
    # Check API connection
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error(f"❌ API Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ API Not Available")
        st.info(f"Make sure API is running at: {API_URL}")
    except Exception as e:
        st.error(f"❌ Connection Error: {str(e)}")

# Main content - UPLOAD CSV MODE
if mode == "📁 Upload CSV":
    st.header("📁 Upload Billing Data CSV")
    st.markdown("Upload a CSV file with the following columns:")
    st.code("consumption_kwh, rate_per_kwh, total_amount, expected_amount")
    
    uploaded = st.file_uploader("Choose CSV", type="csv")
    
    if uploaded is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded)
            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head())
            
            if st.button("🔍 Run Anomaly Detection", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        # Reset file pointer
                        uploaded.seek(0)
                        files = {"file": ("uploaded.csv", uploaded, "text/csv")}
                        
                        # Call API
                        response = requests.post(
                            f"{API_URL}/predict_batch",
                            files=files,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Total Records", result['total_records'])
                            with col2:
                                st.metric("🚨 Anomalies Detected", result['anomalies_detected'])
                            with col3:
                                st.metric("📈 Anomaly %", f"{result['anomaly_percentage']:.1f}%")
                            
                            # Results table
                            results_df = pd.DataFrame(result['results'])
                            
                            st.subheader("📋 Detection Results")
                            # Color-code anomalies
                            def color_rows(row):
                                if row['is_anomaly']:
                                    return ['background-color: #ffcccc'] * len(row)
                                return [''] * len(row)
                            
                            st.dataframe(results_df.style.apply(color_rows, axis=1))
                            
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.pie(
                                    results_df,
                                    names='is_anomaly',
                                    title='Anomaly Distribution',
                                    color_discrete_map={True: '#ff4b4b', False: '#00cc96'},
                                    labels={True: 'Anomaly', False: 'Normal'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.histogram(
                                    results_df,
                                    x='consumption_kwh',
                                    color='is_anomaly',
                                    title='Consumption Distribution',
                                    color_discrete_map={True: '#ff4b4b', False: '#00cc96'},
                                    labels={True: 'Anomaly', False: 'Normal'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Download results
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_data,
                                file_name="anomaly_detection_results.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            st.text(response.text[:500])
                    except requests.exceptions.Timeout:
                        st.error("⏰ Request timed out. Try a smaller file.")
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Could not connect to API. Make sure it's running.")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")

# Main content - SINGLE RECORD MODE
else:
    st.header("📝 Single Record Prediction")
    st.markdown("Enter billing data to check for anomalies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        consumption = st.number_input("Consumption (kWh)", value=450.5, min_value=0.0, step=10.0)
        rate = st.number_input("Rate per kWh", value=0.125, min_value=0.0, max_value=1.0, step=0.001)
    
    with col2:
        total = st.number_input("Total Amount", value=56.31, min_value=0.0, step=1.0)
        expected = st.number_input("Expected Amount", value=56.31, min_value=0.0, step=1.0)
    
    if st.button("🔍 Check for Anomaly", type="primary"):
        payload = {
            "consumption_kwh": consumption,
            "rate_per_kwh": rate,
            "total_amount": total,
            "expected_amount": expected
        }
        
        with st.spinner("Checking..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['is_anomaly']:
                        st.error("🚨 **ANOMALY DETECTED!**")
                        st.warning(f"This record has been flagged as anomalous.")
                    else:
                        st.success("✅ **Normal Record**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
                    with col2:
                        st.metric("Severity", result['severity'])
                else:
                    st.error(f"❌ API Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()
st.caption("Built by Hari Prasad Devarapalli, Ph.D. | Exelon/ComEd - Sr. Software Engineer 2")
st.caption(f"🔗 [GitHub Repository](https://github.com/harideva/BillingAnomalyDetection)")