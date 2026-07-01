elif mode == "Upload CSV":
    st.header("Upload Billing Data CSV")
    st.code("consumption_kwh, rate_per_kwh, total_amount, expected_amount")
    
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        try:
            # Read CSV with error handling
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            
            if st.button("Run Anomaly Detection"):
                with st.spinner("Processing..."):
                    try:
                        # Reset file pointer
                        uploaded.seek(0)
                        files = {"file": ("test.csv", uploaded, "text/csv")}
                        
                        # Make API call with timeout
                        response = requests.post(
                            f"{API_URL}/predict_batch", 
                            files=files,
                            timeout=30
                        )
                        
                        # Check status
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
                            st.download_button("Download Results", csv, "anomaly_results.csv")
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            st.text(response.text[:500])  # Show first 500 chars of error
                    except requests.exceptions.Timeout:
                        st.error("⏰ API Timeout - Try with smaller file")
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Connection Error - API not reachable")
                    except Exception as e:
                        st.error(f"❌ Unexpected Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")