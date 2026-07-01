"""
Training Pipeline for Billing Anomaly Detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import mlflow
import mlflow.sklearn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model():
    """Train Isolation Forest model with MLflow tracking"""
    
    if not os.path.exists('data/processed/features.csv'):
        print("⚠️ Data not found. Generating data now...")
        from data.generate import generate_billing_data
        df, features = generate_billing_data()
        print(f"✅ Generated {len(df)} records with {df['is_anomaly'].sum()} anomalies")
    
    data = pd.read_csv('data/processed/features.csv')
    
    X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        max_samples='auto',
        bootstrap=False
    )
    
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", 0.05)
        mlflow.log_param("random_state", 42)
        
        model.fit(X_train_scaled)
        
        train_pred = model.predict(X_train_scaled)
        train_anomaly_pct = (train_pred == -1).mean() * 100
        
        test_pred = model.predict(X_test_scaled)
        test_anomaly_pct = (test_pred == -1).mean() * 100
        
        mlflow.log_metric("train_anomaly_pct", train_anomaly_pct)
        mlflow.log_metric("test_anomaly_pct", test_anomaly_pct)
        
        mlflow.sklearn.log_model(model, "anomaly_detector")
        mlflow.log_artifact("data/processed/features.csv")
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        print(f"✅ Model saved to models/model.pkl")
        print(f"📊 Training anomaly percentage: {train_anomaly_pct:.2f}%")
        print(f"📊 Test anomaly percentage: {test_anomaly_pct:.2f}%")
        
        return model, scaler

if __name__ == "__main__":
    model, scaler = train_model()
