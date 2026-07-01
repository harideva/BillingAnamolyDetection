#!/usr/bin/env python3
"""
BILLING ANOMALY DETECTION MVP - AUTOMATED BUILDER
One command to build, train, and deploy the entire system

Usage:
    python auto_build.py
    python auto_build.py --skip-tests
    python auto_build.py --github-token=YOUR_TOKEN
"""

import os
import sys
import subprocess
import shutil
import json
import time
import webbrowser
import urllib.request
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_NAME = "billing-anomaly-detection"
PYTHON_VERSION_REQUIRED = (3, 8)
PORT_API = 8000
PORT_DASHBOARD = 8501
PORT_MLFLOW = 5000

# ============================================================================
# CODE FILES (All the code we wrote earlier)
# ============================================================================

CODE_FILES = {
    "data/generate.py": '''"""
Synthetic Billing Data Generator for Oracle CC&B Anomaly Detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

def generate_billing_data(normal_count=5000, anomaly_count=250):
    """Generate synthetic billing data with intentional anomalies"""
    
    accounts = [f"CUS{str(i).zfill(6)}" for i in range(1, 1001)]
    meters = [f"MTR{str(i).zfill(6)}" for i in range(1, 2001)]
    
    tariff_codes = ['TOU_R1', 'TOU_R2', 'TOU_C1', 'RE_R1', 'RE_C1']
    tariff_rates = {'TOU_R1': 0.125, 'TOU_R2': 0.148, 'TOU_C1': 0.095, 
                    'RE_R1': 0.108, 'RE_C1': 0.082}
    
    seasons = ['Summer', 'Winter', 'Spring', 'Fall']
    seasons_multiplier = {'Summer': 1.3, 'Winter': 1.1, 'Spring': 0.9, 'Fall': 0.8}
    
    data = []
    
    for i in range(normal_count):
        account = random.choice(accounts)
        meter = random.choice(meters)
        tariff = random.choice(tariff_codes)
        season = random.choice(seasons)
        
        base_consumption = np.random.normal(350, 75)
        consumption = max(50, base_consumption * seasons_multiplier[season])
        rate = tariff_rates[tariff]
        amount = consumption * rate + np.random.normal(0, 2)
        amount = max(0, round(amount, 2))
        
        data.append({
            'account_id': account,
            'meter_id': meter,
            'billing_cycle': f'BILL_2026_{str(random.randint(1,6)).zfill(2)}',
            'previous_read': random.randint(8000, 12000),
            'current_read': random.randint(12000, 18000),
            'consumption_kwh': round(consumption, 1),
            'tariff_code': tariff,
            'rate_per_kwh': rate,
            'total_amount': amount,
            'season': season,
            'customer_type': 'Residential' if tariff in ['TOU_R1', 'TOU_R2', 'RE_R1'] else 'Commercial',
            'is_anomaly': 0
        })
    
    anomaly_types = ['consumption_spike', 'rate_mismatch', 'calculation_error', 'seasonal_deviation']
    
    for i in range(anomaly_count):
        account = random.choice(accounts)
        meter = random.choice(meters)
        tariff = random.choice(tariff_codes)
        season = random.choice(seasons)
        anomaly_type = random.choice(anomaly_types)
        
        if anomaly_type == 'consumption_spike':
            consumption = np.random.normal(1200, 300)
            rate = tariff_rates[tariff]
        elif anomaly_type == 'rate_mismatch':
            consumption = np.random.normal(350, 75)
            rate = random.choice([r for r in tariff_rates.values() if r != tariff_rates[tariff]])
        elif anomaly_type == 'calculation_error':
            consumption = np.random.normal(350, 75)
            rate = tariff_rates[tariff]
            amount = consumption * rate * random.uniform(1.5, 3.0)
            amount = round(amount, 2)
        elif anomaly_type == 'seasonal_deviation':
            consumption = np.random.normal(100, 30) if season == 'Summer' else np.random.normal(600, 100)
            rate = tariff_rates[tariff]
        
        if anomaly_type != 'calculation_error':
            amount = consumption * rate + np.random.normal(0, 2)
            amount = max(0, round(amount, 2))
        
        data.append({
            'account_id': account,
            'meter_id': meter,
            'billing_cycle': f'BILL_2026_{str(random.randint(1,6)).zfill(2)}',
            'previous_read': random.randint(8000, 12000),
            'current_read': random.randint(12000, 18000),
            'consumption_kwh': round(consumption, 1),
            'tariff_code': tariff,
            'rate_per_kwh': rate,
            'total_amount': amount,
            'season': season,
            'customer_type': 'Residential' if tariff in ['TOU_R1', 'TOU_R2', 'RE_R1'] else 'Commercial',
            'is_anomaly': 1,
            'anomaly_type': anomaly_type
        })
    
    df = pd.DataFrame(data)
    df['expected_amount'] = df['consumption_kwh'] * df['rate_per_kwh']
    df['amount_diff'] = df['total_amount'] - df['expected_amount']
    df['diff_percent'] = (df['amount_diff'] / df['expected_amount']) * 100
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    df.to_csv('data/raw/billing_data.csv', index=False)
    features = df[['consumption_kwh', 'rate_per_kwh', 'total_amount', 'expected_amount']].copy()
    features.to_csv('data/processed/features.csv', index=False)
    
    return df, features

if __name__ == "__main__":
    df, features = generate_billing_data()
    print(f"Generated {len(df)} records")
    print(f"Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
''',
    
    "models/train.py": '''"""
Training Pipeline for Billing Anomaly Detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
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
''',
    
    "api/main.py": '''"""
FastAPI Service for Billing Anomaly Detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Optional
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Billing Anomaly Detection API",
    description="Detect anomalies in utility billing data using Isolation Forest",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.getenv('MODEL_PATH', 'models/model.pkl')
scaler_path = os.getenv('SCALER_PATH', 'models/scaler.pkl')

if not os.path.exists(model_path):
    print("⚠️ Model not found. Training model...")
    from models.train import train_model
    train_model()
    print("✅ Model trained successfully")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

class BillingRecord(BaseModel):
    consumption_kwh: float
    rate_per_kwh: float
    total_amount: float
    expected_amount: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "consumption_kwh": 450.5,
                "rate_per_kwh": 0.125,
                "total_amount": 56.31,
                "expected_amount": 56.31
            }
        }

class PredictionResponse(BaseModel):
    record_id: Optional[int] = None
    prediction: int
    anomaly_score: float
    is_anomaly: bool
    severity: str
    timestamp: str

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_single(record: BillingRecord):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = np.array([[
        record.consumption_kwh,
        record.rate_per_kwh,
        record.total_amount,
        record.expected_amount
    ]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    score = model.score_samples(features_scaled)[0]
    is_anomaly = prediction == -1
    
    return PredictionResponse(
        prediction=int(prediction),
        anomaly_score=float(score),
        is_anomaly=is_anomaly,
        severity="HIGH" if is_anomaly else "NORMAL",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.read_csv(file.file)
    required = ['consumption_kwh', 'rate_per_kwh', 'total_amount', 'expected_amount']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    
    features = df[required].values
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    scores = model.score_samples(features_scaled)
    
    df['prediction'] = predictions
    df['anomaly_score'] = scores
    df['is_anomaly'] = predictions == -1
    df['severity'] = df['is_anomaly'].map({True: 'HIGH', False: 'NORMAL'})
    
    total = len(df)
    anomalies = df['is_anomaly'].sum()
    anomaly_pct = (anomalies / total) * 100 if total > 0 else 0
    
    return {
        "total_records": total,
        "anomalies_detected": int(anomalies),
        "anomaly_percentage": round(anomaly_pct, 2),
        "results": df.to_dict(orient='records')
    }
''',
    
    "dashboard/app.py": '''"""
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
''',
    
    "requirements.txt": '''numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.1
fastapi==0.100.1
uvicorn==0.23.1
pydantic==2.2.1
streamlit==1.25.0
plotly==5.15.0
mlflow==2.4.1
python-dotenv==1.0.0
requests==2.31.0
''',

    ".gitignore": '''__pycache__/
*.pyc
*.pyo
venv/
env/
.env
data/raw/
data/processed/
*.csv
models/*.pkl
mlflow/
mlruns/
.vscode/
.idea/
.DS_Store
''',
}

# ============================================================================
# THE MAIN BUILD SCRIPT
# ============================================================================

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ⚡ BILLING ANOMALY DETECTION MVP - AUTO BUILDER           ║
║                                                              ║
║   Building your AI-powered utility billing anomaly system   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_python():
    version = sys.version_info
    if version.major < PYTHON_VERSION_REQUIRED[0] or version.minor < PYTHON_VERSION_REQUIRED[1]:
        print(f"❌ Python {PYTHON_VERSION_REQUIRED[0]}.{PYTHON_VERSION_REQUIRED[1]}+ required")
        print(f"   Current: {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}")

def create_structure():
    dirs = ['data', 'data/raw', 'data/processed', 'models', 'api', 'dashboard', 'mlflow', 'tests']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        init_file = Path(d) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
    print("✅ Folder structure created")

#`def write_files():
#`    for path, content in CODE_FILES.items():
#`        Path(path).parent.mkdir(parents=True, exist_ok=True)
#`        with open(path, 'w') as f:
#`            f.write(content)
#`    print(f"✅ {len(CODE_FILES)} files written")

def write_files():
    for path, content in CODE_FILES.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # FIX: Write with UTF-8 encoding
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    print("✅ {len(CODE_FILES)} files written")


def setup_venv():
    if not Path('venv').exists():
        print("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
    print("✅ Virtual environment ready")

###def install_deps():
###    print("📦 Installing dependencies...")
###    pip = 'venv/bin/pip' if os.name != 'nt' else 'venv\\Scripts\\pip'
###    subprocess.run([pip, 'install', '--upgrade', 'pip'], check=True)
###    subprocess.run([pip, 'install', '-r', 'requirements.txt'], check=True)
###    print("✅ Dependencies installed")

def install_deps():
    print("Installing dependencies...")
    python = 'venv/bin/python' if os.name != 'nt' else 'venv\\Scripts\\python'
    pip = 'venv/bin/pip' if os.name != 'nt' else 'venv\\Scripts\\pip'
    
    # Upgrade pip using python -m pip instead of pip directly
    subprocess.run([python, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([pip, 'install', '-r', 'requirements.txt'], check=True)
    print("Dependencies installed")

def run_data_generation():
    print("📊 Generating synthetic data...")
    python = 'venv/bin/python' if os.name != 'nt' else 'venv\\Scripts\\python'
    subprocess.run([python, 'data/generate.py'], check=True)
    print("✅ Data generated")

def run_model_training():
    print("🤖 Training Isolation Forest model...")
    python = 'venv/bin/python' if os.name != 'nt' else 'venv\\Scripts\\python'
    subprocess.run([python, 'models/train.py'], check=True)
    print("✅ Model trained")

def start_services():
    print("🌐 Starting services...")
    
    python = 'venv/bin/python' if os.name != 'nt' else 'venv\\Scripts\\python'
    
    # Start API
    api_cmd = f'{python} -m uvicorn api.main:app --host 0.0.0.0 --port {PORT_API}'
    if os.name == 'nt':
        subprocess.Popen(['start', 'cmd', '/c', api_cmd], shell=True)
    else:
        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', api_cmd])
    print(f"✅ API started on http://localhost:{PORT_API}")
    
    time.sleep(3)
    
    # Start Dashboard
    dash_cmd = f'{python} -m streamlit run dashboard/app.py --server.port {PORT_DASHBOARD}'
    if os.name == 'nt':
        subprocess.Popen(['start', 'cmd', '/c', dash_cmd], shell=True)
    else:
        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', dash_cmd])
    print(f"✅ Dashboard started on http://localhost:{PORT_DASHBOARD}")
    
    time.sleep(3)
    webbrowser.open(f'http://localhost:{PORT_DASHBOARD}')

def github_push(token=None):
    if not token:
        token = input("GitHub token (or press Enter to skip): ").strip()
    if not token:
        print("⏭️ Skipping GitHub push")
        return
    
    print("📤 Pushing to GitHub...")
    subprocess.run(['git', 'init'], check=True)
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'commit', '-m', 'MVP: AI-powered billing anomaly detection'], check=True)
    subprocess.run(['git', 'branch', '-M', 'main'], check=True)
    
    repo = input("GitHub repo URL (https://...): ").strip()
    if repo:
        subprocess.run(['git', 'remote', 'add', 'origin', repo], check=True)
        subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
        print("✅ Pushed to GitHub")

def main():
    print_banner()
    
    print("\n🚀 Starting automated build...\n")
    
    check_python()
    create_structure()
    write_files()
    setup_venv()
    install_deps()
    run_data_generation()
    run_model_training()
    
    print("\n" + "="*60)
    print("✅ BUILD COMPLETE!")
    print("="*60)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ✅ What's Ready:                                           ║
║     • Synthetic billing data generated                      ║
║     • Isolation Forest model trained                        ║
║     • FastAPI REST API ready                               ║
║     • Streamlit dashboard ready                            ║
║                                                              ║
║  🌐 Access:                                                 ║
║     Dashboard: http://localhost:{PORT_DASHBOARD}            ║
║     API: http://localhost:{PORT_API}                       ║
║     API Docs: http://localhost:{PORT_API}/docs             ║
║                                                              ║
║  📂 Next Steps:                                             ║
║     1. Start services (if not auto-started)                ║
║     2. Upload CSV or test single records                   ║
║     3. Deploy to cloud or push to GitHub                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    start_services()
    
    push = input("\n\n🚀 Push to GitHub? (y/n): ").strip().lower()
    if push == 'y':
        github_push()
    
    print("\n✅ All done! Happy anomaly hunting! ⚡")

if __name__ == "__main__":
    main()