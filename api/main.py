"""
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

### app.add_middleware(
###     CORSMiddleware,
###     allow_origins=["*"],
###     allow_credentials=True,
###     allow_methods=["*"],
###     allow_headers=["*"],
### )

# CORS - Updated for better file upload support
#### app.add_middleware(
####     CORSMiddleware,
####     allow_origins=["*"],
####     allow_credentials=True,
####     allow_methods=["*"],
####     allow_headers=["*"],
####     expose_headers=["*"]
#### )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://billing-anomaly-dashboard.onrender.com",
        "http://localhost:8501",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
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
