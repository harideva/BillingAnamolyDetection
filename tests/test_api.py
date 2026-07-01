import requests
import json

print("=" * 50)
print("Testing Billing Anomaly Detection API")
print("=" * 50)

# 1. Health Check
print("\n1. Health Check:")
r = requests.get("http://localhost:8000/")
print(json.dumps(r.json(), indent=2))

# 2. Single Prediction - Normal
print("\n2. Single Prediction - Normal Record:")
payload = {
    "consumption_kwh": 350.5,
    "rate_per_kwh": 0.125,
    "total_amount": 43.81,
    "expected_amount": 43.81
}
r = requests.post("http://localhost:8000/predict", json=payload)
print(json.dumps(r.json(), indent=2))

# 3. Single Prediction - Anomaly (Consumption Spike)
print("\n3. Single Prediction - Anomaly (1200 kWh spike):")
payload = {
    "consumption_kwh": 1200.0,
    "rate_per_kwh": 0.125,
    "total_amount": 150.00,
    "expected_amount": 150.00
}
r = requests.post("http://localhost:8000/predict", json=payload)
print(json.dumps(r.json(), indent=2))

# 4. Batch Prediction with CSV
print("\n4. Batch Prediction - Upload test.csv:")
try:
    with open("test.csv", "rb") as f:
        files = {"file": ("test.csv", f, "text/csv")}
        r = requests.post("http://localhost:8000/predict_batch", files=files)
        result = r.json()
        print(f"Total Records: {result['total_records']}")
        print(f"Anomalies Detected: {result['anomalies_detected']}")
        print(f"Anomaly Percentage: {result['anomaly_percentage']}%")
        print("\nResults:")
        for record in result['results']:
            print(f"  Consumption: {record['consumption_kwh']} kWh | Anomaly: {record['is_anomaly']} | Severity: {record['severity']}")
except FileNotFoundError:
    print("ERROR: test.csv not found! Please create test.csv first.")