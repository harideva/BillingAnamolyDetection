"""
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
