import time
import random
import pandas as pd
import joblib
from datetime import datetime
import csv

model = joblib.load("models/xgboost_failure_model.pkl")
scaler = joblib.load("models/xgboost_scaler.pkl")

while True:
    telemetry = {
        "Type": random.randint(0, 2),
        "Air temperature [K]": round(random.uniform(295, 305), 2),
        "Process temperature [K]": round(random.uniform(305, 315), 2),
        "Rotational speed [rpm]": random.randint(1200, 1600),
        "Torque [Nm]": round(random.uniform(20, 80), 2),
        "Tool wear [min]": random.randint(0, 250),
        "TWF": random.choices([0,1], weights=[95,5])[0],
        "HDF": random.choices([0,1], weights=[97,3])[0],
        "PWF": random.choices([0,1], weights=[98,2])[0],
        "OSF": random.choices([0,1], weights=[96,4])[0],
        "RNF": random.choices([0,1], weights=[99,1])[0]
    }
    
    df = pd.DataFrame([telemetry])
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    print("\n-*-*-*-*-*-*-*-*-*-*-")
    print("LIVE TELEMETRY")
    print(df)
    print(f"\nFailure Prediction: {prediction}")
    print(f"Failure Probability: {probability:.2%}")
    if probability > 0.7:
        print("🚨 CRITICAL ALERT: Possible Equipment Failure!")

    elif probability > 0.4:
        print("⚠ WARNING: Abnormal Equipment Condition")

    else:
        print("✅ Equipment Operating Normally")
        
    timestamp = datetime.now()
    if probability > 0.7:
        status = "CRITICAL"
    elif probability > 0.4:
        status = "WARNING"
    else:
        status = "NORMAL"
    if status != "NORMAL":
        with open("logs/events.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                round(probability, 4),
                status
            ])

    time.sleep(3)