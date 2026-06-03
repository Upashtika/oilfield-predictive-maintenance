import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
df = pd.read_csv("data/ai4i/ai4i2020.csv")
df.drop(["UDI", "Product ID"], axis=1, inplace=True)
encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])
X = df.drop(["Machine failure"], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
model.fit(X_scaled)
predictions = model.predict(X_scaled)
df["Anomaly"] = predictions
anomaly_count = (df["Anomaly"] == -1).sum()
print("\nAnomaly Detection Completed")
print(f"Total anomalies detected: {anomaly_count}")
anomalies = df[df["Anomaly"] == -1]
print(anomalies.head())