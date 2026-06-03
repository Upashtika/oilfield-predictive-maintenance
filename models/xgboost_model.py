import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/ai4i/ai4i2020.csv")
df.drop(["UDI", "Product ID"], axis=1, inplace=True)
encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])
X = df.drop(["Machine failure"], axis=1)
y = df["Machine failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nXGBoost Model Training Successful")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "models/xgboost_failure_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nModel and scaler saved successfully.") 