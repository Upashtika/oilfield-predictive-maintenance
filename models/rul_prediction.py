import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# nasa FD001 COLUMN NAMES
columns = ["unit", "cycle"]
for i in range(1, 4):
    columns.append(f"setting_{i}")
for i in range(1, 22):
    columns.append(f"sensor_{i}")
    
# Load dataset
df = pd.read_csv("data/nasa/train_FD001.txt", sep=r"\s+",header=None)

#removing extra empty columns
df = df.iloc[:, :26]
df.columns = columns
# Calculate RUL
max_cycle = df.groupby("unit")["cycle"].max()
df["RUL"] = df.apply(lambda row: max_cycle[row["unit"]] - row["cycle"], axis=1)

print(df.columns.tolist())
print(df.head())

#features
X = df.drop(columns=["unit", "cycle", "RUL"])

# Target
y = df["RUL"]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("\nRUL Model Training Completed")
print(f"MAE: {mae:.2f} cycles")

# Save the model
joblib.dump(model, "models/rul_model.pkl")
print("RUL model saved successfully")