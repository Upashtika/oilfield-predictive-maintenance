import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv("data/ai4i/ai4i2020.csv")
print("Original Shape:", df.shape)
df.drop(["UDI", "Product ID"], axis=1, inplace=True)
label_encoder = LabelEncoder()
df["Type"] = label_encoder.fit_transform(df["Type"])
X = df.drop(["Machine failure"], axis=1)
y = df["Machine failure"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nPreprocessing Completed successfully.")
print("Processed Feature Shape:", X_scaled.shape)