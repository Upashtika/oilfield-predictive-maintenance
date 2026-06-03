import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/bearing/bearing_features.csv")
#Features
X = df[["RMS", "Peak", "Kurtosis", "Skewness"]]
#Labels
y = df["Fault"]
# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)
# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
# Predictions
prediction = model.predict(X_test)
print("\nBearing Fault Classification Results")
print("-*-*-*-*-*-*-*-*-*-*-*")
print("Accuracy:", accuracy_score(y_test, prediction))
print("\nClassification Report:")
print(
	classification_report(y_test, prediction,target_names=encoder.classes_)
)

#save the model
joblib.dump(model,"models/bearing_fault_model.pkl")
joblib.dump(encoder,"models/bearing_label_encoder.pkl")
print("\nModel saved successfully.")