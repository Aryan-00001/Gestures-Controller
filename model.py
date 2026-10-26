# train_gesture_model.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = "gesture_dataset.csv"
MODEL_PATH = "gesture_model.pkl"

# Load dataset
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Validate dataset
if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

df = df.dropna()

X = df.drop(columns=["label"])
y = df["label"].astype(int)

if X.shape[1] != 126:
    raise ValueError(
        f"Expected 126 hand features (two hands), found {X.shape[1]}. "
        "Collect the dataset again using the two-hand collector."
    )

if y.nunique() < 2:
    raise ValueError("Collect samples for at least two different gestures.")

class_counts = y.value_counts()
if class_counts.min() < 2:
    raise ValueError("Each gesture needs at least two samples for stratified testing.")

print("Features per sample:", X.shape[1])
print("Total samples:", len(df))
print("Samples per label:\n", class_counts.sort_index())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Test model on held-out samples
y_pred = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save trained model
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
