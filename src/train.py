# src/train.py
# Full professional training pipeline with MLflow tracking

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score


# ----------------------------------------
# MLflow Experiment Setup
# ----------------------------------------
mlflow.set_experiment("Fintech_Complaint_Risk_Analysis")


# ----------------------------------------
# Load FINAL Feature Dataset
# ----------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/processed/complaints_features.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Columns:", df.columns)


# ----------------------------------------
# Safety Check (Prevents KeyError)
# ----------------------------------------
if "risk_label" not in df.columns:
    raise ValueError("risk_label column not found in dataset. Check complaints_features.csv")


# ----------------------------------------
# Define Features and Target
# ----------------------------------------
X = df.drop(columns=["risk_label"])
y = df["risk_label"]


# ----------------------------------------
# Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------------------
# Start MLflow Run
# ----------------------------------------
with mlflow.start_run():

    model = LogisticRegression(max_iter=1000)

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # ----------------------------------------
    # Save Model & Feature Columns Locally
    # ----------------------------------------
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "../models/logistic_regression_model.pkl"
    )

    FEATURES_PATH = os.path.join(
        os.path.dirname(__file__),
        "../models/feature_columns.pkl"
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    print("\n Model and feature columns saved successfully!")
