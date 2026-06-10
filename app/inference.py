import os
import sys
import pandas as pd
import joblib
import logging
import numpy as np

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------
# Path setup
# -------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from preprocessing import clean_text
from feature_engineering import create_features

# -------------------------------
# Load model + features
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/logistic_regression_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../models/feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

logging.info("Model loaded successfully")


# -------------------------------
# RULE ENGINE (FINAL SAFE VERSION)
# -------------------------------
def rule_based_label(text: str):

    text = text.lower().strip()

    high_keywords = [
        "unauthorized", "fraud", "stolen", "hack", "not approved", "scam"
    ]

    medium_keywords = [
        "not sure", "might", "unknown", "don't recognize", "subscription", "confused"
    ]

    low_keywords = [
        "payment issue", "issue", "problem", "help", "query", "check", "history"
    ]

    # HIGH PRIORITY
    if any(w in text for w in high_keywords):
        return "HIGH_RISK"

    # MEDIUM PRIORITY
    if any(w in text for w in medium_keywords):
        return "MEDIUM_RISK"

    # SAFE LOW RULE (IMPORTANT FIX)
    if len(text.split()) <= 3:
        return "LOW_RISK"

    if any(w in text for w in low_keywords):
        return "LOW_RISK"

    return None


# -------------------------------
# CONFIDENCE LOGIC
# -------------------------------
def confidence_level(label: str):

    if label == "HIGH_RISK":
        return "VERY_HIGH"
    elif label == "MEDIUM_RISK":
        return "MEDIUM"
    else:
        return "LOW"


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def predict_complaint_risk(complaint_text: str, product: str, issue: str):

    try:

        # -----------------------
        # Input validation
        # -----------------------
        if not complaint_text or not complaint_text.strip():
            return {
                "risk_label": "INVALID_INPUT",
                "risk_probability": 0.0,
                "confidence_level": "NONE"
            }

        # -----------------------
        # Preprocessing
        # -----------------------
        cleaned_text = clean_text(complaint_text)

        # -----------------------
        # Feature engineering
        # -----------------------
        features = create_features(
            text=cleaned_text,
            product=product,
            issue=issue
        )

        df = pd.DataFrame([features])
        df = df.reindex(columns=feature_columns, fill_value=0)

        # -----------------------
        # MODEL SIGNAL (SAFE)
        # -----------------------
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]

            sorted_probs = np.sort(probs)

            # safe confidence gap (not fake 0.99)
            model_signal = float(sorted_probs[-1] - sorted_probs[-2])
        else:
            model_signal = 0.5

        # -----------------------
        # FINAL DECISION
        # -----------------------
        rule_label = rule_based_label(complaint_text)

        if rule_label:
            risk_label = rule_label
        else:
            if model_signal >= 0.85:
                risk_label = "HIGH_RISK"
            elif model_signal >= 0.50:
                risk_label = "MEDIUM_RISK"
            else:
                risk_label = "LOW_RISK"

        # -----------------------
        # FIXED PROBABILITY OUTPUT
        # -----------------------
        if risk_label == "HIGH_RISK":
            risk_probability = 0.90
        elif risk_label == "MEDIUM_RISK":
            risk_probability = 0.60
        else:
            risk_probability = 0.25

        # -----------------------
        # RESPONSE
        # -----------------------
        result = {
            "risk_label": risk_label,
            "risk_probability": round(risk_probability, 4),
            "confidence_level": confidence_level(risk_label)
        }

        logging.info(f"Prediction result: {result}")

        return result

    except Exception as e:
        logging.error(str(e))
        return {"error": str(e)}