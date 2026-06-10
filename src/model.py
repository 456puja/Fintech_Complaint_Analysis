#Defines all ML models (baseline + final).

# src/model.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_baseline_model():
    """Return Logistic Regression as baseline"""
    return LogisticRegression(max_iter=1000)

def get_final_model():
    """Return Random Forest as final model"""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )


#Purpose: Separate model definitions from training code for modularity.