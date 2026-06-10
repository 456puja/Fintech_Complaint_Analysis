#Contains helper functions like loading/saving models or logging.

# src/utils.py

import joblib
import os

def save_object(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"Object saved at {path}")

def load_object(path: str):
    return joblib.load(path)


#Purpose: Reusable utilities to avoid repetition.