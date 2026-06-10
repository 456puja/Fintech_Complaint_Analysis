#Generate numerical & categorical features for ML.

# src/feature_engineering.py

def create_features(text: str, product: str, issue: str) -> dict:
    """
    Generate features for model input.

    Returns:
        dict of features
    """
    features = {
        "char_length": len(text),
        "word_length": len(text.split()),
        "risk_keyword_flag": int(any(
            kw in text for kw in ["fraud", "scam", "chargeback", "unauthorized"]
        )),
        "product_encoded": hash(product) % 1000,  # simple encoding example
        "issue_encoded": hash(issue) % 1000
    }
    return features



#Purpose: Convert text and categorical info into ML-ready features.