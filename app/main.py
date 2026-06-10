# app/main.py

from fastapi import FastAPI
from app.schemas import ComplaintRequest
from app.inference import predict_complaint_risk
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Fintech Complaint Risk Analysis API",
    description="API to predict risk label for fintech customer complaints",
    version="1.0"
)

# -------------------------------
# Health Check
# -------------------------------
@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/health")
def health_check():
    return {"status": "OK"}

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(request: ComplaintRequest):
    try:
        logging.info(f"Request received: {request}")

        result = predict_complaint_risk(
            complaint_text=request.complaint_text,
            product=request.product,
            issue=request.issue
        )

        logging.info(f"Prediction result: {result}")

        return result

    except Exception as e:
        logging.error(f"Error: {str(e)}")

        return {
            "error": str(e)
        }