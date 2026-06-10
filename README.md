# 📌 Fintech Complaint Risk Analysis

## Production-Ready NLP System using FastAPI & Docker

### 🚀 Project Overview

This project is an end-to-end NLP-based Machine Learning system designed to automatically analyze fintech customer complaints, classify risk levels, and prioritize high-impact issues to support faster resolution and operational efficiency.

The system processes raw complaint narratives, applies advanced text preprocessing and feature engineering techniques, trains and evaluates machine learning models, and exposes the final model as a production-ready REST API using FastAPI and Docker.

* Official source: CFPB Consumer Complaint Database
* Dataset Link: https://www.consumerfinance.gov/data-research/consumer-complaints/?utm_source=chatgpt.com
* Domain: FinTech / Consumer Finance
* Dataset Type: Structured Tabular Dataset
* Data Nature: Consumer Complaint Records
* Number of Records: 12,714,866
* Number of Features: 18

### 🎯 Project Objective

To design and deploy an end-to-end NLP-based machine learning system that analyzes fintech customer complaint text, classifies complaints into risk categories, and enables real-time, scalable inference through a production-ready API to support business prioritization and operational efficiency.

### 💼 Business Objective

* Develop an intelligent system for automated risk assessment of financial consumer complaints.
* Enable early identification and prioritization of potentially critical customer issues.
* Streamline complaint handling workflows through machine learning-driven risk classification.
* Enhance operational efficiency and support proactive risk management in financial services.


### 🏗️ Project Architecture

<pre>
Fintech_Complaint_Analysis/
│
├── app/                         # FastAPI application layer (Model Serving)
│   ├── main.py                  # FastAPI entry point
│   ├── inference.py             # Prediction logic
│   └── schemas.py               # Request & response validation (Pydantic models)
│
├── src/                         # Core ML pipeline modules
│   ├── preprocessing.py         # Text cleaning & preprocessing logic
│   ├── feature_engineering.py   # TF-IDF / Vectorization / Feature transformation
│   ├── model.py                 # Model definition & loading
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Model evaluation metrics
│   └── utils.py                 # Helper utilities
│
├── data/                        # Dataset storage (excluded in .gitignore)
│   ├── raw/                     # Original raw dataset
│   └── processed/               # Cleaned & transformed datasets
│
├── notebooks/                   # End-to-end experimentation workflowc
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_feature_selection.ipynb
│   ├── 06_model_selection.ipynb
│   ├── 07_model_training.ipynb
│   └── 08_model_evaluation.ipynb
│
├── models/                      # Serialized model artifacts
│   ├── logistic_regression_model.pkl
│   ├── feature_columns.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Containerization configuration
├── .dockerignore                # Docker ignore rules
├── mlruns/                      # MLflow experiment tracking (ignored in git)
├── mlflow.db                    # MLflow backend database (ignored in git)
└── README.md                    # Project documentation
</pre>


### 🔬 End-to-End Workflow

1️⃣ Data Understanding
* Analyzed fintech complaint dataset
* Identified key textual and categorical features
* Assessed missing values and distribution

2️⃣ Exploratory Data Analysis (EDA)
* Complaint distribution analysis
* Risk pattern analysis
* Temporal trend visualization

3️⃣ Text Preprocessing
* Lowercasing
* Special character removal
* Tokenization (NLTK)
* Stopword removal
* Lemmatization
* Text normalization

4️⃣ Feature Engineering
* TF-IDF vectorization
* Complaint length feature
* Risk keyword indicators
* Temporal features
* Encoded categorical variables

5️⃣ Feature Selection
* Removed low-importance features
* Reduced dimensionality
* Improved model stability

6️⃣ Model Selection & Training

Baseline Model: Logistic Regression
Compared models using:
* Accuracy
* Precision
* Recall
* F1-score

Selected final model based on:
* Performance
* Interpretability
* Business suitability

#### 📊 MLflow Integration:
* Tracked experiment parameters
* Logged evaluation metrics (Accuracy, F1-Score)
* Logged trained model artifacts
* Stored environment configuration (conda.yaml, requirements.txt)
* Ensured full experiment reproducibility


7️⃣ Model Evaluation
* Confusion matrix
* Classification metrics
* Risk prioritization validation


### 🧠 Model Details
* Algorithm: Logistic Regression
* Feature Engineering: TF-IDF + Engineered Metadata Features
* Evaluation Metric: F1-Score (Primary)
* Experiment Tracking: MLflow
* Deployment Format: .pkl model using joblib


### 📊 MLflow Experiment Tracking

This project integrates MLflow for:
* Experiment tracking
* Hyperparameter logging
* Metric comparison
* Model artifact storage
* Environment reproducibility

Run MLflow UI Locally
mlflow ui

Open in browser:
http://127.0.0.1:5000/

You can view:
* Parameters
* Metrics 
* Artifacts
* Model versions

### 📦 Model Artifacts
```
models/
├── logistic_regression_model.pkl
├── feature_columns.pkl
└── tfidf_vectorizer.pkl
```

### 📦 MLflow Artifacts:
```
├── MLmodel
├── model.pkl
├── conda.yaml
├── python_env.yaml
├── requirements.txt
```

### 🚀 Running the Project Locally

Follow the steps below to run the Fintech Complaint Risk Analysis API on your local machine.

### 📋 Prerequisites

Make sure the following are installed:

* Python 3.10 or above
* pip (Python package manager)
* Git
* Virtual Environment (venv)


1️⃣ Clone the Repository
git clone <your-repository-url>
cd Fintech_Complaint_Analysis

2️⃣ Create a Virtual Environment
python -m venv venv

3️⃣ Activate the Virtual Environment
venv\Scripts\activate

4️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

5️⃣ Train Model & Track Experiment
python src/train.py

6️⃣ Start the FastAPI Server
uvicorn app.main:app --reload

The server will start at:
http://127.0.0.1:8000/

7️⃣ Access API Documentation
http://127.0.0.1:8000/docs

This will open the interactive Swagger UI where We can test the API endpoints.


### 📡 API Endpoints

#### ❤️ Health Check
🔍  GET /health

#### 🤖 Predict Complaint Risk
📤 POST /predict


### 🧪 API Usage Examples

Example Request Body:

##### Example 1: Fraudulent Transaction (HIGH_RISK)

**Input**

```json
{
  "complaint_text": "Unauthorized transactions were made on my credit card. I believe this is fraud and I need immediate assistance.",
  "product": "Credit Card",
  "issue": "Fraudulent Transaction"
}
```

**Output**

```json
{
  "risk_label": "HIGH_RISK",
  "risk_probability": 0.9,
  "confidence_level": "VERY_HIGH"
}
```
##### Example 2: Card Hacked / Stolen (HIGH_RISK)

**Input**
```json 
{
  "complaint_text": "My card was hacked and used without my permission.",
  "product": "Credit Card",
  "issue": "Card Compromise"
}
```
**Output**
```json 
{
  "risk_label": "HIGH_RISK",
  "risk_probability": 0.9,
  "confidence_level": "VERY_HIGH"
}
```

##### Example 3: Unknown Transaction (MEDIUM_RISK)

**Input**
```json 
{
  "complaint_text": "I noticed a transaction that I don't recognize and I am not sure if it belongs to me.",
  "product": "Credit Card",
  "issue": "Unknown Transaction"
}
```
**Output**
```json 
{
  "risk_label": "MEDIUM_RISK",
  "risk_probability": 0.6,
  "confidence_level": "MEDIUM"
}
```

##### Example 4: Payment History Query (LOW_RISK)

**Input**
```json 
{
  "complaint_text": "I need help checking my payment history.",
  "product": "Payment Service",
  "issue": "General Query"
}
```
**Output**
```json 
{
  "risk_label": "LOW_RISK",
  "risk_probability": 0.25,
  "confidence_level": "LOW"
}
```

##### Example 5: Balance Inquiry (LOW_RISK)

**Input**
```json 
{
  "complaint_text": "I want to check my account balance.",
  "product": "Bank Account",
  "issue": "Balance Inquiry"
}
```

**Output**
```json 
{
  "risk_label": "LOW_RISK",
  "risk_probability": 0.25,
  "confidence_level": "LOW"
}
```

🛑 Stop the Server
CTRL + C


### 🐳 Docker Deployment

1️⃣ Build Docker Image
docker build -t fintech-complaint-api .

2️⃣ Run Container
docker run -d -p 8000:8000 --name fintech_api fintech-complaint-api

3️⃣ Access API
http://localhost:8000/docs


### 🛠️ Tech Stack

* Python 3.10 – Core language for ML workflows and API development
* pandas, NumPy – Data preprocessing, feature engineering and numerical operations
* NLTK
* Logistic Regression -  Selected based on evaluation metrics
* Scikit-learn - Model training, evaluation, pipelines
* FastAPI - Real-time inference API
* Uvicorn - ASGI server for serving the FastAPI application
* Docker - Containerized deployment
* Joblib - Model and pipeline serialization
* Jupyter Notebook / PyCharm – Experimentation and development


### 🏁 Conclusion

This project demonstrates the complete NLP-based Machine Learning lifecycle, from text preprocessing and feature engineering to model training, experiment tracking with MLflow, and production deployment using FastAPI and Docker.

The system is designed to automatically analyze fintech customer complaints, classify risk levels, and provide scalable real-time predictions through a REST API. It showcases practical applications of Natural Language Processing (NLP), MLOps, Machine Learning, and model deployment in a real-world fintech use case.


### ⭐ Thank you for visiting this project!


