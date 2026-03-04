from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ============================
# 1️⃣ Load Trained Pipeline
# ============================

import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Telco Churn Prediction API")


# ============================
# 2️⃣ Define Input Schema
# ============================

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ============================
# 3️⃣ Health Check Endpoint
# ============================

@app.get("/")
def home():
    return {"status": "Churn Prediction API Running"}


# ============================
# 4️⃣ Prediction Endpoint
# ============================

@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }