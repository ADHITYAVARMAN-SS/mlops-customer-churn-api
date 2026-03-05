from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ============================
# Load Model
# ============================

model = joblib.load("model.pkl")

app = FastAPI(title="Telco Churn Prediction API")

# ============================
# Input Schema
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
# Health Check
# ============================

@app.get("/")
def home():
    return {"status": "Churn Prediction API Running"}


# ============================
# Business Decision Logic
# ============================

def churn_business_decision(prob):

    if prob > 0.7:
        return {
            "risk_level": "High",
            "recommended_action": "Offer retention discount or call customer"
        }

    elif prob > 0.4:
        return {
            "risk_level": "Medium",
            "recommended_action": "Send promotional email or loyalty reward"
        }

    else:
        return {
            "risk_level": "Low",
            "recommended_action": "No action required"
        }


# ============================
# Prediction Endpoint
# ============================

@app.post("/predict")
def predict(data: CustomerData):

    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    decision = churn_business_decision(probability)

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": decision["risk_level"],
        "recommended_action": decision["recommended_action"]
    }