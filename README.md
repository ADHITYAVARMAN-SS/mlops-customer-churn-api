# Telco Customer Churn Prediction – MLOps Project

## Overview

This project implements an **end-to-end Machine Learning Operations (MLOps) pipeline** for predicting customer churn in a telecommunications company.

The system trains a machine learning model to predict whether a customer is likely to leave the service and exposes the model through a **FastAPI REST API** deployed using **Docker and cloud infrastructure**.

The goal of this project is to demonstrate **production-ready ML engineering practices**, including:

* Data preprocessing pipelines
* Model training and evaluation
* Experiment tracking with MLflow
* Hyperparameter tuning
* Threshold optimization
* Model deployment with FastAPI
* Containerization using Docker
* Cloud deployment

---

# Dataset

The project uses the **Telco Customer Churn Dataset**, which contains information about telecom customers including:

* demographic information
* account information
* services subscribed
* billing information

Target variable:

```
Churn
0 → Customer stays
1 → Customer leaves
```

Key features include:

* tenure
* MonthlyCharges
* TotalCharges
* Contract type
* Internet service
* Payment method
* Support services

---

# Model Performance

Final trained model metrics:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.7697 |
| Precision | 0.5498 |
| Recall    | 0.7380 |
| F1 Score  | 0.6301 |
| ROC-AUC   | 0.8352 |

### Threshold Optimization

Instead of using the default **0.5 probability threshold**, the model uses:

```
Best Threshold = 0.55
```

This improves the balance between **precision and recall**, which is critical in churn prediction problems.

---

# Business Interpretation

The model predictions are converted into business actions:

| Churn Probability | Risk Level  | Business Action         |
| ----------------- | ----------- | ----------------------- |
| < 0.30            | Low Risk    | No action needed        |
| 0.30 – 0.60       | Medium Risk | Send promotional offers |

> 0.60 | High Risk | Trigger retention campaign |

This helps the company **retain customers before they leave**.

---

# Project Structure

```
mlops-customer-churn-api
│
├── app
│   └── main.py              # FastAPI application
│
├── data
│   └── churn.csv            # Dataset
│
├── model.pkl                # Trained model
│
├── training
│   └── train.py             # Training pipeline
│
├── requirements.txt         # Dependencies
│
├── Dockerfile               # Container configuration
│
└── README.md
```

---

# Training Pipeline

The training pipeline performs the following steps:

### 1 Data Cleaning

* Convert `TotalCharges` to numeric
* Remove missing values
* Drop non-predictive fields like `customerID`

### 2 Feature Engineering

Two pipelines are created:

```
Numerical features → StandardScaler
Categorical features → OneHotEncoder
```

Combined using **ColumnTransformer**.

### 3 Model Training

Model used:

```
RandomForestClassifier
```

Hyperparameters optimized using:

```
RandomizedSearchCV
```

Parameters tuned:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf
* max_features

### 4 Experiment Tracking

All experiments are tracked using:

```
MLflow
```

Logged information:

* parameters
* evaluation metrics
* trained models

### 5 Model Export

Final model is saved using:

```
joblib.dump(model, "model.pkl")
```

---

# API Endpoints

The trained model is served using **FastAPI**.

## Health Check

```
GET /
```

Response

```
{
 "status": "Churn Prediction API Running"
}
```

---

## Predict Churn

```
POST /predict
```

Example request:

```
{
 "gender": "Female",
 "SeniorCitizen": 0,
 "Partner": "Yes",
 "Dependents": "No",
 "tenure": 12,
 "PhoneService": "Yes",
 "MultipleLines": "No",
 "InternetService": "Fiber optic",
 "OnlineSecurity": "No",
 "OnlineBackup": "Yes",
 "DeviceProtection": "No",
 "TechSupport": "No",
 "StreamingTV": "Yes",
 "StreamingMovies": "Yes",
 "Contract": "Month-to-month",
 "PaperlessBilling": "Yes",
 "PaymentMethod": "Electronic check",
 "MonthlyCharges": 89.85,
 "TotalCharges": 1089.25
}
```

Response:

```
{
 "churn_prediction": 1,
 "churn_probability": 0.8574,
 "risk_level": "High",
 "recommended_action": "Immediate retention campaign"
}
```

---

# Running the Project Locally

### 1 Clone repository

```
git clone https://github.com/YOUR_USERNAME/mlops-customer-churn-api.git
```

---

### 2 Create virtual environment

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

---

### 3 Install dependencies

```
pip install -r requirements.txt
```

---

### 4 Train the model

```
python train.py
```

---

### 5 Start API server

```
uvicorn app.main:app --reload
```

API docs available at:

```
http://localhost:8000/docs
```

---

# Running with Docker

Build Docker image

```
docker build -t churn-mlops-api .
```

Run container

```
docker run -p 8000:8000 churn-mlops-api
```

API will be available at:

```
http://localhost:8000/docs
```

---

# Cloud Deployment

The API is deployed using:

```
Render Cloud Platform
```

Deployment stack:

```
FastAPI
Docker
Render
```

---

# Future Improvements

Planned improvements include:

* Model monitoring
* prediction logging
* automated retraining pipeline
* CI/CD integration
* feature importance dashboard

---

# Technologies Used

Python
Scikit-learn
Pandas
FastAPI
MLflow
Docker
Uvicorn

---

This project demonstrates an **end-to-end MLOps workflow for deploying machine learning systems in production environments.**
