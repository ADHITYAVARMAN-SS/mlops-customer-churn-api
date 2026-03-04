import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==============================
# 1️⃣ Load Dataset
# ==============================

df = pd.read_csv("data/churn.csv")

# ==============================
# 2️⃣ Data Cleaning
# ==============================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows with missing values
df = df.dropna()

# Drop customerID (not useful for prediction)
df = df.drop(columns=["customerID"])

# Map target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ==============================
# 3️⃣ Feature / Target Split
# ==============================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ==============================
# 4️⃣ Preprocessing Pipeline
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ==============================
# 5️⃣ Model Pipeline
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ]
)

# ==============================
# 6️⃣ Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 7️⃣ MLflow Tracking
# ==============================

mlflow.set_experiment("telco_churn_prediction")

with mlflow.start_run():

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(pipeline, "model")

    # Save locally for FastAPI
    joblib.dump(pipeline, "model.pkl")

    print("Training complete")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")