import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Optional: uncomment if you want to try XGBoost
# from xgboost import XGBClassifier

# ==============================
#  Load & Clean Data
# ==============================
df = pd.read_csv("data/churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ==============================
#  Preprocessing
# ==============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ==============================
#  Classifier Choice
# ==============================
rf = RandomForestClassifier(
    class_weight="balanced",  # balances imbalanced classes
    random_state=42
)

# Optional: Use XGBoost instead
# rf = XGBClassifier(
#     scale_pos_weight=(y==0).sum() / (y==1).sum(),
#     n_estimators=500,
#     max_depth=8,
#     learning_rate=0.05,
#     eval_metric="logloss",
#     random_state=42
# )

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf),
    ]
)

# ==============================
#  Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
#  Hyperparameter Search
# ==============================
param_dist = {
    "classifier__n_estimators": [300, 400, 500, 600],
    "classifier__max_depth": [8, 10, 12, 15, None],
    "classifier__min_samples_split": [2, 5, 10, 15],
    "classifier__min_samples_leaf": [1, 2, 3, 4],
    "classifier__max_features": ["sqrt", "log2"]
}

search = RandomizedSearchCV(    
    pipeline,
    param_distributions=param_dist,
    n_iter=25,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=42
)

# ==============================
#  MLflow Tracking & Model Training
# ==============================
mlflow.set_experiment("telco_churn_prediction")

with mlflow.start_run():
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Predict probabilities
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # ==============================
    #  Threshold Optimization (Precision-F1 tradeoff)
    # ==============================
    thresholds = np.arange(0.3, 0.8, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_threshold_precision = 0.5

    for t in thresholds:
        y_pred_temp = (y_prob > t).astype(int)
        f1_temp = f1_score(y_test, y_pred_temp)
        precision_temp = precision_score(y_test, y_pred_temp)

        # Track best F1
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = t

        # Track best Precision (with acceptable recall)
        if precision_temp > best_precision and recall_score(y_test, y_pred_temp) > 0.6:
            best_precision = precision_temp
            best_threshold_precision = t

    # Choose which threshold to use
    threshold_to_use = best_threshold_precision  # for higher precision
    y_pred = (y_prob > threshold_to_use).astype(int)

    # ==============================
    # Metrics
    # ==============================
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ==============================
    # Log Metrics & Model
    # ==============================
    mlflow.log_params(search.best_params_)
    mlflow.log_param("best_threshold_f1", best_threshold)
    mlflow.log_param("best_threshold_precision", best_threshold_precision)
    mlflow.log_param("threshold_used", threshold_to_use)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(best_model, "model")
    joblib.dump(best_model, "model.pkl")

    # ==============================
    # Feature Importance
    # ==============================
    model_rf = best_model.named_steps["classifier"]
    feat_names = (
        best_model.named_steps["preprocessor"]
        .transformers_[0][2] +  # numeric
        list(best_model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_cols))
    )

    importances = model_rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12,6))
    plt.bar(range(20), importances[indices][:20])
    plt.xticks(range(20), [feat_names[i] for i in indices[:20]], rotation=90)
    plt.title("Top 20 Feature Importances")
    plt.show()

    # ==============================
    #  Print Summary
    # ==============================
    #print("\nBest Parameters:", search.best_params_)
    #print("Best Threshold (F1):", best_threshold)
    #print("Best Threshold (Precision-focused):", best_threshold_precision)
    #print("Threshold Used:", threshold_to_use)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")