import os
import pandas as pd
from typing import Dict
import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from src.models.base import BaseModel
from src.config import logger
from mlflow.client import MlflowClient


class LogisticRegressionModel(BaseModel):
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        
    def train(self, X_train, y_train):
        logger.info("Training Logistic Regression Model")
        mlflow.autolog(log_models=False, log_model_signatures=True)
        self.model.fit(X_train, y_train)
        #mlflow.sklearn.log_model(self.model, "model")
        logger.info("Logistic Regression Training complete")
        
    def predict(self, X):
        logger.info("Running Predicions")
        return pd.Series(self.model.predict(X))
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    
class SupportVectorMachine(BaseModel):
    
    def __init__(self):
        self.model = SVC(class_weight='balanced', probability=True)
    
    def train(self, X_train, y_train):
        logger.info("SVC training")
        mlflow.autolog(log_models=False, log_model_signatures=True)
        self.model.fit(X_train, y_train)
        logger.info("SVC training complete")
        
    def predict(self, X):
        logger.info("Running Predicions")
        return pd.Series(self.model.predict(X))
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    
    def __init__(self):
        self.model = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            random_state=42
        )
    
    def train(self, X_train, y_train):
        logger.info("random forest training")
        # mlflow.autolog(log_models=False, log_model_signatures=True)
        self.model.fit(X_train, y_train)
        logger.info("random forest training complete")
        
    def predict(self, X):
        logger.info("Running Predicions")
        return pd.Series(self.model.predict(X))
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
   
    
class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=5,
            random_state=42,
            eval_metric="logloss"
        )

    def train(self, X, y):
        logger.info("Training XGBoost")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42
        )

    def train(self, X, y):
        logger.info("Training LightGBM")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def train_model(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models: Dict[str, BaseModel] = {
    "lr": LogisticRegressionModel(),
    "svc": SupportVectorMachine(),
    "rf": RandomForestModel(),
    "xgb": XGBoostModel(),
    "lgbm": LightGBMModel(),
}

    results = {}
    registered_model_name = "Heart-disease-models"
    
        
    for name, model in models.items():
            logger.info(f"Training {name}")
            model.train(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            y_pred = (y_proba > 0.3).astype(int)

            report = classification_report(y_test, y_pred, output_dict=True)

            f1 = report["1"]["f1-score"]
            recall = report["1"]["recall"]
            precision = report["1"]["precision"]

            logger.info(
                f"{name} -> F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}"
            )

            mlflow.log_metric(f"{name}_f1_class1", f1)
            mlflow.log_metric(f"{name}_recall_class1", recall)
            mlflow.log_metric(f"{name}_precision_class1", precision)

            mlflow.log_dict(report, f"{name}_classification_report.json")

            results[name] = {
                "model": model,
                "f1": f1,
                "recall": recall
            }

    def score(m):
        if m["recall"] < 0.3:
            return 0
        return m["f1"]

    best_model_name = max(results, key=lambda x: score(results[x]))
    best_model = results[best_model_name]["model"]

    logger.info(f"Best model: {best_model_name}")

    mlflow.log_param("best_model", best_model_name)
    
    if isinstance(best_model.model, XGBClassifier):
        mlflow.xgboost.log_model(xgb_model=best_model.model, artifact_path="models", registered_model_name=registered_model_name)
    elif isinstance(best_model.model, LGBMClassifier):
        mlflow.lightgbm.log_model(xgb_model=best_model.model, artifact_path="models", registered_model_name=registered_model_name)
    else:

        mlflow.sklearn.log_model(
            sk_model=best_model.model,
            artifact_path="models",
            registered_model_name=registered_model_name
        )
    
    tracking_uri = mlflow.get_tracking_uri()
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)

    logger.info(f"Tracking URI: {tracking_uri}")
    logger.info(f"Registered model name: {registered_model_name}")

    import time
    time.sleep(3)

    mlflow_client = MlflowClient()

    # Retry logic
    for attempt in range(5):
        try:
            versions = mlflow_client.search_model_versions(f"name='{registered_model_name}'")
            if versions:
                model_version = str(max(int(v.version) for v in versions))
                break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2)

    logger.info(f"Registered model version: {model_version}")
    return best_model, X_test, y_test, model_version, registered_model_name