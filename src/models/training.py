import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from src.models.base import BaseModel
from src.config import logger

class LogisticRegressionModel(BaseModel):
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        
    def train(self, X_train, y_train):
        logger.info("Training Logistic Regression Model")
        mlflow.autolog()
        self.model.fit(X_train, y_train)
        #mlflow.sklearn.log_model(self.model, "model")
        logger.info("Logistic Regression Training complete")
        
    def predict(self, X):
        logger.info("Running Predicions")
        return pd.Series(self.model.predict(X))
    
    
class SupportVectorMachine(BaseModel):
    
    def __init__(self):
        self.model = SVC(class_weight='balanced')
    
    def train(self, X_train, y_train):
        logger.info("SVC training")
        mlflow.autolog()
        self.model.fit(X_train, y_train)
        logger.info("SVC training complete")
        
    def predict(self, X):
        logger.info("Running Predicions")
        return pd.Series(self.model.predict(X))


def train_model(df: pd.DataFrame, model_type: str) -> tuple:
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "lr":
        model = LogisticRegressionModel()
        model.train(X_train, y_train)
        
    elif model_type == "svc":
        model = SupportVectorMachine()
        model.train(X_train, y_train)

    run_id = mlflow.active_run().info.run_id
    logger.info(f"MLflow run_id: {run_id}")
    return model, X_test, y_test, run_id