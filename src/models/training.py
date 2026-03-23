import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from src.models.base import BaseModel
from src.config import logger

class LogisticRegressionModel(BaseModel):
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
    def train(self, X_train, y_train):
        logger.info("Training Logistic Regression Model")
        mlflow.autolog()
        self.model.fit(X_train, y_train)
        logger.info("Logistic Regression Training complete")
        
    def predict(self, X):
        logger.info("Running Predicions")
        return pd.Series(self.model.predict(X))
    
    def evaluate(self, X_test, y_test):
        logger.info("Evaluating Model")
        predictions = self.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, predictions),
            "classification_report": classification_report(y_test, predictions),
        }
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        return metrics


def train_model(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    return model, X_test, y_test, metrics