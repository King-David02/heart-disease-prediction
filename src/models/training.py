import os
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from src.models.base import BaseModel
from src.config import logger, settings

class LogisticRegressionModel(BaseModel):
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.dagshub_token

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
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
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_roc_auc": roc_auc_score(y_test, predictions),
            "test_f1": f1_score(y_test, predictions)
        }
        mlflow.log_metrics(metrics)
        
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.matshow(cm, cmap=plt.cm.Blues)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        mlflow.log_figure(fig, "test_confusion_matrix.png")
        plt.close(fig)

        report = classification_report(y_test, predictions)

        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/classification_report.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact("evaluation/classification_report.txt", artifact_path="evaluation")
                
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
        logger.info(f"Test F1: {metrics['test_f1']:.4f}")
        return metrics


def train_model(df: pd.DataFrame) -> tuple:
    
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    dagshub.init(repo_owner='King-David02', repo_name='heart-disease-prediction', mlflow=True)
    mlflow.set_experiment("heart-disease-prediction")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    return model, X_test, y_test, metrics