import os
import mlflow
import dagshub
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from src.config import logger, settings


def evaluate(model, X_test, y_test):
    logger.info("Evaluating Model")
    #mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    predictions = model.predict(X_test)
    metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_roc_auc": roc_auc_score(y_test, predictions),
        "test_f1": f1_score(y_test, predictions),
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

    mlflow.log_artifact(
        "evaluation/classification_report.txt", artifact_path="evaluation"
    )

    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    logger.info(f"Test F1: {metrics['test_f1']:.4f}")
    return metrics
