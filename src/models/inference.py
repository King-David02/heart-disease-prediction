import mlflow
import pandas as pd
from src.config import logger
from zenml import step

@step(experiment_tracker="mlflow_tracker")
def load_model(model_name: str = "Heart-disease-models"):
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
    logger.info("Loading Model from Mlflow")
    return model


def run_inference(model, data: pd.DataFrame) -> pd.Series:
    prediction = model.predict(data)
    logger.info("Inference complete")
    return prediction