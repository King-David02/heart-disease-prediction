from zenml import step
from typing import Annotated
import pandas as pd
from src.models.training import LogisticRegressionModel, train_model
from src.config import logger

@step(experiment_tracker="mlflow_tracker")
def train_model_step(df: pd.DataFrame) -> tuple[
    Annotated[LogisticRegressionModel, "model"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_test"],
    Annotated[dict, "metrics"]
]:
    logger.info("ZenML step: Training model")
    model, X_test, y_test, metrics = train_model(df)
    return model, X_test, y_test, metrics