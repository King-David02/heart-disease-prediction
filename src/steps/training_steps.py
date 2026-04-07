from zenml import step
from typing import Annotated, Tuple
import pandas as pd
from src.models import train_model, LogisticRegressionModel
from src.config import logger

@step(experiment_tracker="mlflow_tracker") 
def train_model_step(df: pd.DataFrame) -> Tuple[
    Annotated[LogisticRegressionModel, "model"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_test"],
    Annotated[str, "run_id"]
]:
    logger.info("ZenML step: Training model")
    model, X_test, y_test, run_id  = train_model(df)
    return model, X_test, y_test, run_id