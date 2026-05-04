import pandas as pd
from zenml import step
from typing import Annotated, Dict
from src.models import evaluate
from src.models.base import BaseModel
from src.config import logger

@step(experiment_tracker="mlflow_tracker")
def evaluation_step(
    model: BaseModel,
    X_test: pd.DataFrame,
    y_test:pd.Series
) -> Annotated[dict, "metrics"]:
    logger.info("ZenMl step: Evaluation Step")
    metrics=evaluate(model, X_test, y_test)
    return metrics