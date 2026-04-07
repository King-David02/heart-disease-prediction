import pandas as pd
from zenml import step
from src.models import LogisticRegressionModel, evaluate
from src.config import logger

@step(experiment_tracker="mlflow_tracker")
def evaluation_step(
    model: LogisticRegressionModel,
    X_test: pd.DataFrame,
    y_test:pd.Series
):
    logger.info("ZenMl step: Evaluation Step")
    metrics=evaluate(model, X_test, y_test)
    return metrics