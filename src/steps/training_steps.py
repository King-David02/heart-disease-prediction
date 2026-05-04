from zenml import step
from typing import Annotated, Tuple, Any
import pandas as pd
from src.models import train_model
from src.models.base import BaseModel
from src.config import logger

@step(experiment_tracker="mlflow_tracker") 
def train_model_step(
    df: pd.DataFrame
    ) -> Tuple[
    Annotated[BaseModel, "model"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_test"],
    Annotated[str, "model_version"],
    Annotated[str, "registered_model_name"],
]:
    model, X_test, y_test, model_version, registered_model_name  = train_model(df)
    return model, X_test, y_test, model_version, registered_model_name