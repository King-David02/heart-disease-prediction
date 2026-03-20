import pandas as pd
from zenml import step
from src.data import load_data, data_validation, data_preprocessing, feature_engineering
from src.config import logger

@step
def load_data_step() -> pd.DataFrame:
    logger.info("ZenML step: loading data")
    return load_data()

@step
def data_validation_step(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("ZenML step: validating data")
    data_validation(df)
    return df

@step
def data_preprocessing_step(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("ZenML step: Preprocessing data")
    return data_preprocessing(df)

@step
def feature_engineering_step(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("ZenML step: Feature engineering")
    return feature_engineering(df)