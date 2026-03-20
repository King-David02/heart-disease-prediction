from zenml.pipelines import pipeline
from src.steps import (
    load_data_step,
    data_validation_step,
    data_preprocessing_step,
    feature_engineering_step
)

@pipeline
def train_pipeline():
    df=load_data_step()
    df=data_validation_step(df)
    df=data_preprocessing_step(df)
    df=feature_engineering_step(df)