from zenml.pipelines import pipeline
from zenml import Model
from src.steps import (
    load_data_step,
    data_validation_step,
    data_preprocessing_step,
    feature_engineering_step,
    train_model_step,
    evaluation_step,
    promote_model_step
)


@pipeline(enable_cache=False)
def train_pipeline():

    df=load_data_step()
    df=data_validation_step(df)
    df=data_preprocessing_step(df)
    df=feature_engineering_step(df)
    model, X_test, y_test, version, name = train_model_step(df)
    evaluation_step(model, X_test, y_test)
    promote_model_step(name, version)