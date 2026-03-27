from zenml.pipelines import pipeline
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from src.steps import (
    load_data_step,
    data_validation_step,
    data_preprocessing_step,
    feature_engineering_step,
    train_model_step
)

@pipeline(enable_cache=False, settings={
        "experiment_tracker": MLFlowExperimentTrackerSettings(
        experiment_name="heart-disease-prediction-experiments"
        )
    })
def train_pipeline():
    df=load_data_step()
    df=data_validation_step(df)
    df=data_preprocessing_step(df)
    df=feature_engineering_step(df)
    model, X_test, y_test, metrics=train_model_step(df)