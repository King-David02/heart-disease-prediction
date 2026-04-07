from zenml.pipelines import pipeline
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step
)
from src.steps import (
    load_data_step,
    data_validation_step,
    data_preprocessing_step,
    feature_engineering_step,
    train_model_step,
    evaluation_step,
)

@pipeline(enable_cache=False)
def train_pipeline():
    df=load_data_step()
    df=data_validation_step(df)
    df=data_preprocessing_step(df)
    df=feature_engineering_step(df)
    model, X_test, y_test, run_id = train_model_step(df)
    metrics=evaluation_step(model, X_test, y_test)
    #version=register_model_step(run_id, metrics)
    mlflow_register_model_step(
        model=model,
        name="lr_model",
        trained_model_name="model",
        run_id=run_id
    )