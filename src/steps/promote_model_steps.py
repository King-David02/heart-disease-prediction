from zenml import step
import mlflow
from mlflow.tracking import MlflowClient
from src.config import logger
logger.info("Promoting Model to production")

@step(experiment_tracker="mlflow_tracker")
def promote_model_step(
    model_name: str,
    version: str
):
    client = MlflowClient()
    tracking_uri = mlflow.get_tracking_uri()
    logger.info(f"Tracking URI in promote step: {tracking_uri}")
    
    client = MlflowClient(tracking_uri=tracking_uri)

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )

    logger.info(f"Model {model_name} v{version} promoted to PRODUCTION")