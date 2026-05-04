from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    project_name: str = "heart-disease-prediction"
    environment: str = "development"
    mlflow_experiment_name: str = "heart-disease-prediction"
    mlflow_tracking_uri: str = ""
    dagshub_username: str = ""
    dagshub_token: str = ""
    repo_name: str = ""
    raw_data_path: str = str(BASE_DIR / "data" / "raw" / "framingham.csv")
    processed_data_path: str = str(BASE_DIR / "data" / "processed")
    
    api_secret_key: str = "dev-secret-key"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
        
    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8" 
