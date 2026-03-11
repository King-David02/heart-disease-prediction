from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    project_name: str = "heart-disease-prediction"
    environment: str = "development"
    mlflow_experiment_name: str = "heart-disease-prediction"
    raw_data_path: str = str(BASE_DIR / "data" / "raw" / "framingham.csv")
    processed_data_path: str = str(BASE_DIR / "data" / "processed")
    
    class config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8" 
