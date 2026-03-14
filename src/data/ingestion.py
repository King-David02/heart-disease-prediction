import pandas as pd
from src.config import settings, logger

def load_data() -> pd.DataFrame:
    logger.info(f"loading Data from {settings.raw_data_path}")
    df = pd.read_csv(settings.raw_data_path)
    logger.info(f"loaded {len(df)} rows and {len(df.columns)} columns")
    return df