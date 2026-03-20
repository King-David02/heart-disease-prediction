import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import logger

CONTINUOUS_COLS = [
    "age", "cigsPerDay", "totChol", "sysBP",
    "diaBP", "BMI", "heartRate", "glucose"
]

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("starting feature engineering")
    
    df = df.copy()
    df["pulBP"] = df["sysBP"] - df["diaBP"]
    logger.debug("created pulse pressure column")
    
    scaler = StandardScaler()
    df[CONTINUOUS_COLS] = scaler.fit_transform(df[CONTINUOUS_COLS])
    logger.debug(f"scaled {len(CONTINUOUS_COLS)} continuous columns")
    
    logger.info(f"Feature engineering complete, shape: {df.shape}")
    
    return df