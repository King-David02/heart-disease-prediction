import pandas as pd
from src.config import logger

CONTINUOUS_COLS = [
    "cigsPerDay", "totChol", "sysBP", "diaBP",
    "BMI", "heartRate", "glucose"
]

CATEGORICAL_COLS = ["education", "BPMeds"]

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing started")
    logger.info(f"Before preprocessing df shape is {df.shape}")
    
    initial_row = len(df)
    df = df.drop_duplicates()
    dropped = initial_row - len(df)
    if dropped > 0:
        logger.warning(f"{dropped} duplicates found")
        
    for cols in CONTINUOUS_COLS:
        if df[cols].isnull().sum() > 0:
            median = df[cols].median()
            df[cols].fillna(median)
            logger.debug(f"{cols} null values replaces with {median:.2f}")
            
    for cols in CATEGORICAL_COLS:
        if df[cols].isnull().sum() > 0:
            mode = df[cols].mode()[0]
            df[cols].fillna(mode)
            logger.debug(f"{cols} missing values was replaced with {mode}")
    logger.info(f"Data shape: {df.shape}")        
    df = df.dropna()
    logger.info(f"Final shape after preprocessing is {df.shape}")
    return df