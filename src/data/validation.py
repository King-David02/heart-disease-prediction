import pandas as pd
from src.config import logger

EXPECTED_COLUMNS = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate",
    "glucose", "TenYearCHD"
]

def data_validation(df: pd.DataFrame) -> bool:
    logger.info("Validating data")
    
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing Columns: {missing_cols}")
    
    for cols in df.columns:
        null_count = df[cols].isnull().sum()
        if null_count > 0:
            logger.warning(f"Columns {cols} has {null_count} null values")
            
    if df.duplicated().sum() > 0:
        logger.warning(f"found {df.duplicated.sum()} duplicate rows")
        
    return True