import pandas as pd
from sklearn.model_selection import train_test_split
from evidently import Report, DataDefinition, Dataset, BinaryClassification
from evidently.presets import DataDriftPreset, DataSummaryPreset
from src.config import logger

def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str = "monitoring/reports/drift_report.html"
) -> dict:
    logger.info("Running drift detection")
    reff_data=Dataset.from_pandas(reference_data)
    curr_data=Dataset.from_pandas(current_data)

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ], 
    )

    report = report.run(curr_data, reff_data)
    report.save_html(output_path)
    logger.info(f"drift saved to {output_path}")