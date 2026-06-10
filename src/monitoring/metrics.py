from prometheus_client import Counter, Histogram, Gauge, generate_latest
from src.config import logger

PREDICTION_TOTAL = Counter(
    "total_number of prediction",
    "Total number of predictions made",
    ["prediction"]
)

PREDICTION_LATENCY = Histogram(
    "latency",
    "Time taken to make a prediction",
    buckets=[0.01, 0.05, 0.1, 0.25, 1.0]
)

MODEL_ACCURACY = Gauge(
    "Model Accuracy",
    "Current Model Accuracy"
)

def record_prediction(prediction: int, latency: float):
    PREDICTION_TOTAL.labels(
        prediction=str(prediction)
    ).inc()
    
    PREDICTION_LATENCY.observe(latency)
    logger.debug(f"Metrics recorded — prediction={prediction}, latency={latency:.3f}s")