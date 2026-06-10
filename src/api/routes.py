import time
import pandas as pd
from fastapi import APIRouter, HTTPException
from src.api.schemas import PatientData, PredictionResponse
from src.models import load_model, run_inference
from src.monitoring.metrics import record_prediction
from src.config import logger, settings

router = APIRouter()

try:
    model = load_model()
    logger.info("Model loading sucessful")
except Exception as e:
    logger.error(f"failed to load model: {e}")
    model = None
    
@router.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@router.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    start = time.time()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        data = pd.DataFrame([patient.model_dump()])
        data["pulBP"] = data["sysBP"] - data["diaBP"]
        
        predictions = run_inference(model, data)
        latency = time.time() - start
        prediction = int(predictions[0])
        # probability =float(model.predict_proba(data)[0][1])
        
        # if probability < 0.3:
        #     risk_level = "low"
            
        # elif probability < 0.6:
        #     risk_level = "Medium"
            
        # else:
        #     risk_level = "High"
            
        # logger.info(f"prediction: {prediction}, probability: {probability:.4f}")
        
        record_prediction(prediction, latency)
        return PredictionResponse(
            prediction=prediction,
            latency=latency
            # probability=round(probability, 4),
            # risk_level=risk_level
        )
        
    except Exception as e:
        logger.error((f"Prediction failed {e}"))
        raise HTTPException(status_code=500, detail=str(e))