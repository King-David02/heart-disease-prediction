from pydantic import BaseModel, Field

class PatientData(BaseModel):
    male: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    age: float = Field(..., ge=0, description="Age in years")
    education: float = Field(..., ge=1, le=4, description="Education level 1-4")
    currentSmoker: int = Field(..., ge=0, le=1)
    cigsPerDay: float = Field(..., ge=0)
    BPMeds: float = Field(..., ge=0, le=1)
    prevalentStroke: int = Field(..., ge=0, le=1)
    prevalentHyp: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    totChol: float = Field(..., ge=0)
    sysBP: float = Field(..., ge=0)
    diaBP: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    heartRate: float = Field(..., ge=0)
    glucose: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = high risk, 0 = low risk")
    latency: float
    # probability: float = Field(..., description="Probability of heart disease")
    # risk_level: str = Field(..., description="Low, Medium or High")