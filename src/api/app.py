from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from src.api.routes import router
from src.api.middleware import log_requests
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict 10-year risk of coronary heart disease",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.middleware("http")(log_requests)
app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {
    "message": "Heart Disease Prediction API",
    "docs": "/docs",
    "health": "/api/v1/health"
}
    
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)