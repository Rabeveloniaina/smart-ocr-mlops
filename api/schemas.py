from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any


class OCRPredictionResponse(BaseModel):
    text: str
    confidence: float
    inference_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    uptime_seconds: float
    model_loaded: bool
    device: str
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    version: str
    trained_at: Optional[str] = None
    dataset_used: Optional[str] = None
    metrics: Dict[str, Any]
    mlflow_run_id: Optional[str] = None
