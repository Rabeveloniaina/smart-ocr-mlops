import time
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger

from api.schemas import OCRPredictionResponse, HealthResponse, ModelInfoResponse
from api.model_loader import model_manager
from monitoring.metrics_store import log_inference_for_monitoring

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OCR API server...")
    model_manager.initialize()
    yield
    logger.info("Stopping OCR API server...")


app = FastAPI(
    title="Smart OCR API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path(__file__).parent / "templates" / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    predictor = model_manager.get_predictor()
    uptime = time.time() - START_TIME
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 2),
        model_loaded=predictor.model is not None,
        device=str(predictor.device),
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    meta = model_manager.get_metadata()
    metrics_path = Path("models/evaluation_metrics.json")
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    return ModelInfoResponse(
        model_name=meta.get("model_name", "CRNN"),
        architecture="ResNet18 + BiLSTM + CTC",
        version=meta.get("version", "1.0.0"),
        trained_at=meta.get("trained_at", "N/A"),
        dataset_used=meta.get("dataset", "Synthetic OCR"),
        metrics=metrics,
        mlflow_run_id=meta.get("mlflow_run_id", None)
    )


@app.post(
    "/predict",
    response_model=OCRPredictionResponse,
    status_code=status.HTTP_200_OK
)
async def predict_handwriting(
    file: UploadFile = File(...),
    use_beam_search: bool = Query(False)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image format (.png, .jpg, .jpeg)."
        )

    try:
        t0 = time.time()
        image_bytes = await file.read()
        
        predictor = model_manager.get_predictor()
        result = predictor.predict_image(image_bytes, use_beam_search=use_beam_search)
        
        duration_ms = (time.time() - t0) * 1000
        
        log_inference_for_monitoring(
            filename=file.filename or "unknown",
            predicted_text=result["text"],
            confidence=result["confidence"],
            latency_ms=duration_ms
        )
        
        return OCRPredictionResponse(
            text=result["text"],
            confidence=result["confidence"],
            inference_time_ms=round(duration_ms, 2)
        )

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during transcription: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
