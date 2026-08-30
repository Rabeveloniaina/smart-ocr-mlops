import json
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from src.predict import OCRPredictor


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _predictor: Optional[OCRPredictor] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, params_path: str = "params.yaml"):
        if self._predictor is None:
            logger.info("Loading model into memory...")
            self._predictor = OCRPredictor(params_path=params_path)
            logger.info("Model loaded in ModelManager.")

    def get_predictor(self) -> OCRPredictor:
        if self._predictor is None:
            self.initialize()
        return self._predictor

    def get_metadata(self) -> Dict[str, Any]:
        meta_path = Path("models/model_metadata.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "model_name": "CRNN",
            "backbone": "resnet18",
            "version": "1.0.0",
            "status": "active"
        }


model_manager = ModelManager()
