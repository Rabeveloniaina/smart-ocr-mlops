import os
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Union, Optional
from loguru import logger

from src.models.crnn import CRNN
from src.models.ctc_decoder import CTCDecoder, DecodingResult
from src.preprocessing import ImagePreprocessor

try:
    import bidi
    import bidi.algorithm
    bidi.get_display = bidi.algorithm.get_display
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


class OCRPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        params_path: str = "params.yaml",
        device: Optional[str] = None
    ):
        with open(params_path, "r", encoding="utf-8") as f:
            self.params = yaml.safe_load(f)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.preprocessor = ImagePreprocessor(
            target_height=self.params["data"]["image_height"],
            target_width=self.params["data"]["image_width"],
            denoise=True
        )

        self.charset = self.params["data"]["charset"]
        self.decoder = CTCDecoder(charset=self.charset)

        if model_path is None:
            model_path = self.params["training"]["best_model_path"]
        
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        self.easy_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easy_reader = easyocr.Reader(['fr', 'en'], gpu=torch.cuda.is_available(), verbose=False)
                logger.info("EasyOCR initialized.")
            except Exception as e:
                logger.warning(f"EasyOCR fallback to CRNN: {e}")

        logger.info(f"OCRPredictor ready on {self.device}")

    def _load_model(self) -> CRNN:
        model = CRNN.from_params()
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.warning(f"Model file not found at {self.model_path}")
        return model

    def predict_image(
        self,
        image_input: Union[str, bytes, np.ndarray],
        use_beam_search: bool = False,
        beam_width: int = 5
    ) -> Dict[str, Any]:
        if isinstance(image_input, (str, Path)):
            raw_img = cv2.imread(str(image_input))
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            raw_img = image_input
        else:
            raise ValueError("Unsupported input format")

        if raw_img is None:
            raise ValueError("Could not read input image")

        start_time = cv2.getTickCount()
        
        if self.easy_reader is not None:
            try:
                ocr_results = self.easy_reader.readtext(raw_img)
                if ocr_results:
                    lines_text = []
                    confs = []
                    boxes = []
                    for bbox, text, prob in ocr_results:
                        clean_t = text.strip()
                        if clean_t:
                            lines_text.append(clean_t)
                            confs.append(prob * 100)

                            pts = [[int(pt[0]), int(pt[1])] for pt in bbox] if isinstance(bbox, (list, tuple, np.ndarray)) else []
                            boxes.append({"text": clean_t, "confidence": round(float(prob * 100), 1), "box": pts})
                    if lines_text:
                        elapsed_ms = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000.0
                        return {
                            "text": "\n".join(lines_text),
                            "confidence": round(float(np.mean(confs)), 2),
                            "num_lines": len(lines_text),
                            "latency_ms": round(elapsed_ms, 1),
                            "engine": "EasyOCR (ResNet + LSTM + CTC)",
                            "details": boxes
                        }
            except Exception as e:
                logger.warning(f"EasyOCR error, using CRNN: {e}")

        gray = self.preprocessor.to_grayscale(raw_img)
        lines = self.preprocessor.segment_lines(gray)

        predicted_texts = []
        confidences = []

        for line_img in lines:
            img_processed = self.preprocessor.preprocess_image(line_img, enhance_contrast=True)
            img_norm = img_processed.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
            tensor = (tensor - 0.5) / 0.5
            tensor = tensor.to(self.device)

            with torch.no_grad():
                log_probs = self.model(tensor)

            results = self.decoder.decode_batch(
                log_probs,
                use_beam_search=use_beam_search,
                beam_width=beam_width
            )
            res: DecodingResult = results[0]
            if res.text.strip():
                predicted_texts.append(res.text.strip())
                confidences.append(res.confidence)

        elapsed_ms = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000.0
        final_text = "\n".join(predicted_texts) if predicted_texts else ""
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": final_text,
            "confidence": round(avg_confidence, 2),
            "num_lines": len(lines),
            "latency_ms": round(elapsed_ms, 1),
            "engine": "Custom CRNN (PyTorch + CTC)",
            "details": [{"text": t, "confidence": c} for t, c in zip(predicted_texts, confidences)]
        }


if __name__ == "__main__":
    predictor = OCRPredictor()
    dummy_img = np.ones((32, 256), dtype=np.uint8) * 255
    res = predictor.predict_image(dummy_img)
    print(res)
