import os
import sys
import subprocess
from pathlib import Path
from loguru import logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def run_command(cmd, desc):
    logger.info(f"Execution: {desc}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        logger.error(f"Error {desc}: {res.stderr}")
        return False
    logger.success(f"{desc} ok")
    return True


def main():
    py = f'"{sys.executable}"'

    run_command(f"{py} scripts/generate_synthetic_data.py", "generate_synthetic_data")
    run_command(f"{py} src/preprocessing.py", "preprocessing")
    run_command(f"{py} scripts/split_dataset.py", "split_dataset")
    run_command(f"{py} -m src.train", "train")
    run_command(f"{py} -m src.evaluate", "evaluate")

    from src.predict import OCRPredictor
    import cv2
    import numpy as np

    predictor = OCRPredictor()
    dummy_img = np.ones((32, 128), dtype=np.uint8) * 255
    cv2.putText(dummy_img, "MLOps", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    res = predictor.predict_image(dummy_img)
    logger.info(f"Inference result: text='{res['text']}', conf={res['confidence']}%")

    run_command(f"{py} -m monitoring.evidently_report", "evidently_report")


if __name__ == "__main__":
    main()
