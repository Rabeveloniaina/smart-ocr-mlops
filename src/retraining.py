import os
import json
import yaml
import time
import subprocess
import torch
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from src.evaluate import evaluate_model
from src.models.crnn import CRNN
from src.models.ctc_decoder import CTCDecoder
from src.data.dataset import OCRDataModule


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AutoRetrainingPipeline:
    def __init__(self, params_path: str = "params.yaml"):
        self.params = load_params(params_path)
        self.best_model_path = Path(self.params["training"]["best_model_path"])
        self.improvement_threshold = self.params["retraining"].get("improvement_threshold", 0.01)

    def run_training_pipeline(self) -> bool:
        logger.info("Running training pipeline for challenger model...")
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            env["MLFLOW_ALLOW_FILE_STORE"] = "true"
            cmd = ["python", "-m", "src.train"]
            subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("Training pipeline finished.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")
            return False

    def evaluate_and_promote(self) -> bool:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        charset = self.params["data"]["charset"]
        decoder = CTCDecoder(charset=charset)
        dm = OCRDataModule("params.yaml")
        val_loader = dm.val_dataloader()

        if not self.best_model_path.exists():
            logger.error("No challenger model checkpoint found.")
            return False

        challenger_model = CRNN.from_params().to(device)
        checkpoint = torch.load(self.best_model_path, map_location=device)
        challenger_model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        challenger_metrics = evaluate_model(challenger_model, val_loader, decoder, device)

        logger.info(f"Challenger Val CER: {challenger_metrics['cer']*100:.2f}%")

        metrics_file = Path("models/metrics.json")
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                old_metrics = json.load(f)
            champion_cer = old_metrics.get("best_val_cer", 1.0)
        else:
            champion_cer = 1.0

        logger.info(f"Champion Val CER: {champion_cer*100:.2f}%")

        cer_diff = champion_cer - challenger_metrics["cer"]
        if cer_diff >= self.improvement_threshold or champion_cer == 1.0:
            logger.info(f"Challenger promoted (CER gain: {cer_diff*100:.2f}%).")
            metadata = {
                "version": f"2.0.{int(time.time())}",
                "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "val_cer": f"{challenger_metrics['cer']*100:.2f}%",
                "status": "deployed"
            }
            with open("models/model_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            return True
        else:
            logger.info(f"Challenger rejected: insufficient gain ({cer_diff*100:.2f}% < {self.improvement_threshold*100:.2f}%).")
            return False

    def trigger_full_cycle(self):
        logger.info("Triggering retraining cycle...")
        success = self.run_training_pipeline()
        if success:
            return self.evaluate_and_promote()
        return False


if __name__ == "__main__":
    pipeline = AutoRetrainingPipeline()
    pipeline.trigger_full_cycle()
