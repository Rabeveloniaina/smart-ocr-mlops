import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger
import editdistance

from src.models.crnn import CRNN
from src.models.ctc_decoder import CTCDecoder
from src.data.dataset import OCRDataModule


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_cer(predictions: List[str], ground_truths: List[str]) -> float:
    total_distance = 0
    total_length = 0
    
    for pred, ref in zip(predictions, ground_truths):
        total_distance += editdistance.eval(pred, ref)
        total_length += len(ref)
        
    return total_distance / max(1, total_length)


def calculate_wer(predictions: List[str], ground_truths: List[str]) -> float:
    total_distance = 0
    total_words = 0
    
    for pred, ref in zip(predictions, ground_truths):
        pred_words = pred.strip().split()
        ref_words = ref.strip().split()
        total_distance += editdistance.eval(pred_words, ref_words)
        total_words += len(ref_words)
        
    return total_distance / max(1, total_words)


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    correct = sum(1 for p, r in zip(predictions, ground_truths) if p == r)
    return correct / max(1, len(ground_truths))


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    decoder: CTCDecoder,
    device: torch.device,
    use_beam_search: bool = False
) -> Dict[str, Any]:
    model.eval()
    predictions = []
    ground_truths = []
    confidences = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            
            log_probs = model(images)
            results = decoder.decode_batch(log_probs, use_beam_search=use_beam_search)
            
            for res, target in zip(results, texts):
                predictions.append(res.text)
                ground_truths.append(target)
                confidences.append(res.confidence)
                
    cer = calculate_cer(predictions, ground_truths)
    wer = calculate_wer(predictions, ground_truths)
    accuracy = calculate_accuracy(predictions, ground_truths)
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    
    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "accuracy": round(accuracy, 4),
        "avg_confidence": round(avg_confidence, 2),
        "samples_evaluated": len(predictions),
        "predictions_sample": [
            {"target": gt, "predicted": pred}
            for gt, pred in zip(ground_truths[:10], predictions[:10])
        ]
    }


def main():
    params = load_params("params.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(params["training"]["best_model_path"])
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
        
    charset = params["data"]["charset"]
    decoder = CTCDecoder(charset=charset)
    
    model = CRNN.from_params("params.yaml")
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    dm = OCRDataModule("params.yaml")
    test_loader = dm.test_dataloader()
    
    logger.info("Evaluating model on test dataset...")
    metrics = evaluate_model(model, test_loader, decoder, device, use_beam_search=True)
    
    logger.info(f"Test CER: {metrics['cer'] * 100:.2f}% | WER: {metrics['wer'] * 100:.2f}% | Acc: {metrics['accuracy'] * 100:.2f}%")
    
    out_file = Path("models/evaluation_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved evaluation metrics to {out_file}")


if __name__ == "__main__":
    main()
