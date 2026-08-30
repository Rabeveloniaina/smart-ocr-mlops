import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from tqdm import tqdm

from src.models.crnn import CRNN
from src.models.ctc_decoder import CTCDecoder
from src.data.dataset import OCRDataModule
from src.data.augmentation import get_training_augmentation
from src.evaluate import evaluate_model


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.CTCLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 5.0
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in tqdm(dataloader, desc="Train", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"]

        optimizer.zero_grad()

        log_probs = model(images)
        T, B, _ = log_probs.shape
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, num_batches)


def main():
    params = load_params("params.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    seed = params["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dm = OCRDataModule("params.yaml")
    train_aug = get_training_augmentation(params)
    train_loader = dm.train_dataloader(transform=train_aug)
    val_loader = dm.val_dataloader()

    charset = params["data"]["charset"]
    decoder = CTCDecoder(charset=charset)

    model = CRNN.from_params("params.yaml").to(device)
    logger.info(f"Model params: {model.count_parameters():,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    lr = float(params["training"]["learning_rate"])
    weight_decay = float(params["training"]["weight_decay"])
    epochs = int(params["training"]["epochs"])
    patience = int(params["training"]["early_stopping_patience"])

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow_cfg = params["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "smart-ocr-manuscrit"))

    best_cer = float("inf")
    patience_counter = 0
    checkpoint_dir = Path(params["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path(params["training"]["best_model_path"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_history = {"epochs": [], "train_loss": [], "val_loss": [], "val_cer": [], "val_wer": []}

    logger.info("Starting MLflow run...")
    with mlflow.start_run(run_name=f"CRNN_{time.strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "model": "CRNN",
            "epochs": epochs,
            "batch_size": params["training"]["batch_size"],
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "hidden_size": params["model"]["rnn_hidden_size"],
            "charset_size": len(charset),
            "image_height": params["data"]["image_height"],
            "image_width": params["data"]["image_width"],
        })

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                gradient_clip=params["training"]["gradient_clip"]
            )
            scheduler.step()

            val_metrics = evaluate_model(model, val_loader, decoder, device)
            val_cer = val_metrics["cer"]
            val_wer = val_metrics["wer"]
            val_acc = val_metrics["accuracy"]

            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val CER: {val_cer*100:.2f}% | "
                f"Val WER: {val_wer*100:.2f}% | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "val_accuracy": val_acc,
                "learning_rate": current_lr
            }, step=epoch)

            metrics_history["epochs"].append(epoch)
            metrics_history["train_loss"].append(train_loss)
            metrics_history["val_cer"].append(val_cer)
            metrics_history["val_wer"].append(val_wer)

            if val_cer < best_cer:
                best_cer = val_cer
                patience_counter = 0
                logger.info(f"New best CER: {best_cer*100:.2f}%. Saving model...")
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_cer": val_cer,
                    "val_wer": val_wer,
                    "val_acc": val_acc,
                    "charset": charset,
                }, str(best_model_path))

                mlflow.log_artifact(str(best_model_path), artifact_path="model_checkpoints")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        try:
            input_example = torch.randn(1, 1, params["data"]["image_height"], params["data"]["image_width"]).numpy()
            mlflow.pytorch.log_model(model, artifact_path="model", input_example=input_example)
        except Exception as e:
            logger.warning(f"MLflow log_model warning: {e}")
        
        dvc_metrics = {
            "best_val_cer": best_cer,
            "final_train_loss": metrics_history["train_loss"][-1] if metrics_history["train_loss"] else None,
            "total_epochs": len(metrics_history["epochs"]),
            "mlflow_run_id": run_id
        }
        with open("models/metrics.json", "w", encoding="utf-8") as f:
            json.dump(dvc_metrics, f, indent=2)

        metadata = {
            "model_name": params["model"]["name"],
            "backbone": params["model"]["backbone"],
            "version": "1.0.0",
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "Synthetic OCR",
            "best_cer": f"{best_cer * 100:.2f}%",
            "mlflow_run_id": run_id,
            "status": "production-ready"
        }
        with open("models/model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Training complete. Best CER: {best_cer*100:.2f}%. Run ID: {run_id}")


if __name__ == "__main__":
    main()
