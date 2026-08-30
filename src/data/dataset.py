import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import yaml
from loguru import logger


class OCRDataset(Dataset):
    def __init__(
        self,
        split_dir: str,
        charset: str,
        img_height: int = 32,
        img_width: int = 128,
        transform=None,
        max_label_length: int = 100,
    ):
        self.split_dir = Path(split_dir)
        self.charset = charset
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self.max_label_length = max_label_length

        self.char_to_idx = {char: i + 1 for i, char in enumerate(charset)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(charset)}
        self.blank_idx = 0

        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from {split_dir}")

    def _load_samples(self) -> List[Dict[str, Any]]:
        labels_path = self.split_dir / "labels.json"
        images_dir = self.split_dir / "images"

        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        samples = []
        for filename, text in labels.items():
            img_path = images_dir / filename
            if img_path.exists() and len(text) <= self.max_label_length:
                clean_text = "".join(c for c in text if c in self.char_to_idx)
                if clean_text:
                    samples.append({
                        "image_path": str(img_path),
                        "text": clean_text,
                    })

        return samples

    def _load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 128

        h, w = img.shape
        target_w = int(w * self.img_height / h)
        target_w = max(1, min(target_w, self.img_width))

        img = cv2.resize(img, (target_w, self.img_height), interpolation=cv2.INTER_LANCZOS4)

        if target_w < self.img_width:
            pad = np.ones((self.img_height, self.img_width - target_w), dtype=np.uint8) * 255
            img = np.concatenate([img, pad], axis=1)
        elif target_w > self.img_width:
            img = img[:, :self.img_width]

        return img

    def _encode_text(self, text: str) -> Tuple[torch.Tensor, int]:
        indices = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        return torch.tensor(indices, dtype=torch.long), len(indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        text = sample["text"]

        img = self._load_image(image_path)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5

        label, label_length = self._encode_text(text)

        return {
            "image": img_tensor,
            "label": label,
            "label_length": label_length,
            "text": text,
            "image_path": image_path,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    label_lengths = [item["label_length"] for item in batch]
    texts = [item["text"] for item in batch]

    images_batch = torch.stack(images, dim=0)
    labels_concat = torch.cat(labels)
    label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long)

    return {
        "images": images_batch,
        "labels": labels_concat,
        "label_lengths": label_lengths_tensor,
        "texts": texts,
    }


class OCRDataModule:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.data_params = self.params["data"]
        self.training_params = self.params["training"]

    def get_dataloader(
        self,
        split: str,
        transform=None,
        shuffle: bool = False,
    ) -> DataLoader:
        split_dir = os.path.join(self.data_params["splits_dir"], split)

        dataset = OCRDataset(
            split_dir=split_dir,
            charset=self.data_params["charset"],
            img_height=self.data_params["image_height"],
            img_width=self.data_params["image_width"],
            transform=transform,
            max_label_length=100,
        )

        return DataLoader(
            dataset,
            batch_size=self.training_params["batch_size"],
            shuffle=shuffle,
            num_workers=self.data_params.get("num_workers", 0),
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=split == "train",
        )

    def train_dataloader(self, transform=None) -> DataLoader:
        return self.get_dataloader("train", transform=transform, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", shuffle=False)
