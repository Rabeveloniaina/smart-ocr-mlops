import json
import shutil
import random
import yaml
from pathlib import Path
from loguru import logger
from typing import Dict, Any


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    processed_dir = Path(params["data"]["processed_dir"])
    splits_dir = Path(params["data"]["splits_dir"])
    
    train_ratio = params["data"]["train_ratio"]
    val_ratio = params["data"]["val_ratio"]
    test_ratio = params["data"]["test_ratio"]
    seed = params["training"]["seed"]
    
    random.seed(seed)
    
    proc_images = processed_dir / "images"
    proc_labels_file = processed_dir / "labels.json"
    
    if not proc_labels_file.exists():
        raise FileNotFoundError(f"Missing file: {proc_labels_file}")
        
    with open(proc_labels_file, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)
        
    items = list(labels_dict.items())
    random.shuffle(items)
    
    total = len(items)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    
    splits = {
        "train": dict(train_items),
        "val": dict(val_items),
        "test": dict(test_items)
    }
    
    for split_name, split_data in splits.items():
        s_dir = splits_dir / split_name
        s_img_dir = s_dir / "images"
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_img_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in split_data.keys():
            src = proc_images / filename
            dst = s_img_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                
        with open(s_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Split [{split_name}]: {len(split_data)} samples.")
        
    logger.info("Dataset split complete.")


if __name__ == "__main__":
    main()
