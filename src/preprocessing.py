import cv2
import json
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ImagePreprocessor:
    def __init__(
        self,
        target_height: int = 32,
        target_width: int = 128,
        denoise: bool = True
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.denoise = denoise

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image.copy()

    def segment_lines(self, gray: np.ndarray) -> List[np.ndarray]:
        h, w = gray.shape
        if h <= self.target_height * 1.5:
            return [gray]

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        proj = np.sum(thresh, axis=1)
        max_proj = np.max(proj) if len(proj) > 0 else 0
        if max_proj == 0:
            return [gray]

        threshold_val = max_proj * 0.02
        active = proj > threshold_val

        line_ranges = []
        in_line = False
        start_y = 0
        for y, is_act in enumerate(active):
            if is_act and not in_line:
                in_line = True
                start_y = max(0, y - 3)
            elif not is_act and in_line:
                in_line = False
                end_y = min(h, y + 3)
                if end_y - start_y >= 12:
                    line_ranges.append((start_y, end_y))
        
        if in_line and (h - start_y >= 12):
            line_ranges.append((start_y, h))

        if not line_ranges:
            return [gray]

        line_crops = []
        for sy, ey in line_ranges:
            crop = gray[sy:ey, :]
            line_crops.append(crop)
        return line_crops

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = self.to_grayscale(image)

        if self.denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)

        h, w = gray.shape
        if h == 0 or w == 0:
            return np.ones((self.target_height, self.target_width), dtype=np.uint8) * 255

        aspect_ratio = w / h
        new_w = int(self.target_height * aspect_ratio)

        if new_w > self.target_width:
            resized = cv2.resize(gray, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            final_image = resized
        else:
            resized = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_AREA)
            final_image = np.ones((self.target_height, self.target_width), dtype=np.uint8) * 255
            final_image[:, :new_w] = resized

        return final_image

    def preprocess_from_path(self, image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return self.preprocess_image(img)

    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image bytes")
        return self.preprocess_image(img)


def run_preprocessing_pipeline(params_path: str = "params.yaml"):
    params = load_params(params_path)
    raw_dir = Path(params["data"]["raw_dir"])
    processed_dir = Path(params["data"]["processed_dir"])
    
    target_h = params["data"]["image_height"]
    target_w = params["data"]["image_width"]

    raw_images_dir = raw_dir / "images"
    raw_labels_path = raw_dir / "labels.json"
    
    out_images_dir = processed_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_path = processed_dir / "labels.json"

    if not raw_labels_path.exists():
        logger.warning(f"Missing labels file: {raw_labels_path}")
        return

    with open(raw_labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)

    preprocessor = ImagePreprocessor(target_height=target_h, target_width=target_w)
    processed_labels = {}

    logger.info(f"Preprocessing {len(labels_dict)} images...")
    for filename, text in tqdm(labels_dict.items()):
        src_path = raw_images_dir / filename
        if not src_path.exists():
            continue

        try:
            processed_img = preprocessor.preprocess_from_path(str(src_path))
            dest_path = out_images_dir / filename
            cv2.imwrite(str(dest_path), processed_img)
            processed_labels[filename] = text
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    with open(out_labels_path, "w", encoding="utf-8") as f:
        json.dump(processed_labels, f, ensure_ascii=False, indent=2)

    logger.info(f"Preprocessing complete: {len(processed_labels)} images saved to {processed_dir}")


if __name__ == "__main__":
    run_preprocessing_pipeline()
