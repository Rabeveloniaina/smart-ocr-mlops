import json
import random
import yaml
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
from tqdm import tqdm

SAMPLE_WORDS = [
    "Bonjour", "Monde", "Machine", "Learning", "MLOps", "Pipeline", "Smart", "OCR",
    "Manuscrit", "Intelligence", "Artificielle", "FastAPI", "Docker", "Model", "Data",
    "Evidently", "Python", "Deep", "Learning", "Neural", "Network", "Systeme", "Projet",
    "Reconnaissance", "Texte", "Production", "Deploiement", "Validation", "Accuracy",
    "IAM", "Dataset", "RIMES", "Entrainement", "Prediction", "Ingenieur", "Universite",
    "Paris", "Lyon", "Tunis", "Casablanca", "Dakar", "Montreal", "2024", "2025", "2026",
    "Test", "Validation", "Score", "Performance", "Graphique", "Serveur", "Conteneur"
]


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_handwritten_style_image(
    text: str,
    height: int = 32,
    width: int = 128
) -> np.ndarray:
    img = np.ones((height * 2, width * 2), dtype=np.uint8) * 255
    
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_ITALIC | cv2.FONT_HERSHEY_PLAIN
    ]
    font = random.choice(fonts)
    scale = random.uniform(0.7, 1.1)
    thickness = random.choice([1, 2])
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(5, int((width * 2 - text_w) / 2 + random.randint(-10, 10)))
    y = max(int(height * 1.3), int((height * 2 + text_h) / 2 + random.randint(-5, 5)))
    
    color = random.randint(0, 40)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    
    pts1 = np.float32([[0, 0], [width * 2, 0], [0, height * 2], [width * 2, height * 2]])
    dx = random.randint(-4, 4)
    dy = random.randint(-4, 4)
    pts2 = np.float32([[dx, dy], [width * 2 - dx, dy], [0, height * 2], [width * 2, height * 2]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (width * 2, height * 2), borderValue=255)
    
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def main(num_samples: int = 500):
    params = load_params("params.yaml")
    raw_dir = Path(params["data"]["raw_dir"])
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    h = params["data"]["image_height"]
    w = params["data"]["image_width"]
    
    labels = {}
    logger.info(f"Generating {num_samples} synthetic images...")
    
    for i in tqdm(range(num_samples)):
        if random.random() > 0.5:
            text = f"{random.choice(SAMPLE_WORDS)} {random.choice(SAMPLE_WORDS)}"
        else:
            text = random.choice(SAMPLE_WORDS)
            
        filename = f"sample_{i:05d}.png"
        img = generate_handwritten_style_image(text, height=h, width=w)
        
        cv2.imwrite(str(images_dir / filename), img)
        labels[filename] = text
        
    labels_file = raw_dir / "labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Generated {len(labels)} samples in {raw_dir}")


if __name__ == "__main__":
    main()
