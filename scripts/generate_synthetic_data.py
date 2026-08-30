import json
import random
import yaml
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
from tqdm import tqdm

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL non disponible, utilisation du fallback OpenCV pur.")

MOTS_FR = [
    "Bonjour", "Monde", "Machine", "Pipeline", "Reconnaissance", "Texte",
    "Manuscrit", "Intelligence", "Artificielle", "Donnees", "Modele",
    "Apprentissage", "Reseau", "Neurones", "Entrainement", "Validation",
    "Production", "Deploiement", "Serveur", "Conteneur", "Automatique",
    "Precision", "Performance", "Evaluation", "Traitement", "Image",
    "Caractere", "Lettre", "Phrase", "Document", "Formulaire", "Facture",
    "Contrat", "Rapport", "Analyse", "Resultat", "Tableau", "Graphique",
    "Calcul", "Statistique", "Methode", "Algorithme", "Fonction", "Variable",
    "Parametre", "Couche", "Convolution", "Attention", "Gradient", "Perte",
    "Paris", "Lyon", "Dakar", "Tunis", "Montreal", "Abidjan", "Bruxelles",
    "Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin", "Juillet",
    "Aout", "Septembre", "Octobre", "Novembre", "Decembre",
    "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche",
    "Ingenieur", "Professeur", "Etudiant", "Directeur", "Manager",
    "Total", "Montant", "Prix", "Quantite", "Reference", "Numero",
    "Adresse", "Telephone", "Email", "Date", "Heure", "Minute",
]

MOTS_EN = [
    "Smart", "OCR", "Deep", "Learning", "Neural", "Network", "Dataset",
    "Training", "Testing", "Accuracy", "Model", "Python", "Docker",
    "FastAPI", "MLflow", "Streamlit", "GitHub", "Action", "Workflow",
    "ResNet", "LSTM", "Attention", "CTC", "Decoder", "Encoder",
    "Batch", "Epoch", "Loss", "Optimizer", "Scheduler", "Gradient",
    "Image", "Label", "Prediction", "Inference", "Confidence", "Score",
    "Token", "Class", "Layer", "Feature", "Weight", "Bias",
    "Pipeline", "Stage", "Data", "Split", "Augment", "Transform",
    "London", "Paris", "Berlin", "Tokyo", "Sydney",
    "Hello", "World", "Text", "Word", "Sentence", "Character",
    "Invoice", "Contract", "Report", "Form", "Document",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
]

NOMBRES = [str(i) for i in range(0, 10)] + [
    "100", "200", "500", "1000", "2024", "2025", "2026",
    "42", "99", "123", "456", "789",
]

PHRASES_COURTES = [
    "Bonjour Monde", "Smart OCR", "Deep Learning", "MLOps Pipeline",
    "Texte Manuscrit", "Neural Network", "Data Science", "Machine Learning",
    "Reseau Neuronal", "FastAPI Docker", "Python 3.10", "GitHub Actions",
    "Mon Projet", "Mon Nom", "Date 2026", "Score 98",
    "Total 120", "Ref 00123", "Page 1", "Tel 06 12",
    "Accuracy Validation", "RIMES Dataset", "IAM Database",
    "Graphique Score", "Pipeline RIMES", "Docker Texte",
    "Network 2026", "Intelligence A", "Prediction OK",
]

ALL_TEXTS = MOTS_FR + MOTS_EN + NOMBRES + PHRASES_COURTES * 3


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_system_fonts() -> list:
    candidates = []
    win_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/courier.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/trebuc.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/framd.ttf",
    ]
    linux_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]
    for p in win_paths + linux_paths:
        if Path(p).exists():
            candidates.append(p)
    return candidates if candidates else None


def generate_pil_handwritten_image(
    text: str,
    height: int = 32,
    width: int = 256,
    font_paths: list = None,
) -> np.ndarray:
    scale = 4
    H, W = height * scale, width * scale

    bg_color = random.randint(230, 255)
    img_pil = Image.new("L", (W, H), color=bg_color)
    draw = ImageDraw.Draw(img_pil)

    font = None
    if font_paths:
        fp = random.choice(font_paths)
        font_size = random.randint(int(H * 0.45), int(H * 0.70))
        try:
            font = ImageFont.truetype(fp, size=font_size)
        except Exception:
            font = None

    if font is None:
        font = ImageFont.load_default()

    ink_color = random.randint(0, 50)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)

    x = max(10, (W - tw) // 2 + random.randint(-20, 20))
    y = max(5, (H - th) // 2 + random.randint(-10, 10))

    draw.text((x, y), text, fill=ink_color, font=font)

    img_np = np.array(img_pil, dtype=np.uint8)

    angle = random.uniform(-6, 6)
    M_rot = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
    img_np = cv2.warpAffine(img_np, M_rot, (W, H), borderValue=bg_color)

    if random.random() > 0.5:
        pts1 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        dx = random.randint(0, int(W * 0.03))
        dy = random.randint(0, int(H * 0.04))
        pts2 = np.float32([
            [dx, dy], [W - dx, random.randint(0, dy)],
            [0, H], [W, H]
        ])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = cv2.warpPerspective(img_np, M_persp, (W, H), borderValue=bg_color)

    noise_std = random.uniform(2, 10)
    noise = np.random.normal(0, noise_std, img_np.shape).astype(np.int16)
    img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    op = random.random()
    kernel = np.ones((2, 2), np.uint8)
    if op < 0.25:
        img_np = cv2.erode(img_np, kernel, iterations=1)
    elif op < 0.5:
        img_np = cv2.dilate(img_np, kernel, iterations=1)

    if random.random() > 0.6:
        dx_map = cv2.GaussianBlur(
            (np.random.rand(H, W) * 2 - 1).astype(np.float32), (15, 15), 0
        ) * 8
        dy_map = cv2.GaussianBlur(
            (np.random.rand(H, W) * 2 - 1).astype(np.float32), (15, 15), 0
        ) * 8
        map_x, map_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (map_x + dx_map).astype(np.float32)
        map_y = (map_y + dy_map).astype(np.float32)
        img_np = cv2.remap(img_np, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR, borderValue=bg_color)

    return cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)


def generate_opencv_fallback_image(
    text: str,
    height: int = 32,
    width: int = 256,
) -> np.ndarray:
    H, W = height * 4, width * 4
    bg = random.randint(230, 255)
    img = np.ones((H, W), dtype=np.uint8) * bg

    fonts_cv = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_ITALIC | cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
    ]
    font = random.choice(fonts_cv)
    scale = random.uniform(1.0, 1.6)
    thickness = random.choice([1, 2])

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (W - tw) // 2 + random.randint(-20, 20))
    y = max(int(H * 0.65), (H + th) // 2 + random.randint(-10, 10))

    color = random.randint(0, 50)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    angle = random.uniform(-6, 6)
    M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (W, H), borderValue=bg)

    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def main(num_samples: int = 2000):
    params = load_params("params.yaml")
    raw_dir = Path(params["data"]["raw_dir"])
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    h = params["data"]["image_height"]
    w = params["data"]["image_width"]

    font_paths = _get_system_fonts() if PIL_AVAILABLE else None
    if font_paths:
        logger.info(f"Polices disponibles : {len(font_paths)} trouvees.")
    else:
        logger.warning("Aucune police TTF trouvee, utilisation du fallback OpenCV.")

    labels = {}
    logger.info(f"Generation de {num_samples} images manuscrites (h={h}, w={w})...")

    for i in tqdm(range(num_samples)):
        rng = random.random()
        if rng < 0.40:
            text = random.choice(MOTS_FR + MOTS_EN)
        elif rng < 0.80:
            text = random.choice(PHRASES_COURTES)
            if random.random() > 0.5:
                text = f"{random.choice(MOTS_FR)} {random.choice(MOTS_EN)}"
        else:
            text = random.choice(NOMBRES)

        charset = params["data"]["charset"]
        text = "".join(c for c in text if c in charset or c == " ")
        text = text.strip()
        if not text:
            text = random.choice(MOTS_EN)

        if PIL_AVAILABLE and font_paths and random.random() > 0.25:
            img = generate_pil_handwritten_image(text, height=h, width=w, font_paths=font_paths)
        else:
            img = generate_opencv_fallback_image(text, height=h, width=w)

        filename = f"sample_{i:05d}.png"
        cv2.imwrite(str(images_dir / filename), img)
        labels[filename] = text

    labels_file = raw_dir / "labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    logger.info(f"Generation terminee : {len(labels)} images dans '{raw_dir}'")
    logger.info(f"Exemples : {list(labels.values())[:5]}")


if __name__ == "__main__":
    main()
