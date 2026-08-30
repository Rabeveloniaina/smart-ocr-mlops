# Smart OCR MLOps Pipeline

Pipeline de reconnaissance de texte manuscrit et déploiement MLOps.

---

## Architecture

Le projet implémente un cycle de vie MLOps complet :

```
[DVC Data] -> [Preprocessing & Training (CRNN)] -> [MLflow Tracking] -> [FastAPI Server] -> [Docker]
     ^                                                                        |
     |                                                                        v
     +------------------- [Retraining (Evidently AI Drift)] <-----------------+
```

### Composants techniques

- **Modèle** : CRNN (ResNet18 backbone + BiLSTM + CTC Loss)
- **Data Versioning** : DVC (`dvc.yaml`, `params.yaml`)
- **Tracking des expériences** : MLflow
- **Serving** : FastAPI REST API (`/predict`, `/health`, `/model-info`)
- **Monitoring** : Evidently AI (Data Drift, Confidence Drift, Latency)
- **Conteneurisation** : Docker & Docker Compose
- **CI/CD** : GitHub Actions

---

## Structure du projet

```text
smart-ocr-mlops/
├── .github/workflows/ci.yml
├── api/
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── templates/index.html
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   ├── best_model.pt
│   ├── model_metadata.json
│   └── evaluation_metrics.json
├── monitoring/
│   ├── alerts.py
│   ├── evidently_report.py
│   └── metrics_store.py
├── scripts/
│   ├── generate_synthetic_data.py
│   └── split_dataset.py
├── src/
│   ├── data/
│   │   ├── augmentation.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── crnn.py
│   │   └── ctc_decoder.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── retraining.py
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

---

## Installation et exécution

### 1. Environnement virtuel

```bash
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate
# Sur Linux/macOS :
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Exécution du pipeline complet

```bash
python demo.py
```

Ou étape par étape avec DVC :

```bash
dvc repro
```

### 3. Lancement de l'API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Interface Web : [http://localhost:8000/](http://localhost:8000/)
- Documentation Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Tests unitaires

```bash
pytest
```

### 5. Déploiement Docker

```bash
docker-compose up --build -d
```
