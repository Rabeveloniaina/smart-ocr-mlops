# ============================================================
# Dockerfile - Smart OCR Manuscrit MLOps Production
# ============================================================

FROM python:3.10-slim AS base

# Empêche Python d'écrire des fichiers .pyc et bufferiser stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Dépendances système pour OpenCV et traitement d'image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source du projet
COPY params.yaml dvc.yaml ./
COPY src/ ./src/
COPY api/ ./api/
COPY monitoring/ ./monitoring/
COPY models/ ./models/

# Création d'un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Exposition du port FastAPI
EXPOSE 8000

# Healthcheck Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Démarrage du serveur Uvicorn en production
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
