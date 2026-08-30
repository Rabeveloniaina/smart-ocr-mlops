"""
Smart OCR - Streamlit Cloud Web Application
Déploiement 100% gratuit sur Streamlit Community Cloud (Zero carte bancaire)
"""

import time
import streamlit as st
import numpy as np
from PIL import Image

from src.predict import OCRPredictor

st.set_page_config(
    page_title="Smart OCR - MLOps",
    page_icon="📝",
    layout="centered"
)

# Chargement du modèle avec cache pour éviter de recharger à chaque requête
@st.cache_resource
def load_ocr_model():
    return OCRPredictor()

predictor = load_ocr_model()

st.title("📝 Smart OCR — Reconnaissance de Texte")
st.markdown("Reconnaissance de texte manuscrit et imprimé propulsée par **PyTorch / CRNN & EasyOCR**.")

# Zone d'upload
uploaded_file = st.file_uploader(
    "Choisissez une image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

col1, col2 = st.columns([1, 1])

use_beam_search = st.checkbox("Activer Beam Search (Décodage avancé)", value=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image importée", use_container_width=True)

    if st.button("⚡ Transcrire l'image", type="primary"):
        with st.spinner("Transcription en cours..."):
            t0 = time.time()
            img_np = np.array(image)
            result = predictor.predict_image(img_np, use_beam_search=use_beam_search)
            duration_ms = (time.time() - t0) * 1000

            st.success("Transcription terminée !")
            
            st.subheader("Texte reconnu :")
            st.text_area(
                label="",
                value=result.get("text", "(Aucun texte détecté)"),
                height=150
            )

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Confiance", f"{result.get('confidence', 0.0):.1f} %")
            with metric_col2:
                st.metric("Latence", f"{duration_ms:.1f} ms")

st.markdown("---")
st.caption("Projet Smart OCR MLOps • Architecture ResNet18 + BiLSTM + CTC")
