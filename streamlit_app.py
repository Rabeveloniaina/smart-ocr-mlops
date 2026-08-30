"""
Smart OCR - Streamlit Cloud (Version légère pour Free Tier)
Utilise EasyOCR directement sans charger le modèle CRNN local.
"""

import time
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Smart OCR - MLOps",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Smart OCR — Reconnaissance de Texte")
st.markdown(
    "Reconnaissance de texte manuscrit et imprimé propulsée par **EasyOCR** "
    "(ResNet + LSTM + CTC Attention), déployée via le pipeline MLOps."
)


@st.cache_resource(show_spinner="Chargement du moteur OCR (EasyOCR)...")
def load_easyocr_reader():
    """Charge EasyOCR une seule fois grâce au cache Streamlit."""
    import easyocr
    return easyocr.Reader(["fr", "en"], gpu=False, verbose=False)


reader = load_easyocr_reader()

st.success("✅ Moteur OCR prêt.", icon="🔍")

uploaded_file = st.file_uploader(
    "Choisissez une image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

use_detail = st.checkbox(
    "Afficher les détails (boîtes de détection, scores par zone)", value=False
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image importée", width=600)

    if st.button("⚡ Transcrire l'image", type="primary"):
        with st.spinner("Analyse de l'image en cours..."):
            t0 = time.time()
            img_np = np.array(image)

            try:
                results = reader.readtext(img_np, detail=1)
                duration_ms = (time.time() - t0) * 1000

                if results:
                    lines = [text for (_, text, prob) in results if text.strip()]
                    confs = [prob * 100 for (_, text, prob) in results if text.strip()]
                    full_text = "\n".join(lines)
                    avg_conf = float(np.mean(confs)) if confs else 0.0
                else:
                    full_text = "(Aucun texte détecté dans l'image)"
                    avg_conf = 0.0

                st.success("Transcription terminée !")

                st.subheader("Texte reconnu :")
                st.text_area(
                    label="Résultat de la transcription",
                    label_visibility="collapsed",
                    value=full_text,
                    height=180
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confiance moyenne", f"{avg_conf:.1f} %")
                with col2:
                    st.metric("Temps de réponse", f"{duration_ms:.0f} ms")

                if use_detail and results:
                    st.subheader("Détail par zone détectée")
                    for i, (bbox, text, prob) in enumerate(results, 1):
                        if text.strip():
                            st.write(f"**Zone {i}** : `{text}` — confiance : `{prob*100:.1f}%`")

            except Exception as e:
                st.error(f"Erreur lors de la transcription : {str(e)}")

st.markdown("---")
st.caption(
    "Projet **Smart OCR MLOps** • Architecture : EasyOCR (ResNet+LSTM+CTC) • "
    "Pipeline : DVC · MLflow · GitHub Actions · Docker"
)
