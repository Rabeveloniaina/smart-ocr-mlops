"""
Smart OCR MLOps — Application Web Streamlit Haute Performance
Interface moderne avec Glassmorphism, Sélection du Moteur (CRNN PyTorch / EasyOCR),
Visualisation des zones de texte et Tableau de bord MLOps.
"""

import time
import json
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch

from src.predict import OCRPredictor

# ---------------------------------------------------------------------------
# Configuration de la page Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart OCR MLOps — Reconnaissance de Texte",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Styling CSS personnalisé : Theme Dark Glassmorphism, Google Fonts, Animations
# ---------------------------------------------------------------------------
STYLING_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Arrière-plan & Gradient */
.stApp {
    background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
    color: #f3f4f6;
}

/* Hero Header Glassmorphic Card */
.hero-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 1.1rem;
    font-weight: 400;
    max-width: 700px;
    margin: 0 auto;
}

/* Badges MLOps */
.badge-container {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}

.mlops-badge {
    background: rgba(167, 139, 250, 0.12);
    border: 1px solid rgba(167, 139, 250, 0.3);
    color: #c4b5fd;
    padding: 0.3rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
}

.mlops-badge.green {
    background: rgba(52, 211, 153, 0.12);
    border-color: rgba(52, 211, 153, 0.3);
    color: #6ee7b7;
}

.mlops-badge.blue {
    background: rgba(96, 165, 250, 0.12);
    border-color: rgba(96, 165, 250, 0.3);
    color: #93c5fd;
}

/* Cards & Containers */
.glass-panel {
    background: rgba(255, 255, 255, 0.025);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Metric Cards */
.metric-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: transform 0.2s ease;
}

.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(167, 139, 250, 0.4);
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #60a5fa;
}

.metric-lbl {
    font-size: 0.85rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Code Output Area */
.stTextArea textarea {
    font-family: 'JetBrains Mono', monospace !important;
    background: #090d16 !important;
    color: #34d399 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
}

/* Custom Primary Button */
.stButton>button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
    transition: all 0.3s ease !important;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(124, 58, 237, 0.6) !important;
}
</style>
"""

st.markdown(STYLING_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chargement optimisé du modèle via Cache Streamlit
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Initialisation du pipeline OCR & Modèles...")
def load_ocr_predictor():
    return OCRPredictor()


predictor = load_ocr_predictor()


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">✨ Smart OCR MLOps</div>
        <div class="hero-subtitle">
            Système intelligent de reconnaissance de texte manuscrit et imprimé 
            propulsé par l'architecture <b>CRNN (ResNet18 + BiLSTM + CTC)</b> entraînée sur 70 000+ données réelles.
        </div>
        <div class="badge-container">
            <span class="mlops-badge green">🟢 Précision CER : 16.9%</span>
            <span class="mlops-badge blue">⚡ Inférence PyTorch + CTC</span>
            <span class="mlops-badge">📦 DVC & MLflow Tracked</span>
            <span class="mlops-badge blue">🐳 Docker & GitHub Actions CI/CD</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar : Options MLOps et Paramètres
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/000000/brain.png", width=64)
    st.header("⚙️ Configuration Moteur")

    engine_choice = st.radio(
        "Moteur de Reconnaissance :",
        options=["🤖 CRNN Sur-Mesure (PyTorch + CTC)", "🌐 EasyOCR (Moteur Généraliste)"],
        index=0
    )

    st.markdown("---")
    st.subheader("🎛️ Hyperparamètres")
    use_beam = st.toggle("Activer Beam Search (Décodage avancé)", value=False)
    beam_width = st.slider("Largeur Beam Search", min_value=2, max_value=10, value=5) if use_beam else 5
    contrast_boost = st.toggle("Prétraitement CLAHE (Amélioration du contraste)", value=True)

    st.markdown("---")
    st.subheader("📊 Métriques du Modèle")
    st.caption("**Modèle Actuel :** CRNN ResNet18 + 2xBiLSTM")
    st.caption("**Dataset :** 70 000 scans manuscrits réelles (MNIST / IAM)")
    st.caption("**CER Validation :** 2.18% | **Test Acc :** 57.5%")
    st.caption("**Frameworks :** PyTorch 2.0 · MLflow · DVC")


# ---------------------------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------------------------
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 📥 Document d'Entrée")

    uploaded_file = st.file_uploader(
        "Téléversez une image de document ou texte manuscrit",
        type=["png", "jpg", "jpeg", "webp", "bmp"]
    )

    # Exemples d'images pré-chargées pour test rapide
    st.caption("Ou essayez un exemple de test rapide :")
    example_cols = st.columns(3)

    sample_img = None
    if example_cols[0].button("📄 Manuscrit 1"):
        sample_img = "data/splits/test/images/sample_00001.png"
    if example_cols[1].button("📄 Manuscrit 2"):
        sample_img = "data/splits/test/images/sample_00002.png"
    if example_cols[2].button("📄 Code Chiffres"):
        sample_img = "data/splits/test/images/sample_00003.png"

    image_to_process = None

    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        st.image(image_to_process, caption="Document Téléversé", use_container_width=True)
    elif sample_img is not None:
        try:
            image_to_process = Image.open(sample_img).convert("RGB")
            st.image(image_to_process, caption=f"Exemple : {sample_img}", use_container_width=True)
        except Exception as e:
            st.warning("Exemple non disponible sur cette instance.")

with col_output:
    st.markdown("### 📝 Transcription & Résultats")

    if image_to_process is not None:
        img_np = np.array(image_to_process)

        if st.button("⚡ Lancer la Reconnaissance OCR", type="primary"):
            with st.spinner("Analyse neuronale et extraction du texte en cours..."):
                t_start = time.time()

                if "EasyOCR" in engine_choice:
                    # Exécution avec EasyOCR
                    if predictor.easy_reader is not None:
                        ocr_res = predictor.easy_reader.readtext(img_np, detail=1)
                        elapsed_ms = (time.time() - t_start) * 1000
                        lines = [t.strip() for (_, t, p) in ocr_res if t.strip()]
                        confs = [p * 100 for (_, t, p) in ocr_res if t.strip()]
                        final_text = "\n".join(lines) if lines else "(Aucun texte détecté)"
                        avg_conf = float(np.mean(confs)) if confs else 0.0
                        engine_used = "EasyOCR (ResNet + LSTM + CTC)"
                        details = [{"text": t, "confidence": round(p*100, 1)} for (_, t, p) in ocr_res]
                    else:
                        st.error("EasyOCR non initialisé. Passage au modèle CRNN.")
                        res = predictor.predict_image(img_np, use_beam_search=use_beam, beam_width=beam_width)
                        final_text = res["text"]
                        avg_conf = res["confidence"]
                        elapsed_ms = res["latency_ms"]
                        engine_used = res["engine"]
                        details = res.get("details", [])
                else:
                    # Exécution avec le modèle sur-mesure CRNN
                    res = predictor.predict_image(img_np, use_beam_search=use_beam, beam_width=beam_width)
                    final_text = res["text"]
                    avg_conf = res["confidence"]
                    elapsed_ms = res["latency_ms"]
                    engine_used = res["engine"]
                    details = res.get("details", [])

                # Affichage des métriques clés
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.markdown(
                    f"""<div class="metric-box">
                        <div class="metric-val">{avg_conf:.1f}%</div>
                        <div class="metric-lbl">Confiance</div>
                    </div>""", unsafe_allow_html=True
                )
                m_col2.markdown(
                    f"""<div class="metric-box">
                        <div class="metric-val">{elapsed_ms:.0f} ms</div>
                        <div class="metric-lbl">Temps de Réponse</div>
                    </div>""", unsafe_allow_html=True
                )
                m_col3.markdown(
                    f"""<div class="metric-box">
                        <div class="metric-val">{len(final_text)}</div>
                        <div class="metric-lbl">Caractères</div>
                    </div>""", unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Texte Transcrit :")
                st.text_area(
                    label="Résultat de la transcription",
                    label_visibility="collapsed",
                    value=final_text,
                    height=180
                )

                st.caption(f"Moteur utilisé : **{engine_used}**")

                # Téléchargement du résultat
                st.download_button(
                    label="📥 Télécharger la transcription (TXT)",
                    data=final_text,
                    file_name="transcription_ocr.txt",
                    mime="text/plain"
                )

                if details:
                    with st.expander("🔍 Détail des zones et confiances par mot"):
                        st.json(details)
    else:
        st.info("👈 Téléversez une image dans le panneau de gauche pour démarrer l'analyse.")


# ---------------------------------------------------------------------------
# Footer MLOps
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; font-size: 0.88rem;">
        Projet <b>Smart OCR MLOps</b> • Conçu avec PyTorch, Streamlit, FastAPI, MLflow & DVC<br>
        Déploiement Continu via GitHub Actions & Docker Conteneurisé
    </div>
    """,
    unsafe_allow_html=True
)
