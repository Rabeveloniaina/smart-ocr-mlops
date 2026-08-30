"""
Smart OCR - Hugging Face Space & Gradio Web Application
Déploiement Cloud Gratuit et Démonstration Interactive
"""

import os
import io
import time
import gradio as gr
import numpy as np
import cv2
from PIL import Image
from loguru import logger

from src.predict import OCRPredictor

# Initialisation du prédicteur OCR
predictor = OCRPredictor()

def perform_ocr(image, use_beam_search=False):
    """
    Fonction d'inférence appelée par l'interface Gradio
    """
    if image is None:
        return "Veuillez téléverser une image.", 0.0, "0 ms"

    try:
        t0 = time.time()
        
        # Conversion PIL vers format compatible
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            img_np = np.array(Image.open(image))

        # Prédiction
        result = predictor.predict_image(img_np, use_beam_search=use_beam_search)
        duration_ms = (time.time() - t0) * 1000

        text = result.get("text", "")
        if not text.strip():
            text = "(Aucun texte détecté avec un seuil de confiance suffisant)"

        confidence = result.get("confidence", 0.0)
        latency = f"{duration_ms:.1f} ms"

        return text, confidence, latency

    except Exception as e:
        logger.error(f"Erreur OCR Gradio: {e}")
        return f"Erreur lors de la transcription : {str(e)}", 0.0, "-- ms"


# Interface graphique Gradio moderne
with gr.Blocks(
    title="Smart OCR - MLOps Cloud Demo",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
    ),
    css="""
    .container { max-width: 960px; margin: auto; }
    .header-title { text-align: center; margin-bottom: 0.5rem; }
    """
) as demo:
    gr.Markdown(
        """
        # 📝 Smart OCR — Reconnaissance de Texte Manuscrit
        ### Démonstration Cloud MLOps (PyTorch + CRNN / ResNet18 + CTC & EasyOCR)
        Téléversez une image de document manuscrit ou imprimé pour obtenir une transcription instantanée.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Image du document / texte manuscrit",
                sources=["upload", "clipboard"]
            )
            beam_search_checkbox = gr.Checkbox(
                label="Activer Beam Search (Décodage avancé)",
                value=False
            )
            submit_btn = gr.Button("⚡ Transcrire le texte", variant="primary", size="lg")

        with gr.Column(scale=1):
            text_output = gr.Textbox(
                label="Texte transcrit",
                placeholder="Le texte transcrit s'affichera ici...",
                lines=8,
                show_copy_button=True
            )
            with gr.Row():
                conf_output = gr.Number(label="Indice de confiance (%)", precision=1)
                latency_output = gr.Textbox(label="Temps de réponse")

    submit_btn.click(
        fn=perform_ocr,
        inputs=[image_input, beam_search_checkbox],
        outputs=[text_output, conf_output, latency_output]
    )

    gr.Markdown(
        """
        ---
        **Architecture MLOps** :
        - Modèle : ResNet18 Backbone + BiLSTM 2 couches + Décodeur CTC Greedy / Beam Search
        - Tracking & CI/CD : MLflow, DVC, GitHub Actions, Docker
        - API REST : FastAPI exposée sur `/predict` et `/health`
        """
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
