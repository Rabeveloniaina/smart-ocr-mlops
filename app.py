"""
Smart OCR - Gradio Web Application & Hugging Face Space
Interface moderne avec thème Soft Indigo, métriques dynamiques et décodage CTC / Beam Search
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

def perform_ocr(image, engine_type="CRNN PyTorch (Sur-mesure)", use_beam_search=False):
    """
    Fonction d'inférence appelée par l'interface Gradio
    """
    if image is None:
        return "Veuillez téléverser une image.", "0.0 %", "0 ms", "Aucune donnée"

    try:
        t0 = time.time()

        if isinstance(image, Image.Image):
            img_np = np.array(image)
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            img_np = np.array(Image.open(image))

        if "EasyOCR" in engine_type and predictor.easy_reader is not None:
            ocr_res = predictor.easy_reader.readtext(img_np, detail=1)
            duration_ms = (time.time() - t0) * 1000
            lines = [t.strip() for (_, t, p) in ocr_res if t.strip()]
            confs = [p * 100 for (_, t, p) in ocr_res if t.strip()]
            full_text = "\n".join(lines) if lines else "(Aucun texte détecté)"
            avg_conf = float(np.mean(confs)) if confs else 0.0
            engine_used = "EasyOCR (ResNet + LSTM + CTC)"
        else:
            result = predictor.predict_image(img_np, use_beam_search=use_beam_search)
            duration_ms = (time.time() - t0) * 1000
            full_text = result.get("text", "")
            if not full_text.strip():
                full_text = "(Aucun texte détecté avec un seuil de confiance suffisant)"
            avg_conf = result.get("confidence", 0.0)
            engine_used = result.get("engine", "CRNN (PyTorch + CTC)")

        confidence_str = f"{avg_conf:.1f} %"
        latency_str = f"{duration_ms:.0f} ms"

        return full_text, confidence_str, latency_str, engine_used

    except Exception as e:
        logger.error(f"Erreur OCR Gradio: {e}")
        return f"Erreur lors de la transcription : {str(e)}", "0.0 %", "-- ms", "Erreur"


# Interface graphique Gradio moderne
with gr.Blocks(
    title="Smart OCR - MLOps Cloud Demo",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
    ),
    css="""
    .container { max-width: 1000px; margin: auto; }
    .header-card { text-align: center; background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem; }
    """
) as demo:
    gr.Markdown(
        """
        # ✨ Smart OCR — Reconnaissance de Texte Manuscrit & Imprimé
        ### Démonstration Cloud MLOps (PyTorch + CRNN / ResNet18 + CTC & EasyOCR)
        Téléversez une image de document manuscrit ou imprimé pour obtenir une transcription instantanée.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📥 Image du document / texte manuscrit",
                sources=["upload", "clipboard"]
            )
            engine_dropdown = gr.Dropdown(
                choices=["CRNN PyTorch (Sur-mesure)", "EasyOCR (Généraliste)"],
                value="CRNN PyTorch (Sur-mesure)",
                label="⚙️ Moteur de Reconnaissance"
            )
            beam_search_checkbox = gr.Checkbox(
                label="Activer Beam Search (Décodage avancé)",
                value=False
            )
            submit_btn = gr.Button("⚡ Transcrire le texte", variant="primary", size="lg")

        with gr.Column(scale=1):
            text_output = gr.Textbox(
                label="📝 Texte transcrit",
                placeholder="Le texte transcrit s'affichera ici...",
                lines=8,
                show_copy_button=True
            )
            with gr.Row():
                conf_output = gr.Textbox(label="Indice de confiance")
                latency_output = gr.Textbox(label="Temps de réponse")
                engine_output = gr.Textbox(label="Moteur utilisé")

    submit_btn.click(
        fn=perform_ocr,
        inputs=[image_input, engine_dropdown, beam_search_checkbox],
        outputs=[text_output, conf_output, latency_output, engine_output]
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
