"""Gradio arayüzü. Proje kökünden: python app_gradio.py"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from spam_detector.pipeline import SmsClassifierPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def _load_default() -> SmsClassifierPipeline:
    for name in ("pipeline_artifacts.joblib", "pipeline_logreg.joblib", "pipeline_rf.joblib"):
        p = MODELS_DIR / name
        if p.is_file():
            return SmsClassifierPipeline.load(MODELS_DIR, name)
    raise FileNotFoundError("models/ içinde .joblib yok. Önce python train.py çalıştırın.")


def analyze(sms_text: str, model_choice: str) -> tuple[str, str]:
    if model_choice.startswith("Logistic"):
        fname = "pipeline_logreg.joblib"
    elif model_choice.startswith("Random"):
        fname = "pipeline_rf.joblib"
    else:
        fname = "pipeline_artifacts.joblib"
    path = MODELS_DIR / fname
    if not path.is_file():
        pipe = _load_default()
    else:
        pipe = SmsClassifierPipeline.load(MODELS_DIR, fname)
    label, spam_p = pipe.predict(sms_text or "")
    if label.startswith("Belirsiz"):
        return label, "—"
    return label, f"{spam_p * 100:.1f}%"


def main() -> None:
    has_lr = (MODELS_DIR / "pipeline_logreg.joblib").is_file()
    has_rf = (MODELS_DIR / "pipeline_rf.joblib").is_file()
    if has_lr and has_rf:
        choices = ["Varsayılan (pipeline_artifacts)", "Logistic Regression", "Random Forest"]
    else:
        choices = ["Varsayılan (pipeline_artifacts)"]

    gr.Interface(
        fn=analyze,
        inputs=[
            gr.Textbox(label="SMS metni", lines=6, placeholder="Metni buraya yazın..."),
            gr.Dropdown(choices=choices, value=choices[0], label="Model"),
        ],
        outputs=[
            gr.Textbox(label="Tahmin (Ham / Spam)"),
            gr.Textbox(label="Spam olasılığı"),
        ],
        title="SMS Spam Tespiti",
        description="Önce `python train.py --demo` veya UCI CSV ile eğitin; modeller `models/` altında.",
    ).launch()


if __name__ == "__main__":
    main()
