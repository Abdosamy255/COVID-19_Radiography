"""Chest X-Ray Disease Classifier — Streamlit App."""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]
IMAGE_SIZE: tuple[int, int] = (224, 224)
MODEL_PATH: Path = Path(__file__).resolve().parent / "model.keras"

CLASS_META: dict[str, dict] = {
    "COVID": {
        "icon": "🦠",
        "color": "#E05C5C",
        "desc": "Findings consistent with COVID-19 pneumonia.",
    },
    "Lung Opacity": {
        "icon": "🌫️",
        "color": "#E09A3A",
        "desc": "Opacity detected — may indicate infection or fluid.",
    },
    "Normal": {
        "icon": "✅",
        "color": "#4CAF91",
        "desc": "No abnormalities detected in this X-ray.",
    },
    "Viral Pneumonia": {
        "icon": "🫁",
        "color": "#6B7FD7",
        "desc": "Findings consistent with viral pneumonia.",
    },
}

# ── CSS Injection ────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #0D0F14;
    color: #E8EAF0;
}

/* ── Main container ── */
.block-container {
    max-width: 780px;
    padding: 2.5rem 2rem 4rem;
}

/* ── Page title ── */
h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.5px !important;
    color: #F0F2F8 !important;
    margin-bottom: 0.25rem !important;
}

/* ── Subheadings ── */
h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #C5C9D8 !important;
    letter-spacing: 0.02em !important;
}

/* ── Caption / subtitle ── */
.stApp [data-testid="stCaptionContainer"] p {
    color: #6B7280;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #1E2130;
    margin: 1.5rem 0;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #13161F;
    border: 1.5px dashed #2A2E3F;
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4A90D9;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4A90D9 0%, #6B7FD7 100%);
    color: #fff;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2.2rem;
    width: 100%;
    transition: opacity 0.15s, transform 0.1s;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0);
}

/* ── Info / success / error boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 0.9rem;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #1E2130;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F1117 !important;
    border-right: 1px solid #1A1D28 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #9DA3B4 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #6B7280;
    font-size: 0.85rem;
}

/* ── Metric card (custom) ── */
.metric-card {
    background: #13161F;
    border: 1px solid #1E2230;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6B7280;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #E8EAF0;
}
.metric-sub {
    font-size: 0.82rem;
    color: #6B7280;
    margin-top: 0.15rem;
}

/* ── Probability bar container ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
}
.prob-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #9DA3B4;
    width: 130px;
    flex-shrink: 0;
}
.prob-bar-track {
    flex: 1;
    height: 6px;
    background: #1E2230;
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.prob-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #6B7280;
    width: 44px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Result banner ── */
.result-banner {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    border-left: 4px solid;
}
.result-banner .result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    margin-bottom: 0.2rem;
}
.result-banner .result-desc {
    font-size: 0.88rem;
    opacity: 0.75;
}

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #4A5568;
    margin: 1.4rem 0 0.7rem;
}

/* ── Footer ── */
.footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #1E2130;
    font-size: 0.78rem;
    color: #3D4455;
    text-align: center;
}
</style>
"""

# ── Model utilities ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=True)
def load_model(model_path: Path) -> tf.keras.Model:
    """Load and cache the Keras model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def process_image(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a DenseNet-preprocessed tensor."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    resized = img.resize(IMAGE_SIZE)
    arr = keras_image.img_to_array(resized)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def predict_image(
    model: tf.keras.Model, img: Image.Image
) -> tuple[str, np.ndarray]:
    """Run inference; return (predicted_label, probability_vector)."""
    tensor = process_image(img)
    raw = model.predict(tensor, verbose=0)

    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    probs = np.asarray(raw).squeeze()
    if probs.ndim > 1:
        probs = probs.reshape(-1)

    label = CLASS_NAMES[int(np.argmax(probs))]
    return label, probs

# ── UI helpers ───────────────────────────────────────────────────────────────

def render_result_banner(label: str, confidence: float) -> None:
    meta = CLASS_META[label]
    color = meta["color"]
    bg = color + "18"          # ~10 % alpha tint
    st.markdown(
        f"""
        <div class="result-banner" style="background:{bg}; border-color:{color};">
            <div class="result-title" style="color:{color};">
                {meta['icon']}&nbsp; {label}
            </div>
            <div class="result-desc">{meta['desc']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_card(confidence: float) -> None:
    pct = f"{confidence:.1%}"
    tier = (
        "High confidence" if confidence >= 0.80
        else "Moderate confidence" if confidence >= 0.55
        else "Low confidence"
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Model confidence</div>
            <div class="metric-value">{pct}</div>
            <div class="metric-sub">{tier}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(probs: np.ndarray) -> None:
    st.markdown('<div class="section-label">Class probabilities</div>', unsafe_allow_html=True)
    for name, prob in zip(CLASS_NAMES, probs):
        color = CLASS_META[name]["color"]
        pct = float(prob)
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label">{CLASS_META[name]['icon']} {name}</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill"
                         style="width:{pct*100:.2f}%; background:{color};"></div>
                </div>
                <div class="prob-pct">{pct:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Model")
        st.markdown(
            f"""
            **Backbone** — DenseNet  
            **Input** — {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} px  
            **Classes** — {len(CLASS_NAMES)}
            """
        )
        st.divider()
        st.markdown("### Classes")
        for name, meta in CLASS_META.items():
            st.markdown(f"{meta['icon']} **{name}**")
        st.divider()
        st.markdown("### Disclaimer")
        st.markdown(
            "_This tool is for research purposes only and does not constitute "
            "medical advice. Always consult a qualified clinician._"
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Chest X-Ray Classifier",
        page_icon="🩻",
        layout="centered",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()

    # ── Header ──
    st.markdown("# Chest X-Ray Classifier")
    st.caption("Upload a chest X-ray image to classify it with a DenseNet model.")
    st.divider()

    # ── Model loading ──
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"**Model failed to load.** {exc}")
        st.stop()

    # ── Upload ──
    uploaded_file = st.file_uploader(
        "Drop an image or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload a chest X-ray to begin.")
        st.markdown('<div class="footer">DenseNet · 224 × 224 · 4 classes</div>', unsafe_allow_html=True)
        return

    # ── Image display ──
    try:
        img = Image.open(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the image: {exc}")
        return

    st.image(img, caption="Uploaded X-ray", use_container_width=True)
    st.markdown("")

    # ── Predict ──
    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running inference…"):
            predicted_class, probabilities = predict_image(model, img)

        confidence = float(np.max(probabilities))
        st.divider()
        render_result_banner(predicted_class, confidence)
        render_confidence_card(confidence)
        render_probability_bars(probabilities)

    st.markdown('<div class="footer">For research use only — not a medical device.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()