"""
TumorSight — Streamlit Dashboard
Two-step prediction:
1) Binary CNN: Tumor / No Tumor
2) Stage model: Stage I / II / III / IV
"""

import numpy as np
from PIL import Image
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TumorSight",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #00BCD4;
        border-bottom: 2px solid #00BCD4;
        padding-bottom: 0.4rem;
    }
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #444;
        text-align: center;
    }
    .prediction-badge {
        font-size: 1.6rem;
        font-weight: 700;
        padding: 0.4rem 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

STAGE_COLORS = {
    "Healthy": "#4CAF50",
    "Stage I": "#8BC34A",
    "Stage II": "#FFC107",
    "Stage III": "#FF5722",
    "Stage IV": "#B71C1C",
}

CLASS_NAMES = ["Healthy", "Stage I", "Stage II", "Stage III", "Stage IV"]
IMG_SIZE = (128, 128)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


def preprocess(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def run_gradcam(model, image: np.ndarray, class_index: int) -> np.ndarray:
    import sys
    import os
    sys.path.insert(0, os.path.abspath("."))
    from src.evaluation.grad_cam import generate_gradcam, overlay_heatmap

    heatmap = generate_gradcam(model, image, class_index)
    overlaid = overlay_heatmap(image, heatmap, alpha=0.45)
    return overlaid


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "assets/logo.png",
        caption="TumorSight Dashboard",
        width=250
    )

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    binary_model_path = st.text_input(
        "Binary model path",
        value=str(BASE_DIR / "models" / "saved_binary" / "best_model.keras")
    )

    stage_model_path = st.text_input(
        "Stage model path",
        value=str(BASE_DIR / "models" / "saved_stages" / "best_model.keras")
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        0.5, 0.99, 0.70, 0.01
    )

    tumor_threshold = st.slider(
        "Tumor detection threshold",
        0.3, 0.9, 0.50, 0.01
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**TumorSight** uses a two-step AI pipeline: first, a binary CNN detects "
        "whether a tumor is present. If a tumor is detected, a second model classifies "
        "the tumor stage."
    )

    st.markdown("---")
    st.caption(
        "⚕️ Clinical decision support only. Always verify with a qualified physician."
    )

# ─── Main content ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">🧠 TumorSight — Clinical Decision Support</div>',
    unsafe_allow_html=True
)

binary_model = load_model(binary_model_path)
stage_model = load_model(stage_model_path)

if binary_model is None:
    st.warning(f"⚠️ Binary model not found at `{binary_model_path}`.")

if stage_model is None:
    st.warning(f"⚠️ Stage model not found at `{stage_model_path}`.")

tabs = st.tabs([
    "🔬 Single Image Analysis",
    "📊 Batch Evaluation",
    "📖 Clinical Reference"
])

# ─── Tab 1: Single image ─────────────────────────────────────────────────────
with tabs[0]:
    col_upload, col_results = st.columns([1, 2])

    with col_upload:
        st.subheader("Upload Medical Image")

        uploaded = st.file_uploader(
            "Supported: PNG, JPG, JPEG",
            type=["png", "jpg", "jpeg"],
            help="Upload a brain MRI, CT scan, or histopathology image."
        )

        if uploaded:
            st.image(
                uploaded,
                caption="Uploaded image",
                use_container_width=True
            )

    with col_results:
        if uploaded and binary_model is not None and stage_model is not None:
            with st.spinner("Running two-step inference..."):
                image = preprocess(uploaded)

                # Step 1: Binary CNN
                binary_prob = binary_model.predict(
                    np.expand_dims(image, 0),
                    verbose=0
                )[0][0]

                # No tumor
                if binary_prob < tumor_threshold:
                    prediction = "Healthy"
                    confidence = float(1 - binary_prob)
                    pred_idx = 0

                    probs = np.zeros(len(CLASS_NAMES))
                    probs[0] = confidence

                    stage_used = False

                # Tumor detected → stage model
                else:
                    stage_probs = stage_model.predict(
                        np.expand_dims(image, 0),
                        verbose=0
                    )[0]

                    pred_idx = int(np.argmax(stage_probs))
                    confidence = float(stage_probs[pred_idx])
                    prediction = CLASS_NAMES[pred_idx]

                    probs = stage_probs
                    stage_used = True

            # ── Prediction banner ──────────────────────────────────────────
            color = STAGE_COLORS.get(prediction, "#607D8B")

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="prediction-badge" style="color:{color};">
                        {prediction}
                    </div>
                    <div style="color:#aaa; margin-top:.4rem;">
                        Confidence:
                        <b style="color:{color}">{confidence:.1%}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

            # ── Binary result ──────────────────────────────────────────────
            st.subheader("Tumor Detection Result")

            c1, c2 = st.columns(2)
            c1.metric("Tumor probability", f"{binary_prob:.1%}")
            c2.metric("Decision threshold", f"{tumor_threshold:.1%}")

            if binary_prob < tumor_threshold:
                st.success("✅ No tumor detected")
            else:
                st.warning("⚠️ Tumor detected — stage classification activated")

            # ── Confidence alert ───────────────────────────────────────────
            if confidence < conf_threshold:
                st.warning(
                    f"⚠️ Manual review required — Confidence {confidence:.1%} "
                    f"< {conf_threshold:.0%}"
                )
            else:
                st.success("✅ High confidence prediction")

            # ── Probability bars ───────────────────────────────────────────
            st.subheader("Class Probabilities")

            import plotly.graph_objects as go

            fig = go.Figure(go.Bar(
                x=[f"{p:.1%}" for p in probs],
                y=CLASS_NAMES,
                orientation="h",
                marker_color=[STAGE_COLORS[c] for c in CLASS_NAMES],
                text=[f"{p:.1%}" for p in probs],
                textposition="outside",
            ))

            fig.update_layout(
                height=280,
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis_title="Probability",
                plot_bgcolor="#0f0f1a",
                paper_bgcolor="#0f0f1a",
                font_color="#eee",
            )

            st.plotly_chart(fig, use_container_width=True)

        elif uploaded:
            st.error("Please load valid binary and stage models first.")

# ─── Tab 2: Batch evaluation ─────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Batch Evaluation")
    st.info("Upload multiple images to evaluate predictions.")

    uploaded_batch = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_batch and binary_model is not None and stage_model is not None:
        results = []
        prog = st.progress(0)

        for i, f in enumerate(uploaded_batch):
            img = preprocess(f)

            binary_prob = binary_model.predict(
                np.expand_dims(img, 0),
                verbose=0
            )[0][0]

            if binary_prob < tumor_threshold:
                prediction = "Healthy"
                confidence = float(1 - binary_prob)
            else:
                stage_probs = stage_model.predict(
                    np.expand_dims(img, 0),
                    verbose=0
                )[0]
                idx = int(np.argmax(stage_probs))
                prediction = CLASS_NAMES[idx]
                confidence = float(stage_probs[idx])

            results.append({
                "File": f.name,
                "Tumor Probability": f"{binary_prob:.2%}",
                "Prediction": prediction,
                "Confidence": f"{confidence:.2%}",
                "Review?": "⚠️ Yes" if confidence < conf_threshold else "✅ No",
            })

            prog.progress((i + 1) / len(uploaded_batch))

        import pandas as pd
        df = pd.DataFrame(results)

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)

        st.download_button(
            "⬇️ Download results CSV",
            csv,
            file_name="tumorsight_batch.csv",
            mime="text/csv"
        )

# ─── Tab 3: Clinical reference ───────────────────────────────────────────────
with tabs[2]:
    st.subheader("Clinical Stage Reference")

    data = {
        "Stage": [
            "Healthy",
            "Stage I",
            "Stage II",
            "Stage III",
            "Stage IV"
        ],
        "Signs": [
            "No pathological signs",
            "Minimal mass, often asymptomatic",
            "Local growth, mild pain",
            "Local invasion, marked symptoms",
            "Metastases, systemic signs"
        ],
        "Imaging": [
            "Normal reference tissue",
            "Micro-calcifications, isolated nodules",
            "Limited organ extension",
            "Lymph node involvement",
            "Multi-focal lesions, secondary foci"
        ],
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📐 Validation KPIs")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Sensitivity Target", "> 95%")
    col2.metric("F2-Score", "Maximized")
    col3.metric("ROC-AUC Target", "> 0.90")
    col4.metric("Inference Latency", "< 200 ms")