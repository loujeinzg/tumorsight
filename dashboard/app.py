"""
TumorSight — Streamlit Dashboard
Interactive visualization of predictions and Grad-CAM heatmaps.
"""

import io
import base64
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
        font-size: 2.4rem; font-weight: 800; color: #00BCD4;
        border-bottom: 2px solid #00BCD4; padding-bottom: 0.4rem;
    }
    .metric-card {
        background: #1E1E2E; border-radius: 12px; padding: 1rem;
        border: 1px solid #444; text-align: center;
    }
    .prediction-badge {
        font-size: 1.6rem; font-weight: 700; padding: 0.4rem 1rem;
        border-radius: 8px;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

STAGE_COLORS = {
    "Healthy":   "#4CAF50",
    "Stage I":   "#8BC34A",
    "Stage II":  "#FFC107",
    "Stage III": "#FF5722",
    "Stage IV":  "#B71C1C",
}
CLASS_NAMES = ["Healthy", "Stage I", "Stage II", "Stage III", "Stage IV"]
IMG_SIZE    = (128, 128)


# ─── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        return None


def preprocess(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def run_gradcam(model, image: np.ndarray, class_index: int) -> np.ndarray:
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from src.evaluation.grad_cam import generate_gradcam, overlay_heatmap
    heatmap  = generate_gradcam(model, image, class_index)
    overlaid = overlay_heatmap(image, heatmap, alpha=0.45)
    return overlaid


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "dashboard/assets/logo.png",
        caption="TumorSight Dashboard",
        width=250
    )

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    model_path = st.text_input(
        "Model path", value="models/saved/best_model.keras"
    )
    conf_threshold = st.slider(
        "Confidence threshold", 0.5, 0.99, 0.70, 0.01
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**TumorSight** is a SaMD-grade deep learning system for tumour "
        "detection and staging from medical imaging (MRI, CT, Histopathology)."
    )

    st.markdown("---")
    st.caption(
        "⚕️ For clinical decision support only. Always verify with a qualified physician."
    )

# ─── Main content ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 TumorSight — Clinical Decision Support</div>',
            unsafe_allow_html=True)
st.markdown("")

model = load_model(model_path)
if model is None:
    st.warning(f"⚠️ Model not found at `{model_path}`. Train the model first, then reload.")

tabs = st.tabs(["🔬 Single Image Analysis", "📊 Batch Evaluation", "📖 Clinical Reference"])

# ── Tab 1: Single image ───────────────────────────────────────────────────────
with tabs[0]:
    col_upload, col_results = st.columns([1, 2])

    with col_upload:
        st.subheader("Upload Medical Image")
        uploaded = st.file_uploader(
            "Supported: PNG, JPG, DICOM",
            type=["png", "jpg", "jpeg"],
            help="Upload a brain MRI, CT scan, or histopathology image."
        )
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)
        target_class = st.selectbox(
            "Grad-CAM target class (optional)",
            ["Auto (predicted)"] + CLASS_NAMES
        )

        if uploaded:
            st.image(uploaded, caption="Uploaded image", use_column_width=True)

    with col_results:
        if uploaded and model is not None:
            with st.spinner("Running inference…"):
                image = preprocess(uploaded)
                probs = model.predict(np.expand_dims(image, 0), verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                prediction = CLASS_NAMES[pred_idx]

            # ── Prediction banner ──────────────────────────────────────────
            color = STAGE_COLORS.get(prediction, "#607D8B")
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="prediction-badge" style="color:{color};">'
                f'{prediction}</div>'
                f'<div style="color:#aaa; margin-top:.4rem;">Confidence: '
                f'<b style="color:{color}">{confidence:.1%}</b></div>'
                f'</div>', unsafe_allow_html=True
            )
            st.markdown("")

            # ── Alerts ────────────────────────────────────────────────────
            if confidence < conf_threshold:
                st.warning(f"⚠️ **MANUAL REVIEW REQUIRED** — Confidence {confidence:.1%} < {conf_threshold:.0%} threshold")
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
                height=280, margin=dict(l=10, r=60, t=10, b=10),
                xaxis_title="Probability", plot_bgcolor="#0f0f1a",
                paper_bgcolor="#0f0f1a", font_color="#eee",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Grad-CAM ───────────────────────────────────────────────────
            if show_gradcam:
                st.subheader("Grad-CAM Explanation")
                tgt = pred_idx if target_class == "Auto (predicted)" \
                      else CLASS_NAMES.index(target_class)
                with st.spinner("Generating Grad-CAM…"):
                    try:
                        overlay = run_gradcam(model, image, tgt)
                        c1, c2 = st.columns(2)
                        c1.image(image, caption="Original", clamp=True)
                        c2.image(overlay, caption=f"Grad-CAM → {CLASS_NAMES[tgt]}")
                    except Exception as e:
                        st.error(f"Grad-CAM error: {e}")

        elif uploaded and model is None:
            st.error("Load a valid model first (see sidebar).")

# ── Tab 2: Batch evaluation ────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Batch Evaluation")
    st.info("Upload multiple images to evaluate model performance across your dataset.")

    uploaded_batch = st.file_uploader(
        "Upload images (batch)", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_batch and model is not None:
        results = []
        prog = st.progress(0)
        for i, f in enumerate(uploaded_batch):
            img   = preprocess(f)
            probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
            idx   = int(np.argmax(probs))
            results.append({
                "File": f.name,
                "Prediction": CLASS_NAMES[idx],
                "Confidence": f"{probs[idx]:.2%}",
                "Review?": "⚠️ Yes" if probs[idx] < conf_threshold else "✅ No",
            })
            prog.progress((i + 1) / len(uploaded_batch))

        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download results CSV", csv,
                           file_name="tumorsight_batch.csv", mime="text/csv")

# ── Tab 3: Clinical reference ──────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Clinical Stage Reference")
    data = {
        "Stage":    ["Healthy", "Stage I", "Stage II", "Stage III", "Stage IV"],
        "Signs":    ["No pathological signs", "Minimal mass, asymptomatic",
                     "Local growth, mild pain", "Local invasion, marked symptoms",
                     "Metastases, systemic signs"],
        "Imaging":  ["Normal reference tissue", "Micro-calcifications, isolated nodules",
                     "Limited organ extension", "Lymph node involvement",
                     "Multi-focal lesions, secondary foci"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📐 Validation KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sensitivity Target", "> 95%", help="Minimize false negatives")
    col2.metric("F2-Score", "Maximized", help="Recall-weighted metric")
    col3.metric("ROC-AUC Target", "> 0.90", help="Per-stage discrimination")
    col4.metric("Inference Latency", "< 200 ms", help="Real-time clinical use")
