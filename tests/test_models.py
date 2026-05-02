"""
TumorSight — Unit Tests
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ─── Preprocessing ────────────────────────────────────────────────────────────
def test_preprocess_shape():
    from PIL import Image
    import tempfile, numpy as np
    from src.preprocessing.pipeline import load_and_preprocess_image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
        img.save(f.name)
        arr = load_and_preprocess_image(f.name)
    assert arr.shape == (128, 128, 3), f"Expected (128,128,3), got {arr.shape}"
    assert arr.max() <= 1.0, "Image not normalized"
    assert arr.min() >= 0.0


# ─── Baseline CNN ──────────────────────────────────────────────────────────────
def test_baseline_cnn_output_shape():
    from src.models.baseline_cnn import build_baseline_cnn, compile_model
    model = build_baseline_cnn()
    compile_model(model)
    x = np.random.rand(2, 128, 128, 3).astype(np.float32)
    preds = model.predict(x, verbose=0)
    assert preds.shape == (2, 5), f"Expected (2,5), got {preds.shape}"
    assert abs(preds.sum(axis=1).mean() - 1.0) < 1e-5, "Probabilities do not sum to 1"


# ─── Focal Loss ───────────────────────────────────────────────────────────────
def test_focal_loss_runs():
    import tensorflow as tf
    from src.training.focal_loss import focal_loss
    loss_fn = focal_loss()
    y_true  = tf.constant([[0, 0, 0, 1, 0]], dtype=tf.float32)
    y_pred  = tf.constant([[0.05, 0.05, 0.1, 0.75, 0.05]], dtype=tf.float32)
    val = loss_fn(y_true, y_pred)
    assert val.numpy() >= 0, "Loss must be non-negative"


# ─── Autoencoder ──────────────────────────────────────────────────────────────
def test_autoencoder_reconstruction_shape():
    from src.models.autoencoder import build_autoencoder
    ae, enc, dec = build_autoencoder()
    x   = np.random.rand(1, 128, 128, 3).astype(np.float32)
    rec = ae.predict(x, verbose=0)
    assert rec.shape == x.shape, f"Reconstruction shape mismatch: {rec.shape}"


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
def test_gradcam_heatmap_shape():
    from src.models.baseline_cnn import build_baseline_cnn, compile_model
    from src.evaluation.grad_cam import generate_gradcam
    model = build_baseline_cnn()
    compile_model(model)
    image = np.random.rand(128, 128, 3).astype(np.float32)
    heatmap = generate_gradcam(model, image, class_index=0,
                               last_conv_layer_name="conv3")
    assert len(heatmap.shape) == 2, "Heatmap should be 2D"
    assert heatmap.max() <= 1.0 + 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
