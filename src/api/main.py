"""
TumorSight — FastAPI Inference API
Endpoints: /predict, /predict/gradcam, /health, /model/info
"""

import os
import io
import base64
import time
import numpy as np
from PIL import Image
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf


# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH       = os.getenv("MODEL_PATH", "models/saved/best_model.keras")
AUTOENCODER_PATH = os.getenv("AE_PATH",    "models/saved/autoencoder.keras")
IMG_SIZE         = (128, 128)
CONFIDENCE_THRESHOLD = 0.70
CLASS_NAMES      = ["Healthy", "Stage I", "Stage II", "Stage III", "Stage IV"]

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TumorSight API",
    description="Medical tumour detection & stage classification — Deep Learning",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── Global model state ───────────────────────────────────────────────────────
_model        = None
_autoencoder  = None
_ae_threshold = None


def _load_models():
    global _model, _autoencoder, _ae_threshold

    if os.path.exists(MODEL_PATH):
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Main model loaded: {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}. Train first.")

    if os.path.exists(AUTOENCODER_PATH):
        _autoencoder = tf.keras.models.load_model(AUTOENCODER_PATH)
        print(f"✅ Autoencoder loaded: {AUTOENCODER_PATH}")


@app.on_event("startup")
def startup_event():
    _load_models()


# ─── Utilities ────────────────────────────────────────────────────────────────
def preprocess_upload(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr                                    # (128, 128, 3)


def reconstruct_error(image: np.ndarray) -> float:
    if _autoencoder is None:
        return 0.0
    inp  = np.expand_dims(image, 0)
    rec  = _autoencoder.predict(inp, verbose=0)
    return float(np.mean((inp - rec) ** 2))


# ─── Response schemas ──────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    prediction:      str
    confidence:      float
    probabilities:   list[float]
    anomaly_flag:    bool
    review_required: bool
    latency_ms:      float


class GradCAMResponse(PredictResponse):
    heatmap_base64: str    # PNG encoded as base64


class HealthResponse(BaseModel):
    status:     str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_path:  str
    input_shape: list
    num_classes: int
    class_names: list[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "model_path":  MODEL_PATH,
        "input_shape": list(_model.input_shape[1:]),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded — run training first")

    t0          = time.perf_counter()
    file_bytes  = await file.read()
    image       = preprocess_upload(file_bytes)
    inp         = np.expand_dims(image, 0)

    probs       = _model.predict(inp, verbose=0)[0].tolist()
    class_idx   = int(np.argmax(probs))
    confidence  = float(probs[class_idx])
    latency_ms  = (time.perf_counter() - t0) * 1000

    ae_error    = reconstruct_error(image)
    ae_thresh   = _ae_threshold or 0.05
    anomaly     = ae_error > ae_thresh

    return PredictResponse(
        prediction      = CLASS_NAMES[class_idx],
        confidence      = round(confidence, 4),
        probabilities   = [round(p, 4) for p in probs],
        anomaly_flag    = anomaly,
        review_required = confidence < CONFIDENCE_THRESHOLD or anomaly,
        latency_ms      = round(latency_ms, 2),
    )


@app.post("/predict/gradcam", response_model=GradCAMResponse, tags=["Inference"])
async def predict_with_gradcam(file: UploadFile = File(...),
                                class_index: Optional[int] = None):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    t0         = time.perf_counter()
    file_bytes = await file.read()
    image      = preprocess_upload(file_bytes)
    inp        = np.expand_dims(image, 0)

    probs      = _model.predict(inp, verbose=0)[0].tolist()
    pred_idx   = int(np.argmax(probs))
    target_idx = class_index if class_index is not None else pred_idx
    confidence = float(probs[pred_idx])

    # Grad-CAM
    from src.evaluation.grad_cam import generate_gradcam, overlay_heatmap
    import cv2
    heatmap   = generate_gradcam(_model, image, target_idx)
    overlaid  = overlay_heatmap(image, heatmap)
    pil_img   = Image.fromarray(overlaid)
    buf       = io.BytesIO()
    pil_img.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

    latency_ms = (time.perf_counter() - t0) * 1000
    ae_error   = reconstruct_error(image)
    ae_thresh  = _ae_threshold or 0.05
    anomaly    = ae_error > ae_thresh

    return GradCAMResponse(
        prediction      = CLASS_NAMES[pred_idx],
        confidence      = round(confidence, 4),
        probabilities   = [round(p, 4) for p in probs],
        anomaly_flag    = anomaly,
        review_required = confidence < CONFIDENCE_THRESHOLD or anomaly,
        latency_ms      = round(latency_ms, 2),
        heatmap_base64  = heatmap_b64,
    )
