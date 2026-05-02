# 🚀 TumorSight — Complete Run Guide (Laptop Setup)

> Step-by-step instructions to train models, prepare data, and run the API + Dashboard locally.

---

## 📋 Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | ≥ 3.9 | `python --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| Docker (optional) | ≥ 20.x | `docker --version` |

> **GPU (optional but recommended):** If you have an NVIDIA GPU, install CUDA 11.x + cuDNN.  
> Without GPU, training still works on CPU — just slower (~5–10× slower per epoch).

---

## ⚡ Step 1 — Install dependencies

```bash
# Clone the project
git clone https://github.com/votre-org/tumorsight.git
cd tumorsight

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# OR
venv\Scripts\activate           # Windows

# Install all packages
pip install -r requirements.txt
```

---

## 🗂️ Step 2 — Prepare the Dataset

### Option A: Use Kaggle datasets (recommended)

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Place your `kaggle.json` API key in `~/.kaggle/kaggle.json`

3. Download the datasets:
   ```bash
   # Brain Tumor MRI (IRM classification)
   kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset -p data/raw/ --unzip

   # Breast Histopathology (optional second dataset)
   kaggle datasets download paultimothymooney/breast-histopathology-images -p data/raw/ --unzip
   ```

4. **Organize into this folder structure** (rename folders to match class names):
   ```
   data/raw/
   ├── healthy/       ← images of healthy tissue
   ├── stage_I/       ← stage I tumour images
   ├── stage_II/      ← stage II tumour images
   ├── stage_III/     ← stage III tumour images
   └── stage_IV/      ← stage IV tumour images
   ```
   > For the Brain Tumor MRI dataset, map: `no_tumor → healthy`, and distribute `glioma`, `meningioma`, `pituitary` across stages I–IV as appropriate for your clinical protocol.

### Option B: Use synthetic data (for quick testing — no download needed)

```bash
python notebooks/exploration.py
```
This generates random images and trains a demo model in ~30 seconds on CPU.

---

## 🔄 Step 3 — Preprocess and Split Data

```bash
python src/preprocessing/pipeline.py \
  --data-dir data/raw/ \
  --output-dir data/splits/
```

This creates:
```
data/splits/
├── X_train.npy   # Training images
├── y_train.npy   # Training labels (one-hot)
├── X_val.npy     # Validation set
├── y_val.npy
├── X_test.npy    # Test set (held out)
└── y_test.npy
```

---

## 🏋️ Step 4 — Train a Model

### 4a. Baseline CNN (fastest, CPU-friendly)

```bash
python src/training/train.py \
  --model baseline \
  --epochs 50 \
  --batch-size 32 \
  --data-dir data/splits/
```

### 4b. ResNet50V2 (better accuracy, needs more RAM)

```bash
python src/training/train.py \
  --model resnet50v2 \
  --epochs 30 \
  --batch-size 16 \
  --data-dir data/splits/ \
  --focal-loss        # Recommended for imbalanced classes
```

### 4c. EfficientNetB0 (best accuracy / size trade-off)

```bash
python src/training/train.py \
  --model efficientnetb0 \
  --epochs 30 \
  --batch-size 16 \
  --data-dir data/splits/ \
  --focal-loss \
  --fine-tune         # Unfreeze top layers after initial training
```

> **Where is the model saved?**  
> → `models/saved/best_model.keras` (auto-created)  
> → Training log: `models/saved/training_log.csv`

### ⏱️ Expected training time (CPU only)

| Model | 50 epochs | Notes |
|-------|-----------|-------|
| Baseline CNN | ~15–30 min | CPU comfortable |
| ResNet50V2   | ~45–90 min | Needs ≥ 8 GB RAM |
| EfficientNetB0 | ~30–60 min | Best CPU option |

---

## 📊 Step 5 — Evaluate the Model

```bash
python -c "
import numpy as np, tensorflow as tf
from src.evaluation.metrics import evaluate

model  = tf.keras.models.load_model('models/saved/best_model.keras')
X_test = np.load('data/splits/X_test.npy')
y_test = np.load('data/splits/y_test.npy')
evaluate(model, X_test, y_test, save_dir='reports/')
"
```

Outputs saved to `reports/`:
- `confusion_matrix.png`
- `roc_curves.png`

---

## 🌐 Step 6 — Launch the FastAPI

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser: **http://localhost:8000/docs** (Swagger UI — interactive API)

### Quick test via curl:

```bash
# Health check
curl http://localhost:8000/health

# Predict on an image
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/your/image.png"

# Predict + Grad-CAM heatmap
curl -X POST "http://localhost:8000/predict/gradcam" \
     -F "file=@/path/to/your/image.png" \
     -o response.json
```

---

## 🖥️ Step 7 — Launch the Dashboard (Streamlit)

```bash
streamlit run dashboard/app.py
```

Open your browser: **http://localhost:8501**

Features:
- Upload medical images
- See prediction + confidence
- Visualize Grad-CAM heatmaps
- Run batch evaluations
- Download results as CSV

---

## 🐳 Step 8 — Docker (optional — full stack)

```bash
cd docker
docker-compose up --build
```

This starts both:
- API at **http://localhost:8000**
- Dashboard at **http://localhost:8501**

---

## 🧪 Step 9 — Run Unit Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/ -v
```

Expected output: all 5 tests pass ✅

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: src` | Run commands from the project root (`cd tumorsight`) |
| `Model not found` error in API | Train first (Step 4), then restart the API |
| Out of memory during training | Reduce `--batch-size` to 8 or 4 |
| Slow training on CPU | Use `--epochs 10` for quick testing; buy time with baseline model |
| CUDA errors | Set `CUDA_VISIBLE_DEVICES=""` to force CPU: `export CUDA_VISIBLE_DEVICES=""` |
| `pydicom` not found for DICOM | `pip install pydicom` |

---

## 📂 Quick Reference — Key Commands

```bash
# Preprocess data
python src/preprocessing/pipeline.py --data-dir data/raw/ --output-dir data/splits/

# Train (baseline)
python src/training/train.py --model baseline --epochs 50 --batch-size 32 --data-dir data/splits/

# Train (ResNet, with focal loss)
python src/training/train.py --model resnet50v2 --epochs 30 --batch-size 16 --data-dir data/splits/ --focal-loss

# API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Dashboard
streamlit run dashboard/app.py

# Tests
pytest tests/ -v

# Docker (full stack)
cd docker && docker-compose up --build
```

---

*TumorSight — Allier la puissance du Deep Learning à la rigueur de l'ingénierie médicale.*
