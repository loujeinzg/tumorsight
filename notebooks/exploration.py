# %% [markdown]
# # TumorSight — Exploration & Prototyping Notebook
# This notebook walks through the full pipeline: data loading → training → evaluation → Grad-CAM.

# %% Setup
import os, sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("TF version:", tf.__version__)
print("GPU available:", bool(tf.config.list_physical_devices('GPU')))

# %% [markdown]
# ## 1. Generate synthetic data (for quick testing without real dataset)

# %%
NUM_SAMPLES = 200
NUM_CLASSES = 5

X_demo = np.random.rand(NUM_SAMPLES, 128, 128, 3).astype(np.float32)
y_raw  = np.random.randint(0, NUM_CLASSES, NUM_SAMPLES)
y_demo = tf.keras.utils.to_categorical(y_raw, NUM_CLASSES)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_demo, y_demo, test_size=0.2)
print(f"Train: {X_train.shape} | Val: {X_val.shape}")

# %% [markdown]
# ## 2. Train baseline CNN (quick demo — 3 epochs)

# %%
from src.models.baseline_cnn import build_baseline_cnn, compile_model
model = build_baseline_cnn()
model = compile_model(model)
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=3, batch_size=16, verbose=1
)

# %% [markdown]
# ## 3. Plot training curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(history.history['accuracy'],     label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title('Accuracy'); axes[1].legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=100); plt.show()

# %% [markdown]
# ## 4. Evaluate

# %%
from src.evaluation.metrics import evaluate
results = evaluate(model, X_val, y_val, save_dir="reports")
print(results)

# %% [markdown]
# ## 5. Grad-CAM visualization

# %%
from src.evaluation.grad_cam import visualize_gradcam
sample_image = X_val[0]
visualize_gradcam(model, sample_image, class_index=None, save_path="gradcam_demo.png")
print("Grad-CAM saved to gradcam_demo.png")
