"""
TumorSight — Preprocessing Pipeline
Handles DICOM, PNG, JPG medical images: resize, normalize, encode labels.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


# ─── Label Mapping ────────────────────────────────────────────────────────────
CLASS_NAMES = ["healthy", "stage_I", "stage_II", "stage_III", "stage_IV"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMG_SIZE = (128, 128)


def load_and_preprocess_image(path: str, target_size=IMG_SIZE) -> np.ndarray:
    """Load an image (DICOM, PNG, JPG), resize and normalize to [0, 1]."""
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
            # Normalize DICOM (may be 16-bit)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            if len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
        except ImportError:
            raise ImportError("Install pydicom: pip install pydicom")
    else:
        img = Image.open(path).convert("RGB")

    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def build_dataset_from_directory(data_dir: str):
    """
    Expects structure:
        data_dir/
            healthy/
            stage_I/
            stage_II/
            stage_III/
            stage_IV/
    Returns X (N, 128, 128, 3) and y_onehot (N, 5).
    """
    images, labels = [], []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARNING] Directory not found: {class_dir}")
            continue

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".dcm"))
        ]

        for fname in tqdm(files, desc=f"Loading {class_name}"):
            path = os.path.join(class_dir, fname)
            try:
                img = load_and_preprocess_image(path)
                images.append(img)
                labels.append(CLASS_TO_IDX[class_name])
            except Exception as e:
                print(f"[SKIP] {fname}: {e}")

    X = np.array(images, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(CLASS_NAMES))
    return X, y


def split_and_save(data_dir: str, output_dir: str,
                   val_size=0.15, test_size=0.15, seed=42):
    """Build dataset, split, and save numpy arrays to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    X, y = build_dataset_from_directory(data_dir)
    labels_idx = np.argmax(y, axis=1)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=labels_idx, random_state=seed
    )
    ratio = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=ratio,
        stratify=np.argmax(y_tmp, axis=1), random_state=seed
    )

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(output_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(output_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_test)

    print(f"\n✅ Dataset splits saved to {output_dir}")
    print(f"   Train : {X_train.shape[0]} samples")
    print(f"   Val   : {X_val.shape[0]} samples")
    print(f"   Test  : {X_test.shape[0]} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


def augmentation_layer():
    """Keras Sequential augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    required=True, help="Raw image directory")
    parser.add_argument("--output-dir",  default="data/splits")
    args = parser.parse_args()
    split_and_save(args.data_dir, args.output_dir)
