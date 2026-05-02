"""
TumorSight — Training Script
Trains baseline CNN or transfer learning model with optional focal loss.

Usage:
    python src/training/train.py --model baseline --epochs 50 --batch-size 32 --data-dir data/splits/
    python src/training/train.py --model resnet50v2 --epochs 30 --batch-size 16 --data-dir data/splits/ --focal-loss
"""

import os
import argparse
import numpy as np
import tensorflow as tf


def parse_args():
    p = argparse.ArgumentParser(description="TumorSight training")
    p.add_argument("--model",      default="baseline",
                   choices=["baseline", "resnet50v2", "efficientnetb0"],
                   help="Model architecture")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data-dir",   default="data/splits",
                   help="Directory with X_train.npy etc.")
    p.add_argument("--save-dir",   default="models/saved",
                   help="Where to save best model weights")
    p.add_argument("--focal-loss", action="store_true",
                   help="Use Focal Loss instead of cross-entropy")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--fine-tune",  action="store_true",
                   help="Unfreeze top layers after initial training (transfer only)")
    return p.parse_args()


def load_splits(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def build_model(args):
    if args.model == "baseline":
        from src.models.baseline_cnn import build_baseline_cnn, compile_model
        model = build_baseline_cnn()
        base  = None
    else:
        from src.models.transfer_learning import build_transfer_model, compile_model
        model, base = build_transfer_model(backbone=args.model)

    if args.focal_loss:
        from src.training.focal_loss import focal_loss
        loss = focal_loss(gamma=2.0, alpha=0.25)
    else:
        loss = "categorical_crossentropy"

    from src.models.baseline_cnn import compile_model as _compile
    # reuse compile helper (works for all models)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model, base


def get_callbacks(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "best_model.keras"),
            monitor="val_recall",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            patience=10,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(save_dir, "training_log.csv")
        ),
    ]


def main():
    args = parse_args()
    print(f"\n🧠 TumorSight — Training [{args.model.upper()}]")
    print(f"   Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"   Focal Loss: {args.focal_loss} | Fine-Tune: {args.fine_tune}\n")

    X_train, y_train, X_val, y_val = load_splits(args.data_dir)
    model, base = build_model(args)
    model.summary()

    # ── Phase 1: Train head ───────────────────────────────────────────────────
    callbacks = get_callbacks(args.save_dir)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Phase 2: Fine-tune (transfer models only) ─────────────────────────────
    if args.fine_tune and base is not None:
        print("\n🔬 Fine-tuning top layers…")
        from src.models.transfer_learning import unfreeze_top_layers
        unfreeze_top_layers(base, num_layers=30)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr / 10),
            loss=model.loss,
            metrics=model.metrics,
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    print(f"\n✅ Training complete. Best model saved to {args.save_dir}/best_model.keras")


if __name__ == "__main__":
    main()
