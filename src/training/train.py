"""
TumorSight — Training Script

Models:
1) binary        : CNN detects Tumor / No Tumor
2) baseline      : CNN multi-class classifier
3) resnet50v2    : Transfer Learning classifier
4) efficientnetb0: Transfer Learning classifier

Examples:
python -m src.training.train --model binary --epochs 10 --batch-size 16 --data-dir data_binary_splits --save-dir models/saved_binary

python -m src.training.train --model efficientnetb0 --epochs 10 --batch-size 16 --data-dir data_stages_splits --save-dir models/saved_stages
"""

import os
import argparse
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="TumorSight training")

    parser.add_argument(
        "--model",
        default="baseline",
        choices=["binary", "baseline", "resnet50v2", "efficientnetb0"],
        help="Model architecture"
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument(
        "--data-dir",
        default="data/splits",
        help="Directory containing X_train.npy, y_train.npy, X_val.npy, y_val.npy"
    )

    parser.add_argument(
        "--save-dir",
        default="models/saved",
        help="Directory where the best model will be saved"
    )

    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use Focal Loss instead of categorical crossentropy"
    )

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze top layers after initial training for transfer models"
    )

    return parser.parse_args()


def load_splits(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))

    print(f"Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"y_train: {y_train.shape} | y_val: {y_val.shape}")

    return X_train, y_train, X_val, y_val


def build_model(args, y_train):
    base = None

    if args.model == "binary":
        from src.models.binary_cnn import build_binary_cnn
        model = build_binary_cnn()

        loss = "binary_crossentropy"

        metrics = [
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ]

    elif args.model == "baseline":
        from src.models.baseline_cnn import build_baseline_cnn
        model = build_baseline_cnn()

        loss = "categorical_crossentropy"

        metrics = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ]

    else:
        from src.models.transfer_learning import build_transfer_model
        model, base = build_transfer_model(backbone=args.model)

        if args.focal_loss:
            from src.training.focal_loss import focal_loss
            loss = focal_loss(gamma=2.0, alpha=0.25)
        else:
            loss = "categorical_crossentropy"

        metrics = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=loss,
        metrics=metrics
    )

    return model, base


def get_callbacks(save_dir: str, monitor_metric: str = "val_recall"):
    os.makedirs(save_dir, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "best_model.keras"),
            monitor=monitor_metric,
            save_best_only=True,
            mode="max",
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
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
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Focal Loss: {args.focal_loss} | Fine-Tune: {args.fine_tune}")
    print(f"Data dir: {args.data_dir}")
    print(f"Save dir: {args.save_dir}\n")

    X_train, y_train, X_val, y_val = load_splits(args.data_dir)

    if args.model == "binary":
        y_train = np.argmax(y_train, axis=1)
        y_val = np.argmax(y_val, axis=1)

        # healthy = 0, stage_I/tumor = 1
        y_train = (y_train != 0).astype("float32")
        y_val = (y_val != 0).astype("float32")

        print(f"Binary y_train: {y_train.shape} | Binary y_val: {y_val.shape}")

    model, base = build_model(args, y_train)
    model.summary()

    callbacks = get_callbacks(args.save_dir)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if args.fine_tune and base is not None:
        print("\n🔬 Fine-tuning top layers...")

        from src.models.transfer_learning import unfreeze_top_layers

        unfreeze_top_layers(base, num_layers=30)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr / 10),
            loss=model.loss,
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    print(f"\n✅ Training complete.")
    print(f"✅ Best model saved to: {args.save_dir}/best_model.keras")


if __name__ == "__main__":
    main()