"""
TumorSight — Clinical Evaluation Metrics
Sensitivity, F2-Score, ROC-AUC per stage, confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    fbeta_score, recall_score,
)


CLASS_NAMES = ["Healthy", "Stage I", "Stage II", "Stage III", "Stage IV"]


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
             save_dir: str = "reports") -> dict:
    """
    Full clinical evaluation suite.
    Returns dict with key metrics.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    y_pred_prob = model.predict(X_test, verbose=0)          # (N, 5)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.argmax(y_test, axis=1)

    # ── Per-class metrics ──────────────────────────────────────────────────
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f2        = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    print(f"   Sensitivity (macro recall) : {recall:.4f}  {'✅' if recall > 0.95 else '⚠️'}")
    print(f"   F2-Score (macro)           : {f2:.4f}")
    print(f"   ROC-AUC  (macro OvR)       : {auc:.4f}  {'✅' if auc > 0.90 else '⚠️'}\n")

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, save_dir)

    # ── ROC curves ────────────────────────────────────────────────────────
    _plot_roc_curves(y_test, y_pred_prob, save_dir)

    return {"recall": recall, "f2": f2, "auc": auc}


def _plot_confusion_matrix(cm: np.ndarray, save_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)
    ax.set(
        xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix — TumorSight"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    path = f"{save_dir}/confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved → {path}")


def _plot_roc_curves(y_test: np.ndarray, y_pred_prob: np.ndarray, save_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373", "#BA68C8"]

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        auc_val = roc_auc_score(y_test[:, i], y_pred_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — per Tumour Stage")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{save_dir}/roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"   ROC curves saved        → {path}")
