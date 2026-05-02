"""
TumorSight — Custom Loss Functions
Focal Loss and Weighted Cross-Entropy to handle class imbalance.
"""

import numpy as np
import tensorflow as tf


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss for multi-class classification.
    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Args:
        gamma: Focusing parameter. Higher → harder examples get more weight.
        alpha: Class balance factor.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    loss_fn.__name__ = f"focal_loss_g{gamma}_a{alpha}"
    return loss_fn


def weighted_cross_entropy(class_weights: np.ndarray):
    """
    Categorical cross-entropy weighted by per-class sample imbalance.

    Args:
        class_weights: Array of shape (num_classes,), e.g. [1.0, 2.0, 3.0, 4.0, 5.0]
    """
    weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        w      = tf.reduce_sum(y_true * weights_tensor, axis=-1)
        return tf.reduce_mean(w * ce)
    loss_fn.__name__ = "weighted_cross_entropy"
    return loss_fn


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Compute inverse-frequency class weights from one-hot labels.
    Returns array of shape (num_classes,).
    """
    counts = y_train.sum(axis=0)                  # samples per class
    total  = counts.sum()
    n_cls  = len(counts)
    weights = total / (n_cls * counts + 1e-8)     # inverse frequency
    weights = weights / weights.min()             # normalize so min = 1
    print("[Class weights]", dict(enumerate(weights.round(3))))
    return weights.astype(np.float32)
