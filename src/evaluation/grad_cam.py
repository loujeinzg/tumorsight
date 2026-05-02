"""
TumorSight — Grad-CAM (Gradient-weighted Class Activation Mapping)
Generates heatmaps highlighting regions that drove the model's decision.
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm


CLASS_NAMES = ["Healthy", "Stage I", "Stage II", "Stage III", "Stage IV"]


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """Auto-detect the last Conv2D layer name."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int,
    last_conv_layer_name: str = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given image and class.

    Args:
        model            : Trained Keras model.
        image            : Preprocessed image (H, W, 3), values in [0, 1].
        class_index      : Target class (0=Healthy … 4=Stage IV).
        last_conv_layer_name : Name of the last convolutional layer (auto-detected if None).

    Returns:
        heatmap (H, W) — normalized to [0, 1].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_tensor = tf.expand_dims(image, axis=0)  # (1, H, W, 3)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)                 # (1, h, w, filters)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # (filters,)

    conv_outputs = conv_outputs[0]                            # (h, w, filters)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]    # (h, w, 1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superimpose the Grad-CAM heatmap on the original image.

    Returns:
        RGB numpy array (H, W, 3) — values uint8 [0, 255].
    """
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colored_map     = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_map     = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

    orig_uint8 = np.uint8(255 * original_image) if original_image.max() <= 1.0 \
                 else original_image.astype(np.uint8)

    superimposed = (colored_map * alpha + orig_uint8 * (1 - alpha)).astype(np.uint8)
    return superimposed


def visualize_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int = None,
    save_path: str = None,
) -> None:
    """
    Display or save a Grad-CAM visualization with side-by-side comparison.
    If class_index is None, uses predicted class.
    """
    probs = model.predict(np.expand_dims(image, 0), verbose=0)[0]
    if class_index is None:
        class_index = int(np.argmax(probs))

    heatmap    = generate_gradcam(model, image, class_index)
    overlaid   = overlay_heatmap(image, heatmap)
    stage_name = CLASS_NAMES[class_index]
    confidence = probs[class_index]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Grad-CAM — {stage_name}  (confidence: {confidence:.2%})",
        fontsize=14, fontweight="bold"
    )
    axes[0].imshow(image); axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Heatmap")
    axes[2].imshow(overlaid); axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Grad-CAM saved → {save_path}")
    else:
        plt.show()
