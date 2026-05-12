import numpy as np
import tensorflow as tf
import cv2


def find_last_conv_layer(model):
    """
    Find last Conv2D layer, even inside nested models like EfficientNet.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, model

        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name, layer

    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam(model, image, class_index):
    """
    Generate Grad-CAM heatmap.
    image shape: (128, 128, 3)
    """

    conv_layer_name, conv_model = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            conv_model.get_layer(conv_layer_name).output,
            model.output
        ]
    )

    img_array = np.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.45):
    """
    Overlay heatmap on original image.
    """

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    image_uint8 = np.uint8(image * 255)

    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_color, alpha, 0)

    return overlay