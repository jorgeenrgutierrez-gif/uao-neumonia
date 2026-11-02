"""Mapa de calor Grad-CAM superpuesto a la radiografía.

Recibe el modelo ya cargado (no lo carga): así el módulo es testeable
con un modelo pequeño y no repite la carga del .h5.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

TARGET_LAYER = "conv10_thisone"
HEATMAP_INTENSITY = 0.8
OUTPUT_SIZE = (512, 512)


def compute_heatmap(
    model, batch: np.ndarray, class_index: int, layer_name: str = TARGET_LAYER
) -> np.ndarray:
    """Heatmap normalizado [0, 1] de la capa convolucional objetivo."""
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.outputs[0]])
    inputs = tf.convert_to_tensor(batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(inputs, training=False)
        class_score = predictions[:, class_index]
    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    weighted = conv_output[0] * pooled_grads  # pondera cada filtro
    heatmap = tf.reduce_mean(weighted, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    max_value = heatmap.max()
    if max_value > 0:
        heatmap /= max_value
    return heatmap


def overlay(heatmap: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Colorea el heatmap (JET) y lo suma a la imagen original 512x512 en RGB."""
    heatmap = cv2.resize(heatmap, OUTPUT_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    base = cv2.resize(array, OUTPUT_SIZE)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    transparency = (heatmap * HEATMAP_INTENSITY).astype(np.uint8)
    superimposed = cv2.add(transparency, base).astype(np.uint8)
    return superimposed[:, :, ::-1]


def grad_cam(model, batch: np.ndarray, array: np.ndarray, class_index: int) -> np.ndarray:
    """Imagen 512x512x3 (RGB, uint8) con el mapa de calor superpuesto."""
    heatmap = compute_heatmap(model, batch, class_index)
    return overlay(heatmap, array)
