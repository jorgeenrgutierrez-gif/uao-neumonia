"""Preprocesamiento de la imagen para el modelo.

Pasos (idénticos al monolito): resize 512x512, escala de grises,
CLAHE (clipLimit=2.0, tile 4x4), normalización a [0, 1] y formato batch.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = (512, 512)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (4, 4)


def preprocess(array: np.ndarray) -> np.ndarray:
    """Devuelve un tensor (1, 512, 512, 1) en float con valores en [0, 1]."""
    array = cv2.resize(array, IMAGE_SIZE)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array
