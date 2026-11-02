"""Lectura de radiografías en formato DICOM y JPEG/PNG.

Responsabilidad única: convertir un archivo en (arreglo para el modelo,
imagen PIL para mostrar). No preprocesa ni predice.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pydicom
from PIL import Image

DICOM_EXTENSIONS = {".dcm"}
RASTER_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_EXTENSIONS = DICOM_EXTENSIONS | RASTER_EXTENSIONS


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Escala a 0-255 igual que el monolito: max(x, 0) / max * 255."""
    img = array.astype(float)
    max_value = img.max()
    if max_value > 0:
        img = (np.maximum(img, 0) / max_value) * 255.0
    return np.uint8(img)


def read_dicom_file(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Lee un DICOM y devuelve (arreglo RGB uint8, imagen para mostrar)."""
    dataset = pydicom.dcmread(path)
    img_array = dataset.pixel_array
    img2 = _to_uint8(img_array)
    img_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    img2show = Image.fromarray(img2)
    return img_rgb, img2show


def read_jpg_file(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Lee un JPEG/PNG y devuelve (arreglo BGR uint8, imagen para mostrar)."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    img2 = _to_uint8(np.asarray(img))
    img2show = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    return img2, img2show


def read_image(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Despacha según la extensión del archivo."""
    suffix = Path(path).suffix.lower()
    if suffix in DICOM_EXTENSIONS:
        return read_dicom_file(path)
    if suffix in RASTER_EXTENSIONS:
        return read_jpg_file(path)
    raise ValueError(f"Formato no soportado: '{suffix}'. Use {sorted(SUPPORTED_EXTENSIONS)}.")
