"""Orquestador (controlador): leer → preprocesar → predecir → explicar.

Es la única puerta de entrada que usa la interfaz. Expone lo que la UI
necesita y oculta el modelo, TensorFlow y el preprocesamiento.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src import grad_cam, load_model, preprocess_img, read_img

LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}
SUPPORTED_FILETYPES = (
    ("DICOM", "*.dcm"),
    ("JPEG", "*.jpeg"),
    ("jpg files", "*.jpg"),
    ("png files", "*.png"),
)


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float  # porcentaje 0-100
    heatmap: np.ndarray  # RGB uint8 512x512x3


def load_image(path: str | Path):
    """Lee DICOM o JPEG y devuelve (arreglo para el modelo, imagen PIL)."""
    return read_img.read_image(path)


def label_for(class_index: int) -> str:
    try:
        return LABELS[class_index]
    except KeyError as exc:
        raise ValueError(f"Clase desconocida: {class_index}") from exc


def predict(array: np.ndarray) -> Prediction:
    """Pipeline completo. Predice una sola vez y reutiliza el modelo cacheado."""
    batch = preprocess_img.preprocess(array)
    model = load_model.load_model()
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    class_index = int(np.argmax(probabilities))
    proba = float(np.max(probabilities) * 100)
    heatmap = grad_cam.grad_cam(model, batch, array, class_index)
    return Prediction(label_for(class_index), proba, heatmap)


def save_result_csv(
    patient_id: str, label: str, probability: float, path: str | Path = "historial.csv"
) -> None:
    """Agrega una fila al historial con el mismo formato del monolito (delimitador '-')."""
    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="-")
        writer.writerow([patient_id, label, f"{probability:.2f}%"])


def export_pdf(
    patient_id: str,
    original: Image.Image,
    prediction: Prediction,
    path: str | Path,
) -> Path:
    """Genera un PDF de una página con la radiografía, el Grad-CAM y el resultado."""
    canvas = Image.new("RGB", (815, 420), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), "SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", fill="black")
    canvas.paste(original.convert("RGB").resize((300, 300)), (50, 50))
    canvas.paste(Image.fromarray(prediction.heatmap).resize((300, 300)), (465, 50))
    draw.text((50, 370), f"Cédula paciente: {patient_id}", fill="black")
    draw.text((465, 360), f"Resultado: {prediction.label}", fill="black")
    draw.text((465, 385), f"Probabilidad: {prediction.probability:.2f}%", fill="black")
    path = Path(path)
    canvas.save(path, "PDF")
    return path
