"""Carga del modelo entrenado (.h5) una sola vez por proceso.

Responsabilidad única: ubicar el archivo del modelo y devolver la
instancia de Keras, cacheada para que las predicciones y Grad-CAM
reutilicen el mismo objeto.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "conv_MLP_84.h5"
MODEL_ENV_VAR = "NEUMONIA_MODEL_PATH"


def resolve_model_path() -> Path:
    """Ruta del modelo: variable de entorno o `models/conv_MLP_84.h5`."""
    return Path(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def load_model():
    """Carga el modelo una sola vez y lo reutiliza en llamadas siguientes."""
    path = resolve_model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en '{path}'. "
            f"Copie conv_MLP_84.h5 a models/ o defina {MODEL_ENV_VAR}."
        )
    import keras  # import diferido: cargar TensorFlow solo cuando se necesita

    return keras.models.load_model(path, compile=False)
