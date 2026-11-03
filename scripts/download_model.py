"""Descarga el modelo entrenado (conv_MLP_84.h5) desde el Release de GitHub.

El modelo pesa 118 MB y GitHub no acepta archivos de más de 100 MB en el
repositorio, por eso se publica como archivo del Release y se descarga la
primera vez que se ejecuta `make run` (o con `make model`).

Uso:
    uv run python scripts/download_model.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://github.com/jorgeenrgutierrez-gif/uao-neumonia/releases/download/"
    "modelo-v1/conv_MLP_84.h5"
)
MODEL_SHA256 = "d9548ff6fd2c29cdd15328ff95e1ad1fc8c7226e04e86c5890526bea2215e18e"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "models" / "conv_MLP_84.h5"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def sha256_of(path: Path) -> str:
    """Huella SHA-256 del archivo, leída por bloques para no cargarlo en memoria."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(
    url: str = MODEL_URL,
    dest: Path = DEFAULT_DEST,
    expected_sha256: str | None = MODEL_SHA256,
) -> Path:
    """Descarga el modelo a `dest` y verifica su integridad.

    Si el archivo ya existe y su huella coincide, no lo vuelve a descargar.
    Descarga primero a un archivo temporal `.part` para no dejar un modelo
    a medias si la conexión se corta.
    """
    dest = Path(dest)
    if dest.is_file() and (expected_sha256 is None or sha256_of(dest) == expected_sha256):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"Descargando el modelo desde {url} ...")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as file:  # noqa: S310
        while block := response.read(CHUNK_SIZE):
            file.write(block)

    if expected_sha256 is not None and sha256_of(partial) != expected_sha256:
        partial.unlink()
        raise ValueError("El archivo descargado no coincide con el modelo esperado (SHA-256).")

    partial.replace(dest)
    print(f"Modelo guardado en {dest}")
    return dest


def main() -> int:
    try:
        download_model()
    except (OSError, ValueError) as exc:
        print(f"No se pudo descargar el modelo: {exc}", file=sys.stderr)
        print("Alternativa: copie conv_MLP_84.h5 a models/ manualmente.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
