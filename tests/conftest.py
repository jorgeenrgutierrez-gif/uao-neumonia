"""Fixtures compartidas: imágenes sintéticas y un modelo pequeño de prueba."""

from pathlib import Path

import cv2
import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

REAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "conv_MLP_84.h5"


@pytest.fixture
def gray_gradient() -> np.ndarray:
    return np.tile(np.linspace(0, 200, 300, dtype=np.uint8), (300, 1))


@pytest.fixture
def dicom_file(tmp_path, gray_gradient) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    path = tmp_path / "torax.dcm"
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = gray_gradient.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = gray_gradient.tobytes()
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    return path


@pytest.fixture
def jpg_file(tmp_path, gray_gradient) -> Path:
    path = tmp_path / "torax.jpeg"
    _write_jpeg(path, gray_gradient)
    return path


@pytest.fixture
def jpg_file_unicode_path(tmp_path, gray_gradient) -> Path:
    """JPEG dentro de una carpeta con tildes, como 'GESTIÓN DE PROCESOS' en Windows."""
    folder = tmp_path / "GESTIÓN Radiografías ñ"
    folder.mkdir()
    path = folder / "tórax.jpeg"
    _write_jpeg(path, gray_gradient)
    return path


def _write_jpeg(path: Path, gray: np.ndarray) -> None:
    # imencode + write_bytes: funciona con rutas Unicode en cualquier sistema operativo.
    ok, buffer = cv2.imencode(".jpeg", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert ok
    path.write_bytes(buffer.tobytes())


@pytest.fixture(scope="session")
def tiny_model():
    """Modelo mínimo con la misma entrada/salida y la capa 'conv10_thisone'."""
    import keras

    inputs = keras.Input(shape=(512, 512, 1))
    x = keras.layers.Conv2D(4, 3, strides=8, activation="relu")(inputs)
    x = keras.layers.Conv2D(8, 3, padding="same", name="conv10_thisone")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(3, activation="softmax", name="fc3")(x)
    return keras.Model(inputs, outputs)
