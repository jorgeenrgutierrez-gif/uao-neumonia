import numpy as np
import pytest
from PIL import Image

from src import read_img


def test_read_dicom_returns_rgb_uint8_array(dicom_file):
    # Arrange
    path = dicom_file
    # Act
    array, preview = read_img.read_dicom_file(path)
    # Assert
    assert array.dtype == np.uint8
    assert array.shape == (300, 300, 3)
    assert isinstance(preview, Image.Image)


def test_read_dicom_scales_max_value_to_255(dicom_file):
    # Arrange
    path = dicom_file
    # Act
    array, _ = read_img.read_dicom_file(path)
    # Assert
    assert array.max() == 255
    assert array.min() == 0


def test_read_jpg_returns_three_channels(jpg_file):
    # Arrange
    path = jpg_file
    # Act
    array, preview = read_img.read_jpg_file(path)
    # Assert
    assert array.ndim == 3 and array.shape[2] == 3
    assert preview.size == (300, 300)


def test_read_jpg_supports_paths_with_accents(jpg_file_unicode_path):
    # Arrange
    path = jpg_file_unicode_path
    # Act
    array, preview = read_img.read_image(path)
    # Assert
    assert array.shape == (300, 300, 3)
    assert preview.size == (300, 300)


def test_read_image_dispatches_by_extension(dicom_file, jpg_file):
    # Arrange
    paths = [dicom_file, jpg_file]
    # Act
    results = [read_img.read_image(p) for p in paths]
    # Assert
    assert all(arr.dtype == np.uint8 for arr, _ in results)


def test_read_image_rejects_unsupported_extension(tmp_path):
    # Arrange
    path = tmp_path / "notas.txt"
    path.write_text("no es una imagen")
    # Act / Assert
    with pytest.raises(ValueError, match="no soportado"):
        read_img.read_image(path)


def test_read_jpg_raises_for_missing_file(tmp_path):
    # Arrange
    path = tmp_path / "no_existe.jpg"
    # Act / Assert
    with pytest.raises(ValueError, match="No se pudo leer"):
        read_img.read_jpg_file(path)
        
        