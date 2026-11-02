"""Prueba con el modelo real; se omite si models/conv_MLP_84.h5 no está (p. ej. en CI)."""

import pytest

from src import integrator

from .conftest import REAL_MODEL

pytestmark = pytest.mark.skipif(not REAL_MODEL.is_file(), reason="modelo .h5 no disponible")


def test_real_model_predicts_valid_label(dicom_file):
    # Arrange
    array, _ = integrator.load_image(dicom_file)
    # Act
    result = integrator.predict(array)
    # Assert
    assert result.label in {"bacteriana", "normal", "viral"}
    assert 0 <= result.probability <= 100
    assert result.heatmap.shape == (512, 512, 3)
