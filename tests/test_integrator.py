import numpy as np
import pytest
from PIL import Image

from src import integrator, load_model


class FakeModel:
    """Doble de prueba: siempre predice 'viral' con 90 %."""

    def predict(self, batch, verbose=0):
        return np.array([[0.05, 0.05, 0.90]])


def test_label_mapping_matches_original_monolith():
    # Arrange
    expected = {0: "bacteriana", 1: "normal", 2: "viral"}
    # Act
    labels = {i: integrator.label_for(i) for i in expected}
    # Assert
    assert labels == expected


def test_unknown_class_raises():
    # Arrange
    invalid_index = 7
    # Act / Assert
    with pytest.raises(ValueError):
        integrator.label_for(invalid_index)


def test_predict_orchestrates_pipeline(monkeypatch):
    # Arrange
    monkeypatch.setattr(load_model, "load_model", lambda: FakeModel())
    monkeypatch.setattr(
        integrator.grad_cam,
        "grad_cam",
        lambda model, batch, array, idx: np.zeros((512, 512, 3), dtype=np.uint8),
    )
    array = np.zeros((600, 600, 3), dtype=np.uint8)
    # Act
    result = integrator.predict(array)
    # Assert
    assert result.label == "viral"
    assert result.probability == pytest.approx(90.0)
    assert result.heatmap.shape == (512, 512, 3)


def test_save_result_csv_keeps_original_format(tmp_path):
    # Arrange
    path = tmp_path / "historial.csv"
    # Act
    integrator.save_result_csv("123456", "normal", 87.456, path)
    # Assert
    assert path.read_text(encoding="utf-8").strip() == "123456-normal-87.46%"


def test_export_pdf_creates_file(tmp_path):
    # Arrange
    prediction = integrator.Prediction("normal", 80.0, np.zeros((512, 512, 3), np.uint8))
    original = Image.new("L", (300, 300))
    # Act
    pdf = integrator.export_pdf("123", original, prediction, tmp_path / "Reporte0.pdf")
    # Assert
    assert pdf.read_bytes().startswith(b"%PDF")
