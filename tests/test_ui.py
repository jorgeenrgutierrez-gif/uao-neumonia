"""Pruebas de la vista Tkinter sin cargar el modelo.

El integrador se reemplaza por dobles de prueba (monkeypatch), así se verifica
que la UI reacciona bien a cada botón sin TensorFlow. Si el equipo no tiene
pantalla (p. ej. CI en Linux sin xvfb), las pruebas se omiten.
"""

import tkinter as tk

import numpy as np
import pytest
from PIL import Image

from src import integrator, ui


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No hay pantalla disponible para Tkinter")
    root.withdraw()
    application = ui.App(root)
    yield application
    root.destroy()


@pytest.fixture
def silent_dialogs(monkeypatch):
    """Evita ventanas emergentes y registra los mensajes mostrados."""
    shown = []
    monkeypatch.setattr(ui, "showinfo", lambda **kw: shown.append(kw["message"]))
    monkeypatch.setattr(ui, "showerror", lambda **kw: shown.append(kw["message"]))
    return shown


def _fake_prediction() -> integrator.Prediction:
    return integrator.Prediction("viral", 91.234, np.zeros((512, 512, 3), dtype=np.uint8))


def test_load_image_enables_predict_button(app, monkeypatch):
    # Arrange
    monkeypatch.setattr(ui.filedialog, "askopenfilename", lambda **kw: "torax.dcm")
    monkeypatch.setattr(
        integrator,
        "load_image",
        lambda path: (np.zeros((300, 300, 3), np.uint8), Image.new("L", (300, 300))),
    )
    # Act
    app.load_img_file()
    # Assert
    assert str(app.button1["state"]) != "disabled"
    assert app.array is not None


def test_cancelled_file_dialog_keeps_predict_disabled(app, monkeypatch):
    # Arrange
    monkeypatch.setattr(ui.filedialog, "askopenfilename", lambda **kw: "")
    # Act
    app.load_img_file()
    # Assert
    assert str(app.button1["state"]) == "disabled"
    assert app.array is None


def test_run_model_shows_label_and_probability(app, monkeypatch):
    # Arrange
    app.array = np.zeros((300, 300, 3), np.uint8)
    monkeypatch.setattr(integrator, "predict", lambda array: _fake_prediction())
    # Act
    app.run_model()
    # Assert
    assert app.text2.get("1.0", "end").strip() == "viral"
    assert app.text3.get("1.0", "end").strip() == "91.23%"


def test_save_without_prediction_does_not_write(app, monkeypatch, silent_dialogs):
    # Arrange
    calls = []
    monkeypatch.setattr(integrator, "save_result_csv", lambda *a, **k: calls.append(a))
    # Act
    app.save_results_csv()
    # Assert
    assert calls == []
    assert silent_dialogs == ["Primero ejecute una predicción."]


def test_delete_clears_results_when_confirmed(app, monkeypatch, silent_dialogs):
    # Arrange
    app.prediction = _fake_prediction()
    app.text2.insert("end", "viral")
    app.patient_id.set("123456")
    monkeypatch.setattr(ui, "askokcancel", lambda **kw: True)
    # Act
    app.delete()
    # Assert
    assert app.prediction is None
    assert app.text2.get("1.0", "end").strip() == ""
    assert app.patient_id.get() == ""
    assert str(app.button1["state"]) == "disabled"
