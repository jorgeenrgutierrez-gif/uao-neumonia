"""Verificación dura de la rúbrica: la UI no conoce el modelo."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
FORBIDDEN_IN_UI = {"tensorflow", "keras", "load_model", "src.load_model"}


def _imported_names(module: Path) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{a.name}" for a in node.names)
            names.update(alias.name for alias in node.names)
    return names


def test_ui_does_not_import_model_or_tensorflow():
    # Arrange
    ui = SRC / "ui.py"
    # Act
    imports = _imported_names(ui)
    # Assert
    assert not imports & FORBIDDEN_IN_UI
    assert not any(name.startswith(("tensorflow", "keras")) for name in imports)


def test_ui_only_imports_integrator_from_project():
    # Arrange
    ui = SRC / "ui.py"
    # Act
    project_imports = {n for n in _imported_names(ui) if n.startswith("src.")}
    # Assert
    assert project_imports == {"src.integrator"}


def test_six_modules_exist():
    # Arrange
    expected = {"load_model", "read_img", "preprocess_img", "grad_cam", "integrator", "ui"}
    # Act
    present = {p.stem for p in SRC.glob("*.py")} - {"__init__"}
    # Assert
    assert present == expected
