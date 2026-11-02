import pytest

from src import load_model


@pytest.fixture(autouse=True)
def clear_cache():
    load_model.load_model.cache_clear()
    yield
    load_model.load_model.cache_clear()


def test_missing_model_raises_clear_error(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setenv(load_model.MODEL_ENV_VAR, str(tmp_path / "no_existe.h5"))
    # Act / Assert
    with pytest.raises(FileNotFoundError, match="NEUMONIA_MODEL_PATH"):
        load_model.load_model()


def test_model_is_loaded_only_once(tmp_path, monkeypatch):
    # Arrange
    fake_file = tmp_path / "modelo.h5"
    fake_file.write_bytes(b"x")
    monkeypatch.setenv(load_model.MODEL_ENV_VAR, str(fake_file))
    calls = []
    import keras

    monkeypatch.setattr(keras.models, "load_model", lambda *a, **k: calls.append(a) or object())
    # Act
    first = load_model.load_model()
    second = load_model.load_model()
    # Assert
    assert first is second
    assert len(calls) == 1
