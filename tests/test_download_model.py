"""Pruebas de la descarga del modelo, sin red: se usa una URL file:// local."""

import hashlib

import pytest

from scripts import download_model as dm


@pytest.fixture
def fake_release(tmp_path):
    """Archivo que simula el .h5 publicado en el Release y su huella SHA-256."""
    source = tmp_path / "release" / "conv_MLP_84.h5"
    source.parent.mkdir()
    content = b"\x89HDF\r\n\x1a\n" + b"pesos" * 1000
    source.write_bytes(content)
    return source.as_uri(), hashlib.sha256(content).hexdigest(), content


def test_download_model_saves_verified_file(tmp_path, fake_release):
    # Arrange
    url, sha, content = fake_release
    dest = tmp_path / "models" / "conv_MLP_84.h5"
    # Act
    result = dm.download_model(url, dest, sha)
    # Assert
    assert result == dest
    assert dest.read_bytes() == content
    assert not dest.with_suffix(".h5.part").exists()


def test_download_model_rejects_wrong_checksum(tmp_path, fake_release):
    # Arrange
    url, _, _ = fake_release
    dest = tmp_path / "models" / "conv_MLP_84.h5"
    # Act / Assert
    with pytest.raises(ValueError, match="SHA-256"):
        dm.download_model(url, dest, "0" * 64)
    assert not dest.exists()
