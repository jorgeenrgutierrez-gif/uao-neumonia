import numpy as np

from src.preprocess_img import preprocess


def test_preprocess_returns_batch_tensor_shape():
    # Arrange
    image = np.random.default_rng(0).integers(0, 255, (700, 900, 3), dtype=np.uint8)
    # Act
    batch = preprocess(image)
    # Assert
    assert batch.shape == (1, 512, 512, 1)


def test_preprocess_normalizes_between_zero_and_one():
    # Arrange
    image = np.full((600, 600, 3), 255, dtype=np.uint8)
    # Act
    batch = preprocess(image)
    # Assert
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


def test_preprocess_accepts_grayscale_input():
    # Arrange
    image = np.zeros((256, 256), dtype=np.uint8)
    # Act
    batch = preprocess(image)
    # Assert
    assert batch.shape == (1, 512, 512, 1)


def test_preprocess_returns_float_dtype():
    # Arrange
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Act
    batch = preprocess(image)
    # Assert
    assert np.issubdtype(batch.dtype, np.floating)
