import numpy as np

from src import grad_cam
from src.preprocess_img import preprocess


def test_heatmap_is_normalized(tiny_model):
    # Arrange
    array = np.random.default_rng(1).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    batch = preprocess(array)
    # Act
    heatmap = grad_cam.compute_heatmap(tiny_model, batch, class_index=0)
    # Assert
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_grad_cam_returns_rgb_512_image(tiny_model):
    # Arrange
    array = np.random.default_rng(2).integers(0, 255, (640, 480, 3), dtype=np.uint8)
    batch = preprocess(array)
    # Act
    image = grad_cam.grad_cam(tiny_model, batch, array, class_index=1)
    # Assert
    assert image.shape == (512, 512, 3)
    assert image.dtype == np.uint8


def test_overlay_handles_all_zero_heatmap():
    # Arrange
    heatmap = np.zeros((31, 31))
    array = np.zeros((512, 512, 3), dtype=np.uint8)
    # Act
    image = grad_cam.overlay(heatmap, array)
    # Assert
    assert not np.isnan(image).any()
