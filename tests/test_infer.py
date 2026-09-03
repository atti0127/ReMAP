import numpy as np
from PIL import Image

from remap.infer import _normalize, _save_visualizations


def test_normalize_handles_constant_map():
    observed = _normalize(np.full((3, 4), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(observed, np.zeros((3, 4), dtype=np.float32))


def test_single_image_visualizations_are_written(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 6), color=(100, 120, 140)).save(image_path)
    scores = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)

    heatmap_path, overlay_path = _save_visualizations(
        image_path, scores, tmp_path, alpha=0.5
    )

    assert Image.open(heatmap_path).size == (5, 4)
    assert Image.open(overlay_path).size == (5, 4)

