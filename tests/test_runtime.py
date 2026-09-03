import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from remap.runtime import gaussian_filter2d


def test_tensor_gaussian_matches_scipy_default_filter():
    generator = np.random.default_rng(41)
    values = generator.normal(size=(2, 3, 47, 53)).astype(np.float32)
    expected = np.stack([
        np.stack([gaussian_filter(channel, sigma=4) for channel in image])
        for image in values
    ])
    observed = gaussian_filter2d(torch.from_numpy(values), sigma=4).numpy()

    np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=2e-7)
