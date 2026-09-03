import unittest

import numpy as np

from metrics import cal_pro_score, cal_pro_score_fast


class AUPROTests(unittest.TestCase):
    def test_fast_aupro_matches_official_threshold_loop(self):
        generator = np.random.default_rng(19)
        for _ in range(2):
            maps = generator.random((3, 18, 18), dtype=np.float64)
            masks = np.zeros((3, 18, 18), dtype=np.uint8)
            masks[0, 2:8, 3:9] = 1
            masks[1, 10:16, 1:6] = 1
            masks[2, 4:9, 10:16] = 1
            masks[2, 12:16, 3:7] = 1
            self.assertAlmostEqual(
                cal_pro_score(masks, maps), cal_pro_score_fast(masks, maps), places=14
            )


if __name__ == "__main__":
    unittest.main()
