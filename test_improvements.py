import unittest

import numpy as np

import calc_catch
from optimized_calc_catch import build_threshold_mask
from benchmark_improvements import score


class ThresholdParityTests(unittest.TestCase):
    def check_parity(self, image, steps, consistency):
        expected = calc_catch._build_threshold_mask(image, steps, consistency)
        actual = build_threshold_mask(image, steps, consistency)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)

    def test_random_and_constant_maps(self):
        rng = np.random.default_rng(20260906)
        for image in (rng.random((31, 27)), rng.integers(0, 256, (31, 27)),
                      np.zeros((7, 9)), np.full((7, 9), 42.0)):
            for steps in (1, 2, 17, 5000):
                for consistency in (0, .05, .075, .1, 1):
                    with self.subTest(steps=steps, consistency=consistency):
                        self.check_parity(image, steps, consistency)

    def test_boundaries_and_adjacent_floats(self):
        for scale in (1e-9, 1.0, 1e9):
            for steps in (7, 100, 5000):
                _, starts, ends = calc_catch._build_threshold_mask(
                    np.array([[0., scale]]), steps, .075)
                boundaries = np.concatenate((starts, ends))
                boundaries = boundaries[(boundaries >= 0) & (boundaries <= scale)]
                image = np.concatenate(([0, scale], boundaries,
                    np.nextafter(boundaries, -np.inf),
                    np.nextafter(boundaries, np.inf))).reshape(1, -1)
                for consistency in (.05, .075, .1):
                    self.check_parity(image, steps, consistency)


class MatchingTests(unittest.TestCase):
    def test_optimal_match_count_is_order_independent(self):
        xy = np.array([[0., 0.], [18., 0.]])
        landmarks = np.array([[8., 0.], [-10., 0.]])
        a, b = score(xy, landmarks), score(xy, landmarks[::-1])
        self.assertEqual((a['tp'], b['tp']), (1, 2))
        self.assertEqual((a['optimal_tp'], b['optimal_tp']), (2, 2))

    def test_extra_detections_reduce_f1(self):
        landmarks = np.array([[0., 0.]])
        self.assertEqual(score(landmarks, landmarks)['optimal_f1'], 1.)
        extra = np.array([[0., 0.], [100., 100.]])
        self.assertAlmostEqual(score(extra, landmarks)['optimal_f1'], 2/3)


if __name__ == '__main__':
    unittest.main()
