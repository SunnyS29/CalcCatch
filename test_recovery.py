import unittest

import numpy as np
from skimage.measure import regionprops

from recovery_experiment import split_activity_peaks


class RecoveryTests(unittest.TestCase):
    @staticmethod
    def image_and_region():
        yy, xx = np.mgrid[:40, :60]
        mask = (xx-30)**2/25**2 + (yy-20)**2/12**2 <= 1
        left = np.exp(-((xx-19)**2+(yy-20)**2)/24.)
        right = np.exp(-((xx-41)**2+(yy-20)**2)/24.)
        return left, right, mask, regionprops(mask.astype(int))

    def test_two_peaks_split_without_losing_or_adding_pixels(self):
        a, b, mask, regions = self.image_and_region()
        result, count = split_activity_peaks(regions, 100*(a+b),
            min_distance=12, prominence_fraction=.2)
        self.assertEqual(count, 1)
        self.assertEqual(len(regionprops(result)), 2)
        np.testing.assert_array_equal(result > 0, mask)
        centers = sorted(r.centroid[1] for r in regionprops(result))
        self.assertLess(centers[0], 30)
        self.assertGreater(centers[1], 30)

    def test_single_peak_and_flat_region_stay_whole(self):
        a, _, mask, regions = self.image_and_region()
        for image in (a, np.ones(mask.shape), np.zeros(mask.shape)):
            result, count = split_activity_peaks(regions, image, prominence_fraction=.2)
            self.assertEqual(count, 0)
            self.assertEqual(len(regionprops(result)), 1)
            np.testing.assert_array_equal(result > 0, mask)

    def test_shallow_secondary_peak_is_suppressed(self):
        a, b, mask, regions = self.image_and_region()
        image = 10 + a + .6*b
        _, before = split_activity_peaks(regions, image, min_distance=12)
        result, after = split_activity_peaks(regions, image,
            min_distance=12, prominence_fraction=.2)
        self.assertEqual(before, 1)
        self.assertEqual(after, 0)
        np.testing.assert_array_equal(result > 0, mask)

    def test_reject_split_if_a_child_would_be_too_small(self):
        a, b, mask, regions = self.image_and_region()
        result, count = split_activity_peaks(regions, a+b, min_area=int(mask.sum()))
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(result > 0, mask)

    def test_empty_input(self):
        result, count = split_activity_peaks([], np.zeros((8, 9)))
        self.assertEqual(count, 0)
        self.assertFalse(result.any())

    def test_deterministic_labels_and_border_regions(self):
        a, b, mask, regions = self.image_and_region()
        image = (a+b)[8:33, 5:56]
        border_mask = mask[8:33, 5:56]
        border_regions = regionprops(border_mask.astype(int))
        x, n = split_activity_peaks(border_regions, image, min_distance=12)
        y, m = split_activity_peaks(border_regions, image, min_distance=12)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x > 0, border_mask)
        self.assertEqual(n, m)

    def test_invalid_settings(self):
        for kwargs in ({'min_distance': 0}, {'min_area': 0},
                       {'prominence_fraction': -1}, {'prominence_fraction': 1}):
            with self.assertRaises(ValueError):
                split_activity_peaks([], np.zeros((8, 9)), **kwargs)


if __name__ == '__main__':
    unittest.main()
