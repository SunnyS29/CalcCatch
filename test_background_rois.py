import unittest
from unittest.mock import patch

import numpy as np
from skimage.measure import label, regionprops

import calc_catch as c


class BackgroundRegionTests(unittest.TestCase):
    def test_background_basin_removed_and_border_neuron_kept(self):
        labels = np.zeros((12, 12), dtype=int)
        labels[:2, :2] = 1
        labels[4:12, 4:12] = 2
        active = np.zeros_like(labels, dtype=bool)
        active[:2, :2] = True
        active[6, 6] = True
        regions = regionprops(labels)
        kept = c._filter_background_regions(regions, labels, active)
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0], regions[0])

    def test_half_active_is_retained(self):
        labels = np.ones((2, 2), dtype=int)
        active = np.array([[True, True], [False, False]])
        self.assertEqual(len(c._filter_background_regions(
            regionprops(labels), labels, active)), 1)
        active[0, 1] = False
        self.assertEqual(c._filter_background_regions(
            regionprops(labels), labels, active), [])

    def test_empty_regions_and_noncontiguous_labels(self):
        labels = np.array([[0, 7], [0, 7]])
        active = labels > 0
        self.assertEqual(c._filter_background_regions([], labels, active), [])
        kept = c._filter_background_regions(regionprops(labels), labels, active)
        self.assertEqual([r.label for r in kept], [7])

    def test_random_masks_match_direct_pixel_counting(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            labels = label(rng.random((24, 27)) > .65)
            active = rng.random(labels.shape) > .5
            regions = regionprops(labels)
            expected = [r.label for r in regions
                        if active[r.coords[:, 0], r.coords[:, 1]].mean() >= .5]
            actual = c._filter_background_regions(regions, labels, active)
            self.assertEqual([r.label for r in actual], expected)

    def test_cli_and_config_default_to_baseline(self):
        self.assertFalse(c.CalcCatchConfig().exclude_background_rois)
        self.assertFalse(c._build_arg_parser().parse_args([]).exclude_background_rois)
        self.assertTrue(c._build_arg_parser().parse_args(
            ['--exclude-background-rois']).exclude_background_rois)

    def test_pipeline_preserves_retained_traces_and_handles_no_rois(self):
        rng = np.random.default_rng(8)
        stack = rng.integers(0, 65536, (12, 12, 6), dtype=np.uint16)
        watershed_regions = np.zeros((12, 12), dtype=bool)
        watershed_regions[:2, :2] = True
        watershed_regions[4:12, 4:12] = True
        active = np.zeros((12, 12), dtype=bool)
        active[:2, :2] = True
        exports = []

        def capture(path, rows, traces, time):
            exports.append((rows, traces.copy(), time.copy()))
            return str(path)

        with patch.object(c, '_to_hwf_stack', return_value=stack), \
             patch.object(c, '_load_landmark_centroids', return_value=np.array([[.5, .5]])), \
             patch.object(c, '_build_threshold_mask', return_value=(active, np.array([0]), np.array([1]))), \
             patch.object(c, '_build_watershed_regions', return_value=watershed_regions), \
             patch.object(c, '_write_excel_output', side_effect=capture):
            baseline = c.run_calc_catch(c.CalcCatchConfig(min_roi_area=1))
            filtered = c.run_calc_catch(c.CalcCatchConfig(
                min_roi_area=1, exclude_background_rois=True))
            self.assertEqual((baseline.detected_roi_count, filtered.detected_roi_count), (2, 1))
            self.assertEqual(filtered.excluded_background_roi_count, 1)
            np.testing.assert_array_equal(exports[0][1][:, :1], exports[1][1])
            self.assertEqual(exports[0][0][:1], exports[1][0])
            np.testing.assert_array_equal(exports[0][2], exports[1][2])
            active[:] = False
            empty = c.run_calc_catch(c.CalcCatchConfig(
                min_roi_area=1, exclude_background_rois=True))
            self.assertEqual(empty.detected_roi_count, 0)
            self.assertEqual(empty.excluded_background_roi_count, 2)
            self.assertEqual(exports[-1][1].shape, (6, 0))


if __name__ == '__main__':
    unittest.main()
