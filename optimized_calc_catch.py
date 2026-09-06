"""Opt-in CalcCatch runner with equivalent, faster threshold counting.

Accepts the same arguments as calc_catch.py. The baseline module is left intact
so benchmarks can compare both implementations in the same environment.
"""
import numpy as np

import calc_catch


def build_threshold_mask(stddev_matrix, number_of_steps, consistency_check):
    std_max = float(np.max(stddev_matrix))
    std_min = float(np.min(stddev_matrix))
    centers = np.linspace(std_max, std_min, number_of_steps)
    half_width = 0.1 * (std_max - std_min) / 2.0
    starts = centers - half_width
    ends = centers + half_width
    # Inclusive interval counts from sorted boundaries preserve rounding
    # at exact endpoints, including coincident windows on constant images.
    counts = (
        np.searchsorted(starts[::-1], stddev_matrix, side='right')
        - np.searchsorted(ends[::-1], stddev_matrix, side='left')
    )
    mask = counts >= int(np.ceil(consistency_check * number_of_steps))
    return mask, starts, ends


if __name__ == '__main__':
    calc_catch._build_threshold_mask = build_threshold_mask
    calc_catch.main()
