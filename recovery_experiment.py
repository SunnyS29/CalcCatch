"""Experimental activity-peak splitting of existing foreground ROIs.

This module does not alter the normal CLI or detection defaults. All candidate
seeds come from the activity image, never from reference annotations.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def split_activity_peaks(regions, stddev, min_distance=10, min_area=10, prominence_fraction=0.):
    if min_distance < 1 or min_area < 1:
        raise ValueError('min_distance and min_area must be positive')
    if not 0 <= prominence_fraction < 1:
        raise ValueError('prominence_fraction must be in [0, 1)')
    smooth = gaussian_filter(np.asarray(stddev, dtype=float), sigma=1.)
    output = np.zeros(stddev.shape, dtype=np.int32)
    next_label = 1
    split_parents = 0
    for region in regions:
        mask = region.image
        local = np.where(mask, smooth[region.slice], 0.)
        seeds = peak_local_max(local, min_distance=min_distance,
            threshold_rel=.3, exclude_border=False, labels=mask.astype(np.uint8), p_norm=2)
        if prominence_fraction > 0 and len(seeds):
            prominent = h_maxima(local, prominence_fraction * float(local.max()))
            seeds = np.array([p for p in seeds if prominent[tuple(p)]], dtype=int).reshape(-1, 2)
        children = None
        if len(seeds) >= 2:
            seeds = sorted(seeds, key=lambda p: (-local[tuple(p)], int(p[0]), int(p[1])))
            markers = np.zeros(mask.shape, dtype=np.int32)
            for index, point in enumerate(seeds, 1):
                markers[tuple(point)] = index
            candidate = watershed(-local, markers, mask=mask, connectivity=2)
            parts = regionprops(candidate)
            # Reject the whole proposed split if any child is too small.
            if len(parts) >= 2 and all(p.area >= min_area for p in parts):
                children = candidate
                split_parents += 1
        destination = output[region.slice]
        if children is None:
            destination[mask] = next_label
            next_label += 1
        else:
            for child in regionprops(children):
                destination[children == child.label] = next_label
                next_label += 1
    return output, split_parents
