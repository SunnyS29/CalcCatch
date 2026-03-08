from __future__ import annotations

"""CalcCatch Python parity implementation.

This module mirrors the MATLAB ROI detection path used in the thesis:
1. pixel-wise standard deviation activity mapping
2. sliding-window threshold consensus
3. negative distance transform
4. h-minima transformation
5. watershed refinement and connected-component ROI extraction
6. contingency table matching against manual centroids

The exported ROI time series are intended for downstream transfer entropy
and graph-theoretical connectivity analyses.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import disk, reconstruction
from skimage.segmentation import watershed


@dataclass
class CalcCatchConfig:
    tiff_stack_path: str = 'data/example_stack.tif'
    original_coordinates_path: str = 'data/example_manual_roi_coordinates.xlsx'
    output_excel_file: str = 'results/algorithm_rois_python.xlsx'
    number_of_steps: int = 5000
    consistency_check: float = 0.075
    consistency_pixel_frac: float = 0.70
    min_roi_area: int = 10
    h: float = 0.5
    frame_rate: float = 2.0
    matching_tolerance: float = 10.0
    write_excel: bool = True
    matlab_strict_mode: bool = True


@dataclass
class CalcCatchResult:
    detected_roi_count: int
    matched_count: int
    matching_indicator: float
    true_positive: int
    false_negative: int
    false_positive: int
    output_excel_file: str | None


def _normalize_header(value: Any) -> str:
    return ' '.join(str(value).strip().lower().replace('_', ' ').split())


def _to_hwf_stack(stack_path: Path) -> np.ndarray:
    """Load TIFF data and enforce MATLAB-compatible (height, width, frames) order."""
    with tifffile.TiffFile(stack_path) as tf:
        num_pages = len(tf.pages)
        arr = np.asarray(tf.asarray())

    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim == 3 and arr.shape[0] == num_pages and arr.shape[-1] != num_pages:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim != 3:
        raise ValueError(f'Unsupported TIFF dimensions: {arr.shape}')

    return arr.astype(np.uint16, copy=False)


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    """Resolve flexible column naming while preserving thesis spreadsheet compatibility."""
    normalized_columns = [(_normalize_header(col), str(col)) for col in df.columns]

    for alias in aliases:
        key = _normalize_header(alias)
        for normalized, original in normalized_columns:
            if normalized == key or normalized.startswith(f'{key}.'):
                return original

    raise KeyError(f'Could not find any of {aliases} in columns {list(df.columns)}')


def _load_landmark_centroids(path: Path) -> np.ndarray:
    """Load manual landmark centroids from the `xy coord` sheet."""
    # Some exports store metadata in row 1 and actual headers in row 2.
    for header_row in (0, 1):
        df = pd.read_excel(path, sheet_name='xy coord', header=header_row)
        try:
            x_col = _find_column(df, ('x centre', 'x center', 'x'))
            y_col = _find_column(df, ('y centre', 'y center', 'y'))
        except KeyError:
            continue

        centroids = (
            df[[x_col, y_col]]
            .apply(pd.to_numeric, errors='coerce')
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        if centroids.size:
            return centroids

    raise KeyError('Could not parse x/y centroid columns from sheet "xy coord".')


def _imhmin_like(image: np.ndarray, h: float) -> np.ndarray:
    """MATLAB-like h-minima transformation via reconstruction by erosion.

    In CalcCatch, h=0.5 suppresses shallow local minima and helps prevent
    over-segmentation in dense ENS neuron clusters.
    """
    if h <= 0:
        return image
    return reconstruction(image + h, image, method='erosion')


def _build_threshold_mask(
    stddev_matrix: np.ndarray,
    number_of_steps: int,
    consistency_check: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    std_max = float(np.max(stddev_matrix))
    std_min = float(np.min(stddev_matrix))
    stddev_range = np.linspace(std_max, std_min, number_of_steps)
    window_overlap = 0.1 * (std_max - std_min)
    window_starts = stddev_range - window_overlap / 2.0
    window_ends = stddev_range + window_overlap / 2.0

    # Sliding-window consensus mask with 10% overlap in the standard-deviation domain.
    pixel_count = np.zeros(stddev_matrix.shape, dtype=np.uint32)
    for start, end in zip(window_starts, window_ends):
        pixel_count += ((stddev_matrix >= start) & (stddev_matrix <= end)).astype(np.uint32)

    minimum_windows = int(np.ceil(consistency_check * number_of_steps))
    threshold_mask = pixel_count >= minimum_windows
    return threshold_mask, window_starts, window_ends


def _build_watershed_regions(threshold_mask: np.ndarray, h: float, strict_mode: bool) -> np.ndarray:
    """Generate watershed-separated ROI regions using MATLAB-compatible semantics."""
    # Negative distance transform mirrors: -bwdist(~threshold_mask)
    dist_trans = -distance_transform_edt(threshold_mask)

    if strict_mode:
        # MATLAB equivalent:
        # dist_trans = -bwdist(~threshold_mask);
        # dist_trans(~threshold_mask) = -Inf;
        # new_boundaries = watershed(imhmin(dist_trans, h));
        dist_trans[~threshold_mask] = -1e12  # finite surrogate for -Inf
        dist_trans_mod = _imhmin_like(dist_trans, h)
        labels = watershed(dist_trans_mod, connectivity=2, watershed_line=True)
        ridges = labels == 0
        ridges = binary_dilation(ridges, structure=disk(1))
        return ~ridges

    # Conservative variant that constrains watershed to active mask.
    floor_val = float(np.min(dist_trans[threshold_mask])) - 1.0 if np.any(threshold_mask) else -1.0
    dist_trans[~threshold_mask] = floor_val
    dist_trans_mod = _imhmin_like(dist_trans, h)
    labels = watershed(dist_trans_mod, mask=threshold_mask, connectivity=2, watershed_line=True)
    ridges = labels == 0
    ridges = binary_dilation(ridges, structure=disk(1))
    return threshold_mask & (~ridges)


def _matching_with_candidate_resolution(
    detected_xy: np.ndarray,
    landmark_xy: np.ndarray,
    tolerance: float,
) -> tuple[int, int, int, float]:
    """Replicate contingency table matching with nearest-candidate resolution."""
    num_landmark = landmark_xy.shape[0]
    num_detected = detected_xy.shape[0]

    if num_landmark == 0:
        return 0, 0, num_detected, float('nan')

    dx = np.abs(detected_xy[None, :, 0] - landmark_xy[:, None, 0])
    dy = np.abs(detected_xy[None, :, 1] - landmark_xy[:, None, 1])
    match_matrix = (dx <= tolerance) & (dy <= tolerance)

    assigned_detection = np.full(num_landmark, -1, dtype=np.int64)
    used = np.zeros(num_detected, dtype=bool)

    for i in range(num_landmark):
        candidates = np.where(match_matrix[i] & ~used)[0]
        if candidates.size == 0:
            continue
        distances = np.sqrt(np.sum((detected_xy[candidates] - landmark_xy[i]) ** 2, axis=1))
        best = candidates[np.argmin(distances)]
        assigned_detection[i] = best
        used[best] = True

    matched_count = int(np.sum(assigned_detection >= 0))
    tp = matched_count
    fn = int(num_landmark - matched_count)
    fp = int(num_detected - matched_count)
    indicator = matched_count / num_landmark
    return tp, fn, fp, indicator


def _compute_roi_consistency(
    roi_std_values: list[np.ndarray],
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    consistency_pixel_frac: float,
) -> np.ndarray:
    """Compute per-ROI consistency scores using the same threshold-window rule as MATLAB."""
    scores = np.zeros(len(roi_std_values), dtype=np.int64)
    for i, roi_std in enumerate(roi_std_values):
        sorted_std = np.sort(roi_std)
        threshold_pixels = int(np.ceil(consistency_pixel_frac * sorted_std.size))
        left = np.searchsorted(sorted_std, window_starts, side='left')
        right = np.searchsorted(sorted_std, window_ends, side='right')
        active_counts = right - left
        scores[i] = int(np.sum(active_counts >= threshold_pixels))
    return scores


def _write_excel_output(
    output_path: Path,
    roi_table_rows: list[dict[str, float]],
    roi_time_series: np.ndarray,
    time_vector: np.ndarray,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roi_df = pd.DataFrame(roi_table_rows)
    time_df = pd.DataFrame(roi_time_series, columns=[f'ROI_{i + 1}' for i in range(roi_time_series.shape[1])])
    time_df.insert(0, 'Time', time_vector)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        roi_df.to_excel(writer, sheet_name='ROI_Data', index=False)
        time_df.to_excel(writer, sheet_name='Time_Series', index=False)

    return str(output_path)


def run_calc_catch(config: CalcCatchConfig) -> CalcCatchResult:
    """Run the MATLAB-parity ROI pipeline and return summary metrics."""
    stack = _to_hwf_stack(Path(config.tiff_stack_path))
    landmark_centroids = _load_landmark_centroids(Path(config.original_coordinates_path))

    img_height, img_width, num_frames = stack.shape
    time_vector = np.arange(num_frames, dtype=np.float64) / config.frame_rate

    stack_float = stack.astype(np.float64)
    # Pixel-wise standard deviation over time replicates MATLAB std(...,0,3).
    stddev_matrix = np.std(stack_float, axis=2, ddof=0)

    threshold_mask, window_starts, window_ends = _build_threshold_mask(
        stddev_matrix,
        config.number_of_steps,
        config.consistency_check,
    )

    # h-minima transformation with h=0.5 is the key anti-fragmentation control.
    watershed_regions = _build_watershed_regions(threshold_mask, float(config.h), config.matlab_strict_mode)

    cc_labels = label(watershed_regions, connectivity=2)
    regions = [region for region in regionprops(cc_labels) if region.area >= config.min_roi_area]
    detected_roi_count = len(regions)

    roi_table_rows: list[dict[str, float]] = []
    roi_time_series = np.zeros((num_frames, detected_roi_count), dtype=np.float64)
    detected_centroids_xy = np.zeros((detected_roi_count, 2), dtype=np.float64)
    roi_std_values: list[np.ndarray] = []

    for i, region in enumerate(regions):
        rr = region.coords[:, 0]
        cc = region.coords[:, 1]
        roi_cube = stack_float[rr, cc, :]

        y_centre, x_centre = region.centroid
        detected_centroids_xy[i] = [x_centre, y_centre]

        roi_table_rows.append(
            {
                'x_Centre': float(x_centre),
                'y_Centre': float(y_centre),
                'Mean_Intensity': float(np.mean(roi_cube)),
                'Std_Intensity': float(np.std(roi_cube, ddof=0)),
            }
        )

        roi_time_series[:, i] = np.mean(roi_cube, axis=0)
        roi_std_values.append(stddev_matrix[rr, cc])

    # Retained for MATLAB-parity staging even though it is not exported directly.
    _ = _compute_roi_consistency(
        roi_std_values,
        window_starts,
        window_ends,
        config.consistency_pixel_frac,
    )

    # Contingency table matching against manual centroids.
    tp, fn, fp, indicator = _matching_with_candidate_resolution(
        detected_centroids_xy,
        landmark_centroids,
        config.matching_tolerance,
    )

    output_excel_file = None
    if config.write_excel:
        output_excel_file = _write_excel_output(
            Path(config.output_excel_file),
            roi_table_rows,
            roi_time_series,
            time_vector,
        )

    return CalcCatchResult(
        detected_roi_count=detected_roi_count,
        matched_count=tp,
        matching_indicator=float(indicator),
        true_positive=tp,
        false_negative=fn,
        false_positive=fp,
        output_excel_file=output_excel_file,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CalcCatch Python parity baseline.')
    parser.add_argument('--tiff', default='data/example_stack.tif', help='Path to TIFF stack.')
    parser.add_argument(
        '--coords',
        default='data/example_manual_roi_coordinates.xlsx',
        help='Path to manual coordinate Excel file (sheet: xy coord).',
    )
    parser.add_argument('--output', default='results/algorithm_rois_python.xlsx', help='Output Excel file.')
    parser.add_argument('--number-of-steps', type=int, default=5000)
    parser.add_argument('--consistency-check', type=float, default=0.075)
    parser.add_argument('--consistency-pixel-frac', type=float, default=0.70)
    parser.add_argument('--min-roi-area', type=int, default=10)
    parser.add_argument('--h', type=float, default=0.5)
    parser.add_argument('--frame-rate', type=float, default=2.0)
    parser.add_argument('--matching-tolerance', type=float, default=10.0)
    parser.add_argument('--metrics-json', default='', help='Optional JSON output path for metrics.')
    parser.add_argument('--no-excel', action='store_true', help='Skip Excel output writing.')
    parser.add_argument('--expected-matching-indicator', type=float, default=None)
    parser.add_argument('--indicator-tolerance', type=float, default=0.02)
    parser.add_argument(
        '--no-matlab-strict-mode',
        action='store_true',
        help='Disable strict MATLAB watershed semantics.',
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    config = CalcCatchConfig(
        tiff_stack_path=args.tiff,
        original_coordinates_path=args.coords,
        output_excel_file=args.output,
        number_of_steps=args.number_of_steps,
        consistency_check=args.consistency_check,
        consistency_pixel_frac=args.consistency_pixel_frac,
        min_roi_area=args.min_roi_area,
        h=args.h,
        frame_rate=args.frame_rate,
        matching_tolerance=args.matching_tolerance,
        write_excel=not args.no_excel,
        matlab_strict_mode=not args.no_matlab_strict_mode,
    )

    result = run_calc_catch(config)
    payload: dict[str, Any] = {
        'config': asdict(config),
        'result': asdict(result),
    }

    print(json.dumps(payload, indent=2))

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    if args.expected_matching_indicator is not None and not np.isnan(result.matching_indicator):
        delta = abs(result.matching_indicator - args.expected_matching_indicator)
        if delta > args.indicator_tolerance:
            raise SystemExit(
                f'Matching indicator parity check failed: got {result.matching_indicator:.4f}, '
                f'expected {args.expected_matching_indicator:.4f} +/- {args.indicator_tolerance:.4f}'
            )


if __name__ == '__main__':
    main()
