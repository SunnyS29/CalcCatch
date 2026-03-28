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
from time import perf_counter
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
    min_roi_consistency_ratio: float | None = None
    merge_centroid_distance_px: float | None = None
    merge_trace_correlation: float | None = None
    merge_bbox_gap_px: float | None = None
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
    consistency_filter_summary: dict[str, float | int]
    merge_summary: dict[str, float | int]
    # Profiling note: stage timings were added during optimization review so
    # we can identify bottlenecks without changing ROI detection semantics.
    stage_timings_sec: dict[str, float]


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

    # Optimization note: we keep only per-pixel threshold hit counts rather
    # than a full [height, width, step] logical cube. This preserves the
    # original sliding-window consensus rule with lower memory cost.
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
        # Optimization note: sorted ROI standard-deviation values let us
        # count active pixels per window with searchsorted rather than
        # re-materializing every threshold mask.
        sorted_std = np.sort(roi_std)
        threshold_pixels = int(np.ceil(consistency_pixel_frac * sorted_std.size))
        left = np.searchsorted(sorted_std, window_starts, side='left')
        right = np.searchsorted(sorted_std, window_ends, side='right')
        active_counts = right - left
        scores[i] = int(np.sum(active_counts >= threshold_pixels))
    return scores


def _apply_consistency_filter(
    consistency_scores: np.ndarray,
    number_of_steps: int,
    min_roi_consistency_ratio: float | None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Optionally reject ROIs with weak support across the sliding-window sweep."""
    if min_roi_consistency_ratio is None:
        keep_mask = np.ones(consistency_scores.shape[0], dtype=bool)
        summary = {
            'pre_consistency_filter_regions': int(consistency_scores.shape[0]),
            'post_consistency_filter_regions': int(consistency_scores.shape[0]),
            'min_roi_consistency_ratio': -1.0,
            'minimum_count': -1,
        }
        return keep_mask, summary

    minimum_count = int(np.ceil(min_roi_consistency_ratio * number_of_steps))
    keep_mask = consistency_scores >= minimum_count
    summary = {
        'pre_consistency_filter_regions': int(consistency_scores.shape[0]),
        'post_consistency_filter_regions': int(np.sum(keep_mask)),
        'min_roi_consistency_ratio': float(min_roi_consistency_ratio),
        'minimum_count': minimum_count,
    }
    return keep_mask, summary


def _bbox_gap(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
    """Return the pixel gap between two bounding boxes along their closest edges."""
    min_row_a, min_col_a, max_row_a, max_col_a = bbox_a
    min_row_b, min_col_b, max_row_b, max_col_b = bbox_b

    row_gap = max(0, max(min_row_b - max_row_a, min_row_a - max_row_b))
    col_gap = max(0, max(min_col_b - max_col_a, min_col_a - max_col_b))
    return float(max(row_gap, col_gap))


def _trace_correlation(trace_a: np.ndarray, trace_b: np.ndarray) -> float:
    """Compute Pearson correlation between two ROI mean traces."""
    std_a = float(np.std(trace_a))
    std_b = float(np.std(trace_b))
    if std_a == 0.0 or std_b == 0.0:
        return -1.0
    return float(np.corrcoef(trace_a, trace_b)[0, 1])


def _apply_merge_filter(
    detected_centroids_xy: np.ndarray,
    roi_time_series: np.ndarray,
    roi_table_rows: list[dict[str, float]],
    roi_areas: np.ndarray,
    roi_bboxes: list[tuple[int, int, int, int]],
    merge_centroid_distance_px: float | None,
    merge_trace_correlation: float | None,
    merge_bbox_gap_px: float | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, float]],
    np.ndarray,
    list[tuple[int, int, int, int]],
    dict[str, float | int],
]:
    """Optionally merge ROI pairs that are both spatially close and temporally redundant."""
    roi_count = detected_centroids_xy.shape[0]
    if (
        merge_centroid_distance_px is None
        or merge_trace_correlation is None
        or merge_bbox_gap_px is None
        or roi_count == 0
    ):
        summary = {
            'pre_merge_regions': int(roi_count),
            'post_merge_regions': int(roi_count),
            'merge_centroid_distance_px': -1.0,
            'merge_trace_correlation': -1.0,
            'merge_bbox_gap_px': -1.0,
            'merge_links': 0,
        }
        return detected_centroids_xy, roi_time_series, roi_table_rows, roi_areas, roi_bboxes, summary

    parent = np.arange(roi_count, dtype=np.int64)
    merge_links = 0

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(idx_a: int, idx_b: int) -> None:
        root_a = find(idx_a)
        root_b = find(idx_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(roi_count):
        for j in range(i + 1, roi_count):
            centroid_distance = float(np.linalg.norm(detected_centroids_xy[i] - detected_centroids_xy[j]))
            if centroid_distance > merge_centroid_distance_px:
                continue
            if _bbox_gap(roi_bboxes[i], roi_bboxes[j]) > merge_bbox_gap_px:
                continue
            if _trace_correlation(roi_time_series[:, i], roi_time_series[:, j]) < merge_trace_correlation:
                continue
            union(i, j)
            merge_links += 1

    groups: dict[int, list[int]] = {}
    for idx in range(roi_count):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    if len(groups) == roi_count:
        summary = {
            'pre_merge_regions': int(roi_count),
            'post_merge_regions': int(roi_count),
            'merge_centroid_distance_px': float(merge_centroid_distance_px),
            'merge_trace_correlation': float(merge_trace_correlation),
            'merge_bbox_gap_px': float(merge_bbox_gap_px),
            'merge_links': int(merge_links),
        }
        return detected_centroids_xy, roi_time_series, roi_table_rows, roi_areas, roi_bboxes, summary

    merged_centroids: list[np.ndarray] = []
    merged_traces: list[np.ndarray] = []
    merged_rows: list[dict[str, float]] = []
    merged_areas: list[float] = []
    merged_bboxes: list[tuple[int, int, int, int]] = []

    for group in groups.values():
        group_areas = roi_areas[group]
        area_sum = float(np.sum(group_areas))
        weights = group_areas / area_sum if area_sum > 0 else np.full(len(group), 1.0 / len(group))

        merged_centroids.append(np.sum(detected_centroids_xy[group] * weights[:, None], axis=0))
        merged_traces.append(np.sum(roi_time_series[:, group] * weights[None, :], axis=1))
        merged_rows.append(
            {
                'x_Centre': float(np.sum(detected_centroids_xy[group, 0] * weights)),
                'y_Centre': float(np.sum(detected_centroids_xy[group, 1] * weights)),
                'Mean_Intensity': float(np.sum([roi_table_rows[idx]['Mean_Intensity'] * weights[k] for k, idx in enumerate(group)])),
                'Std_Intensity': float(np.sum([roi_table_rows[idx]['Std_Intensity'] * weights[k] for k, idx in enumerate(group)])),
            }
        )
        merged_areas.append(area_sum)
        min_row = min(roi_bboxes[idx][0] for idx in group)
        min_col = min(roi_bboxes[idx][1] for idx in group)
        max_row = max(roi_bboxes[idx][2] for idx in group)
        max_col = max(roi_bboxes[idx][3] for idx in group)
        merged_bboxes.append((min_row, min_col, max_row, max_col))

    summary = {
        'pre_merge_regions': int(roi_count),
        'post_merge_regions': int(len(merged_rows)),
        'merge_centroid_distance_px': float(merge_centroid_distance_px),
        'merge_trace_correlation': float(merge_trace_correlation),
        'merge_bbox_gap_px': float(merge_bbox_gap_px),
        'merge_links': int(merge_links),
    }
    return (
        np.asarray(merged_centroids, dtype=np.float64),
        np.column_stack(merged_traces),
        merged_rows,
        np.asarray(merged_areas, dtype=np.float64),
        merged_bboxes,
        summary,
    )


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
    # Profiling note: these timers were added after parity validation so we
    # can measure runtime hotspots without altering detection logic.
    timings: dict[str, float] = {}

    stage_start = perf_counter()
    stack = _to_hwf_stack(Path(config.tiff_stack_path))
    landmark_centroids = _load_landmark_centroids(Path(config.original_coordinates_path))
    timings['load_inputs'] = perf_counter() - stage_start

    img_height, img_width, num_frames = stack.shape
    time_vector = np.arange(num_frames, dtype=np.float64) / config.frame_rate

    stage_start = perf_counter()
    stack_float = stack.astype(np.float64)
    # Pixel-wise standard deviation over time replicates MATLAB std(...,0,3).
    stddev_matrix = np.std(stack_float, axis=2, ddof=0)
    timings['stddev_matrix'] = perf_counter() - stage_start

    stage_start = perf_counter()
    threshold_mask, window_starts, window_ends = _build_threshold_mask(
        stddev_matrix,
        config.number_of_steps,
        config.consistency_check,
    )
    timings['sliding_window_thresholding'] = perf_counter() - stage_start

    # h-minima transformation with h=0.5 is the key anti-fragmentation control.
    stage_start = perf_counter()
    watershed_regions = _build_watershed_regions(threshold_mask, float(config.h), config.matlab_strict_mode)
    timings['watershed_refinement'] = perf_counter() - stage_start

    stage_start = perf_counter()
    cc_labels = label(watershed_regions, connectivity=2)
    regions = [region for region in regionprops(cc_labels) if region.area >= config.min_roi_area]
    detected_roi_count = len(regions)
    timings['connected_components'] = perf_counter() - stage_start

    roi_table_rows: list[dict[str, float]] = []
    roi_time_series = np.zeros((num_frames, detected_roi_count), dtype=np.float64)
    detected_centroids_xy = np.zeros((detected_roi_count, 2), dtype=np.float64)
    roi_std_values: list[np.ndarray] = []
    roi_areas = np.zeros(detected_roi_count, dtype=np.float64)
    roi_bboxes: list[tuple[int, int, int, int]] = []

    stage_start = perf_counter()
    for i, region in enumerate(regions):
        rr = region.coords[:, 0]
        cc = region.coords[:, 1]
        roi_cube = stack_float[rr, cc, :]

        y_centre, x_centre = region.centroid
        detected_centroids_xy[i] = [x_centre, y_centre]
        roi_areas[i] = float(region.area)
        roi_bboxes.append(tuple(int(value) for value in region.bbox))

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
    timings['roi_feature_extraction'] = perf_counter() - stage_start

    # This keeps the original MATLAB-style consistency score available for
    # ranking, and in this experiment it can also be used as an optional
    # post-detection acceptance rule.
    stage_start = perf_counter()
    roi_consistency_scores = _compute_roi_consistency(
        roi_std_values,
        window_starts,
        window_ends,
        config.consistency_pixel_frac,
    )
    keep_mask, consistency_filter_summary = _apply_consistency_filter(
        roi_consistency_scores,
        config.number_of_steps,
        config.min_roi_consistency_ratio,
    )
    if not np.all(keep_mask):
        roi_table_rows = [row for row, keep in zip(roi_table_rows, keep_mask) if keep]
        roi_time_series = roi_time_series[:, keep_mask]
        detected_centroids_xy = detected_centroids_xy[keep_mask]
        roi_std_values = [values for values, keep in zip(roi_std_values, keep_mask) if keep]
        roi_areas = roi_areas[keep_mask]
        roi_bboxes = [bbox for bbox, keep in zip(roi_bboxes, keep_mask) if keep]
        detected_roi_count = int(np.sum(keep_mask))
    timings['roi_consistency'] = perf_counter() - stage_start

    stage_start = perf_counter()
    (
        detected_centroids_xy,
        roi_time_series,
        roi_table_rows,
        roi_areas,
        roi_bboxes,
        merge_summary,
    ) = _apply_merge_filter(
        detected_centroids_xy,
        roi_time_series,
        roi_table_rows,
        roi_areas,
        roi_bboxes,
        config.merge_centroid_distance_px,
        config.merge_trace_correlation,
        config.merge_bbox_gap_px,
    )
    detected_roi_count = detected_centroids_xy.shape[0]
    timings['roi_merge'] = perf_counter() - stage_start

    # Contingency table matching against manual centroids.
    stage_start = perf_counter()
    tp, fn, fp, indicator = _matching_with_candidate_resolution(
        detected_centroids_xy,
        landmark_centroids,
        config.matching_tolerance,
    )
    timings['contingency_matching'] = perf_counter() - stage_start

    output_excel_file = None
    if config.write_excel:
        stage_start = perf_counter()
        output_excel_file = _write_excel_output(
            Path(config.output_excel_file),
            roi_table_rows,
            roi_time_series,
            time_vector,
        )
        timings['excel_export'] = perf_counter() - stage_start

    return CalcCatchResult(
        detected_roi_count=detected_roi_count,
        matched_count=tp,
        matching_indicator=float(indicator),
        true_positive=tp,
        false_negative=fn,
        false_positive=fp,
        output_excel_file=output_excel_file,
        consistency_filter_summary=consistency_filter_summary,
        merge_summary=merge_summary,
        stage_timings_sec=timings,
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
    parser.add_argument('--min-roi-consistency-ratio', type=float, default=None)
    parser.add_argument('--merge-centroid-distance-px', type=float, default=None)
    parser.add_argument('--merge-trace-correlation', type=float, default=None)
    parser.add_argument('--merge-bbox-gap-px', type=float, default=None)
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
        min_roi_consistency_ratio=args.min_roi_consistency_ratio,
        merge_centroid_distance_px=args.merge_centroid_distance_px,
        merge_trace_correlation=args.merge_trace_correlation,
        merge_bbox_gap_px=args.merge_bbox_gap_px,
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
