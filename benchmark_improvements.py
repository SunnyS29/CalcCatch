"""Reproducible threshold benchmark and bounded parameter experiment.

Usage: python benchmark_improvements.py --data-root /path/to/Calcium_Imaging
The development/validation split is fixed before candidate evaluation. It is
spatial within one acquisition, not evidence of independent generalization.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from skimage.measure import label, regionprops

import calc_catch as c
from optimized_calc_catch import build_threshold_mask


def score(xy, landmarks):
    tp, fn, fp, recall = c._matching_with_candidate_resolution(xy, landmarks, 10.)
    n, m = len(xy), len(landmarks)
    optimal = 0
    if n and m:
        edges = np.max(np.abs(landmarks[:, None] - xy[None]), axis=2) <= 10.
        assignment = maximum_bipartite_matching(csr_matrix(edges), perm_type='column')
        optimal = int(np.count_nonzero(assignment >= 0))
    return dict(detections=n, landmarks=m, tp=tp, fn=fn, fp=fp,
                recall=recall, precision=tp/n if n else 0.,
                f1=2*tp/(n+m) if n+m else 0., optimal_tp=optimal,
                optimal_f1=2*optimal/(n+m) if n+m else 0.)


def segmented_centroids(std_path, consistency, h, output):
    std = np.load(std_path)
    mask = build_threshold_mask(std, 5000, consistency)[0]
    regions = c._build_watershed_regions(mask, h, True)
    props = regionprops(label(regions, connectivity=2))
    rows = [[p.centroid[1], p.centroid[0], p.area] for p in props if p.area >= 10]
    np.save(output, np.array(rows).reshape(-1, 3))


def run(data_root, output, timeout):
    output.mkdir(parents=True, exist_ok=True)
    report = dict(environment={'python': platform.python_version(), 'packages': {
        p: importlib.metadata.version(p) for p in
        ('numpy', 'scipy', 'scikit-image', 'pandas', 'tifffile', 'openpyxl')}},
        grid=[], failures=[])

    def save():
        (output/'evaluation.json').write_text(json.dumps(report, indent=2))

    tiff = data_root/'ER620033D57ASAHIGH.tif'
    coords = data_root/'ER620033D57ASAHIGH_normROIs.xlsx'
    stack = c._to_hwf_stack(tiff)
    std = stack.std(axis=2, ddof=0)
    del stack
    landmarks = c._load_landmark_centroids(coords)
    std_path = output/'stddev-high.npy'
    np.save(std_path, std)
    np.save(output/'landmarks-high.npy', landmarks)
    original = c._build_threshold_mask
    np.testing.assert_array_equal(original(std, 5000, .075)[0],
                                  build_threshold_mask(std, 5000, .075)[0])
    report['threshold_timings'] = {'baseline': [], 'optimized': []}
    for _ in range(3):
        for name, fn in [('baseline', original), ('optimized', build_threshold_mask)]:
            start = perf_counter()
            fn(std, 5000, .075)
            report['threshold_timings'][name].append(perf_counter()-start)
    config = c.CalcCatchConfig(tiff_stack_path=str(tiff),
        original_coordinates_path=str(coords), write_excel=False)
    report['full_pipeline'] = {'baseline': [], 'optimized': []}
    try:
        for name, fn in [('baseline', original), ('optimized', build_threshold_mask)]:
            c._build_threshold_mask = fn
            for _ in range(3):
                result = asdict(c.run_calc_catch(config))
                report['full_pipeline'][name].append(result)
                print(name, result['matched_count'],
                      round(sum(result['stage_timings_sec'].values()), 3), flush=True)
    finally:
        c._build_threshold_mask = original
    expected = {k: v for k, v in report['full_pipeline']['baseline'][0].items()
                if k != 'stage_timings_sec'}
    for result in report['full_pipeline']['optimized']:
        assert expected == {k: v for k, v in result.items() if k != 'stage_timings_sec'}
    report['threshold_exact_parity_recordings'] = 1
    save()

    def development(xy):
        return score(xy[xy[:, 0] < 246], landmarks[landmarks[:, 0] < 246])

    # Child processes bound native watershed calls that can otherwise stall.
    for consistency, h in itertools.product((.06, .075, .09), (.25, .5, 1., 2.)):
        path = output/f'centroids-{consistency}-{h}.npy'
        command = [sys.executable, str(Path(__file__).resolve()), '--segment',
                   str(std_path), str(consistency), str(h), str(path)]
        try:
            subprocess.run(command, check=True, timeout=timeout, capture_output=True, text=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            report['failures'].append(dict(consistency=consistency, h=h,
                error=type(error).__name__, detail=str(error)))
            print('Skipped', consistency, h, type(error).__name__, flush=True)
            save()
            continue
        points = np.load(path)
        for area in (10, 20, 40):
            xy = points[points[:, 2] >= area, :2]
            report['grid'].append(dict(consistency=consistency, h=h, min_area=area,
                path=str(path), development=development(xy)))
        print('Evaluated', consistency, h, flush=True)
        save()

    if not report['grid']:
        raise RuntimeError('No parameter candidates completed; see evaluation.json.')
    selected = max(report['grid'], key=lambda r: (
        r['development']['optimal_f1'], r['development']['precision'],
        -abs(r['consistency']-.075), -abs(r['h']-.5), -r['min_area']))
    report['selected'] = selected
    baseline_path = output/'centroids-0.075-0.5.npy'
    if not baseline_path.exists():
        raise RuntimeError('Baseline segmentation failed; see evaluation.json.')
    baseline_xy = np.load(baseline_path)[:, :2]
    points = np.load(selected['path'])
    selected_xy = points[points[:, 2] >= selected['min_area'], :2]
    report['comparison'] = {}
    for name, xy in [('baseline', baseline_xy), ('selected', selected_xy)]:
        report['comparison'][name] = dict(full=score(xy, landmarks),
            development=development(xy),
            heldout=score(xy[xy[:, 0] > 266], landmarks[landmarks[:, 0] > 266]),
            one_based_sensitivity=score(xy+1, landmarks))
        np.save(output/f'{name}-centroids.npy', xy)
    save()

    second_tiff = data_root/'new_spontaneous/ER61033D43FSALOW.tif'
    second = c._to_hwf_stack(second_tiff).std(axis=2, ddof=0)
    np.testing.assert_array_equal(original(second, 5000, .075)[0],
                                  build_threshold_mask(second, 5000, .075)[0])
    report['threshold_exact_parity_recordings'] = 2
    report['notes'] = [
        'Spatial holdout is from the same recording, not an independent acquisition.',
        'Second recording has no spatial annotations; used for threshold parity only.',
        'Maximum-cardinality matching is separate from legacy matching.',
        'Zero-based coordinates are unchanged; +1 sensitivity is separate.',
        'Timing excludes Excel export and uses three warm repeated runs.',
        'Timed-out settings were excluded from candidate selection.']
    save()
    print(json.dumps(report['comparison'], indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--segment':
        segmented_centroids(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--data-root', required=True, type=Path)
        parser.add_argument('--output', type=Path, default=Path('results'))
        parser.add_argument('--timeout', type=float, default=12.)
        args = parser.parse_args()
        run(args.data_root, args.output.resolve(), args.timeout)
