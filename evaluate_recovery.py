"""Reproduce exploratory splitting tests, including the follow-up prominence test."""
import argparse
import json
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from skimage.measure import label, regionprops

import calc_catch as c
from optimized_calc_catch import build_threshold_mask
from recovery_experiment import split_activity_peaks


def scores(xy, landmarks):
    tp, fn, fp, recall = c._matching_with_candidate_resolution(xy, landmarks, 10.)
    edges = np.max(np.abs(landmarks[:, None]-xy[None]), axis=2) <= 10.
    assignment = maximum_bipartite_matching(csr_matrix(edges), perm_type='column')
    optimal = int(np.sum(assignment >= 0))
    return dict(detections=len(xy), tp=tp, fp=fp, fn=fn, recall=recall,
        precision=tp/len(xy) if len(xy) else 0., f1=2*tp/(len(xy)+len(landmarks)),
        optimal_tp=optimal, optimal_fp=len(xy)-optimal,
        optimal_matched_indices=np.flatnonzero(assignment >= 0).tolist())


def evaluate(data_root, output):
    output.mkdir(parents=True, exist_ok=True)
    stack = c._to_hwf_stack(data_root/'ER620033D57ASAHIGH.tif')
    std = stack.astype(np.float64).std(axis=2)
    del stack
    landmarks = c._load_landmark_centroids(data_root/'ER620033D57ASAHIGH_normROIs.xlsx')
    mask = build_threshold_mask(std, 5000, .075)[0]
    labels = label(c._build_watershed_regions(mask, .5, True), connectivity=2)
    regions = c._filter_background_regions(
        [r for r in regionprops(labels) if r.area >= 10], labels, mask)
    xy = np.array([[r.centroid[1], r.centroid[0]] for r in regions]).reshape(-1, 2)
    baseline = scores(xy, landmarks)
    baseline_labels = np.where(np.isin(labels, [r.label for r in regions]), labels, 0)
    np.savez_compressed(output/'baseline.npz', std=std, labels=baseline_labels,
                        centroids=xy, landmarks=landmarks)
    report = dict(baseline=baseline, candidates=[],
        environment={'python': platform.python_version(), 'packages': {
            p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-image')}},
        exploration_history='Distances 8, 10 and 12 were evaluated first. Prominence 0.2 at distance 12 was a follow-up after viewing those results.',

        selection_rule='Recover at least one additional annotation without increasing FP or reducing precision, under both scoring methods.',
        independent_validation=False,
        limitation='One previously examined annotated acquisition; settings are exploratory, not held-out validation.')
    for distance, prominence in ((8, 0.), (10, 0.), (12, 0.), (12, .2)):
        start = perf_counter()
        split, count = split_activity_peaks(regions, std, min_distance=distance, prominence_fraction=prominence)
        child_regions = regionprops(split)
        candidate_xy = np.array([[r.centroid[1], r.centroid[0]] for r in child_regions]).reshape(-1, 2)
        score = scores(candidate_xy, landmarks)
        eligible = (score['tp'] > baseline['tp'] and score['fp'] <= baseline['fp']
            and score['precision'] >= baseline['precision']
            and score['optimal_tp'] > baseline['optimal_tp']
            and score['optimal_fp'] <= baseline['optimal_fp'])
        gained = sorted(set(score['optimal_matched_indices'])-set(baseline['optimal_matched_indices']))
        lost = sorted(set(baseline['optimal_matched_indices'])-set(score['optimal_matched_indices']))
        result = dict(min_distance=distance, prominence_fraction=prominence, split_parent_count=count,
            refinement_seconds=perf_counter()-start, scores=score,
            recovered_annotation_indices=gained, lost_annotation_indices=lost,
            meets_predeclared_gate=eligible)
        report['candidates'].append(result)
        np.savez_compressed(output/f'split-{distance}-prominence-{prominence}.npz', labels=split, centroids=candidate_xy)
        print(json.dumps(result), flush=True)
        (output/'evaluation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/recovery'))
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        evaluate(args.data_root, args.output)
    else:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
            '--data-root', str(args.data_root), '--output', str(args.output.resolve())]
        try:
            subprocess.run(command, check=True, timeout=60.)
        except subprocess.TimeoutExpired:
            raise SystemExit('Recovery evaluation exceeded 60 seconds; worker stopped.')
