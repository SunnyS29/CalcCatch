"""Compare the opt-in background filter with the existing fast runner.

Each recording runs in a bounded subprocess. Real inputs are read only;
outputs go into the git-ignored results directory.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from unittest.mock import patch

import numpy as np

import calc_catch as c
from optimized_calc_catch import build_threshold_mask


def evaluate(data_root, recording, output):
    labelled = recording == 'high'
    tiff = data_root / ('ER620033D57ASAHIGH.tif' if labelled else
                        'new_spontaneous/ER61033D43FSALOW.tif')
    config_args = dict(tiff_stack_path=str(tiff),
        original_coordinates_path=str(data_root/'ER620033D57ASAHIGH_normROIs.xlsx'),
        write_excel=False)
    report = dict(recording=recording, spatial_annotations_available=labelled,
                  repetitions={'baseline': [], 'filtered': []})
    c._build_threshold_mask = build_threshold_mask
    if not labelled:
        c._load_landmark_centroids = lambda _: np.empty((0, 2))
    # Warm both paths before alternating timed runs.
    for enabled in (False, True):
        c.run_calc_catch(c.CalcCatchConfig(**config_args, exclude_background_rois=enabled))
    for _ in range(3):
        for name, enabled in [('baseline', False), ('filtered', True)]:
            result = asdict(c.run_calc_catch(c.CalcCatchConfig(
                **config_args, exclude_background_rois=enabled)))
            result['total_seconds'] = sum(result['stage_timings_sec'].values())
            if not labelled:
                for field in ('matched_count', 'matching_indicator', 'true_positive',
                              'false_negative', 'false_positive'):
                    result.pop(field)
            report['repetitions'][name].append(result)

    exports = []

    def capture(_path, rows, traces, time):
        exports.append((rows, traces.copy(), time.copy()))
        return None

    export_args = dict(config_args, write_excel=True)
    with patch.object(c, '_write_excel_output', side_effect=capture):
        for enabled in (False, True):
            c.run_calc_catch(c.CalcCatchConfig(**export_args, exclude_background_rois=enabled))
    before, after = exports
    keys = [(r['x_Centre'], r['y_Centre']) for r in before[0]]
    assert len(set(keys)) == len(keys), 'Ambiguous centroids in trace comparison'
    indices = [keys.index((r['x_Centre'], r['y_Centre'])) for r in after[0]]
    assert [before[0][i] for i in indices] == after[0]
    np.testing.assert_array_equal(before[1][:, indices], after[1])
    np.testing.assert_array_equal(before[2], after[2])
    report['retained_features_and_traces_exactly_equal'] = True
    report['retained_original_roi_indices_zero_based'] = indices
    report['median_seconds'] = {name: statistics.median(r['total_seconds'] for r in rows)
                                for name, rows in report['repetitions'].items()}
    report['speedup'] = report['median_seconds']['baseline']/report['median_seconds']['filtered']
    if labelled:
        report['accuracy'] = {}
        for name, rows in report['repetitions'].items():
            r = rows[0]
            tp, fp, fn = r['true_positive'], r['false_positive'], r['false_negative']
            report['accuracy'][name] = dict(tp=tp, fp=fp, fn=fn,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'repetitions'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/background-filter'))
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--recording', choices=('high', 'low'))
    args = parser.parse_args()
    if args.recording:
        evaluate(args.data_root, args.recording, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(python=platform.python_version(), packages={
        p: importlib.metadata.version(p) for p in
        ('numpy', 'scipy', 'scikit-image', 'pandas', 'tifffile', 'openpyxl')},
        notes=['Both variants use the previously tested fast threshold implementation.',
               'Timings exclude Excel I/O and use three warm alternating repetitions.',
               'Accuracy is exploratory on one previously examined recording.',
               'LOW has no spatial annotations and cannot validate detection accuracy.'],
        outcomes={})
    for recording in ('high', 'low'):
        command = [sys.executable, '-B', str(Path(__file__).resolve()),
            '--data-root', str(args.data_root), '--recording', recording,
            '--output', str((args.output/f'{recording}.json').resolve())]
        try:
            subprocess.run(command, check=True, timeout=args.timeout)
            manifest['outcomes'][recording] = 'completed'
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            manifest['outcomes'][recording] = type(error).__name__
            print(recording, type(error).__name__, flush=True)
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
