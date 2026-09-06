# Performance evaluation, 6 September 2026

The opt-in runner `optimized_calc_catch.py` replaces the 5,000 full-image
threshold scans with two binary searches per pixel. It accepts the same CLI
arguments as `calc_catch.py`; the MATLAB and Python baseline files are unchanged.

## Measured result

Median of three repeated runs on a local 512 x 512 x 240 uint16 recording,
with Excel export disabled and the same installed dependencies for both runs:

| Measurement | Baseline | Optimized |
| --- | ---: | ---: |
| Threshold stage | 1.9653 s | 0.01576 s |
| Full pipeline, including input loading | 3.0428 s | 1.6841 s |
| Detected ROIs | 87 | 87 |
| Matched annotations | 51 / 86 | 51 / 86 |
| Precision | 0.5862 | 0.5862 |
| Recall (legacy matching indicator) | 0.5930 | 0.5930 |
| F1 | 0.5896 | 0.5896 |

This is approximately 125x faster for thresholding and 1.81x faster overall
in this run. Timings vary with machine load and caching. Threshold masks were
exactly equal on two real recordings and 107 synthetic/edge configurations,
including constant maps, exact interval boundaries and adjacent floats.
All full-pipeline summary values except timing were equal across implementations.

## Detection experiments

The predeclared parameter grid used consistency values 0.06, 0.075 and 0.09;
h values 0.25, 0.5, 1 and 2; and minimum ROI areas 10, 20 and 40.
Twenty-four parameter combinations completed. Four segmentation jobs at
consistency 0.06 exceeded their 12-second timeouts, preventing evaluation of
the other twelve combinations. A preceding unbounded experiment stalled inside
the watershed implementation and was terminated. The harness uses child
processes so native-code stalls can be terminated without losing prior results.

Candidates were selected by maximum-cardinality F1 on centroids with x < 246.
Centroids with x > 266 were held out, with a 20-pixel exclusion strip. Selection
returned the original settings (0.075, 0.5, 10). Held-out F1 remained 0.5316.
This sweep provides no evidence for changing the detection defaults. The spatial
holdout shares an acquisition with the development region and is not independent
validation. The second recording lacks spatial annotations and was used only
for threshold equivalence, not accuracy scoring.

The existing greedy matcher can return different counts when annotation order
changes. The benchmark reports order-independent maximum-cardinality matching
separately, preserving the legacy score for comparison. Both methods produce
51 matches for the original full-image baseline. Existing coordinate conventions
are unchanged; adding one pixel to the baseline centroids also produced 51 matches.

## Reproduce

```sh
python -m pip install -r requirements.txt
python -m unittest -v test_improvements
python optimized_calc_catch.py --tiff /path/to/stack.tif --coords /path/to/coordinates.xlsx --no-excel
python benchmark_improvements.py --data-root /path/to/Calcium_Imaging
```

The benchmark expects the existing local HIGH recording and its coordinate
spreadsheet at the root, and the LOW recording under `new_spontaneous/`.
Inputs are read only. Generated JSON, timings and coordinate arrays stay in
the git-ignored `results/` directory; no research data are bundled with this change.

Measured environment: Python 3.12.4, NumPy 2.5.2, SciPy 1.18.1,
scikit-image 0.26.0, pandas 3.0.5, tifffile 2026.8.23 and openpyxl 3.1.5.
Baseline revision: `a6fae51`.
