# Increment 1: exclude background-dominated ROIs

The optional `--exclude-background-rois` flag retains a detected region only
when at least half its pixels are present in the existing activity mask.
This is applied after watershed and minimum-area selection, before intensity
and trace extraction. Retained regions keep their geometry, centroids, features
and traces. The result reports `excluded_background_roi_count`.

The strict watershed path takes the complement of watershed ridges. That can
create large background basins as well as neuron candidates. On the labelled
recording, the two largest regions cover 193,415 and 34,435 pixels but contain
only 0.56% and 0.79% activity-mask pixels. Their centroids are over 37 pixels
from the nearest annotation. Extracting their intensity cubes accounts for
much of the remaining runtime after the earlier threshold optimization.

## Measured results

Both variants use `optimized_calc_catch.py`. Timings include input loading but
exclude Excel I/O, and are medians of three warm, alternating repetitions on
the same machine. Each recording is 512 x 512 pixels with 240 frames.

| Measurement | Existing fast runner | With background filter |
| --- | ---: | ---: |
| HIGH runtime | 1.465 s | 0.447 s |
| HIGH detected regions | 87 | 82 |
| HIGH matched annotations | 51 | 51 |
| HIGH unmatched detections (benchmark FP) | 36 | 31 |
| HIGH missed annotations | 35 | 35 |
| HIGH precision | 58.62% | 62.20% |
| HIGH recall | 59.30% | 59.30% |
| HIGH F1 | 0.5896 | 0.6071 |
| LOW runtime | 1.278 s | 0.387 s |
| LOW detected regions | 77 | 73 |

Runtime improved by 3.27x on HIGH and 3.30x on LOW relative to the previously
optimized runner. These are measured runtime improvements, not detection
accuracy multipliers. Retained feature values, time vectors and ROI traces
were exactly equal on both recordings. Exported ROI columns are renumbered
after filtering; compare ROI identity by centroid, not the column number.
The benchmark saves retained original zero-based indices for trace comparison.

Accuracy results are exploratory on the one previously examined annotated
recording. LOW has no spatial annotations, so its reduced ROI count is not
evidence of better detection. This filter did not recover any additional missed
neurons. Further labelled acquisitions are needed before changing defaults.
The half-active rule is a fixed foreground-support criterion, not a new tuned
area cutoff. The existing default and MATLAB pipeline remain unchanged.

## Run and verify

```sh
python optimized_calc_catch.py --tiff /path/to/stack.tif --coords /path/to/coordinates.xlsx --exclude-background-rois
python -m unittest discover -v
python benchmark_background_rois.py --data-root /path/to/Calcium_Imaging
```

Eleven unit/integration tests passed, including the earlier threshold parity
cases, foreground-support boundaries, neurons at image borders, empty results,
and unchanged retained traces. Empty ROI exports retain their column headers
and time axis. A real CLI Excel export was read back and verified to contain
82 ROI rows and 240 time rows with 82 trace columns plus Time.

Generated reports and exports are local under `results/background-filter/`,
which is ignored by Git. Input datasets are read only. The new benchmark uses
bounded subprocesses to avoid the watershed stalls found during the prior
parameter sweep. Software versions are recorded in its manifest.
