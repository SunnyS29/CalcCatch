# Exploratory recovery evaluation

## Decision

Keep the existing detector unchanged. Splitting potentially merged ROIs recovered
four additional annotations, but added one unmatched detection. No candidate
passed the predeclared gate: recover at least one annotation without increasing
false positives or reducing precision, under both matching methods.

This is a separate experiment, not a production option or a validated accuracy
improvement. No production CLI, defaults, or extraction code were changed.

## Results

Recording: ER620033D57ASAHIGH.tif, with 86 manual centroids from
ER620033D57ASAHIGH_normROIs.xlsx. Baseline uses the optimized threshold search,
5,000 steps, consistency 0.075, strict watershed with fraction 0.5, minimum ROI
area 10, and the existing background-support filter enabled.

| Candidate | Detections | Matches | Unmatched (FP) | Misses | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 82 | 51 | 31 | 35 | 62.20% | 59.30% | 0.6071 |
| Peak distance 8 | 95 | 55 | 40 | 31 | 57.89% | 63.95% | 0.6077 |
| Peak distance 10 | 89 | 55 | 34 | 31 | 61.80% | 63.95% | 0.6286 |
| Peak distance 12 | 87 | 55 | 32 | 31 | 63.22% | 63.95% | 0.6358 |
| Distance 12, prominence 0.2 | 87 | 55 | 32 | 31 | 63.22% | 63.95% | 0.6358 |

Distances 8, 10 and 12 were evaluated first; prominence 0.2 was tested after
inspecting those results. These are adaptive experiments on a previously examined
recording, not held-out validation. Both the legacy matcher and a
maximum-cardinality bipartite matcher gave the same aggregate counts. Matching
uses the existing tolerance of 10 pixels on each coordinate, not Euclidean
distance. The best candidates split five parents into ten children. Under the
maximum-cardinality assignment, newly matched zero-based annotation indices were
27, 43, 78 and 83, with no previously matched annotation lost.

An unmatched centroid is counted as a false positive against this annotation set;
it is not proof of a biologically false neuron. Conversely, proximity to a manual
centroid does not establish correct segmentation or valid calcium traces.

## Method And Checks

The experimental module smooths the temporal standard-deviation image with a
one-pixel Gaussian, finds separated activity peaks inside each retained ROI,
and uses them as watershed markers. It rejects a whole split if any resulting
child has fewer than 10 pixels. The optional prominence check rejects shallow
peaks. The union of ROI pixels is preserved, but individual masks and therefore
their extracted traces can change. Trace quality was not evaluated here.

Seven new tests cover two-peak splits, flat/single-peak parents, shallow peaks,
small-child rejection, empty inputs, determinism/borders, and invalid parameters.
Together with the existing regression tests, 18 tests pass. This is an accuracy
experiment; a full end-to-end runtime improvement was not established. JSON
refinement timings cover splitting and scoring only.

The local overlay shows all five proposed splits, original contours, child
centroids, and nearby manual annotations. Several regions are elongated and
need manual review; the overlay alone does not validate biological identity.

## Independent Validation Still Needed

No second independently annotated recording with spatial coordinates and clear
provenance was located. The LOW workbook contains traces but no spatial labels.
Other coordinate files appear to be algorithm exports, or lack a matching raw
recording and clear annotation provenance. They were not treated as ground truth.
Further tuning on this recording risks overfitting. Before promoting a splitting
strategy, obtain another manually annotated acquisition and evaluate centroid
matching, segmentation quality, and extracted traces without retuning on it.

## Reproduce

Run from the repository with its scientific Python dependencies installed:

```sh
python -B -m unittest discover -v
python -B evaluate_recovery.py --data-root "/path/to/Calcium_Imaging"
python -B plot_recovery.py
```

The evaluator runs its worker with a 60-second timeout. Plotting additionally
requires Matplotlib. Results, cached label maps and the overlay are written to
the ignored `results/recovery/` directory and should remain local, alongside raw
data. `evaluation.json` records the environment and all four candidates, including
failures of the acceptance gate. `proposed-splits.png` is the review overlay.

Technique references: [scikit-image peak detection](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max)
and [marker-based watershed](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html).
