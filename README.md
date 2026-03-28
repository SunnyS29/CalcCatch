# CalcCatch: A Statistical Pipeline for Automated ROI Detection and Functional Connectivity Analysis

CalcCatch is a pipeline for automated ROI detection and functional connectivity analysis in Enteric Nervous System calcium imaging datasets.

## The Problem
Manual ROI annotation for ENS neurons is a major bottleneck and can require up to 4 months of expert labor. CalcCatch was developed as a Master's project at the University of Melbourne to provide a scalable and objective alternative.

## The Workflow (The 5 Stations)

### Station 1: Data Ingestion
Load 3D TIFF stacks into structured matrices and generate time vectors for temporal alignment.

### Station 2: Activity Mapping
Apply sliding window thresholding to pixel-wise standard deviation and identify consistently active regions.

### Station 3: Watershed Refinement
Use a negative distance transform with h-minima transformation before watershed segmentation to suppress noise and reduce over-segmentation in dense neuron clusters.

### Station 4: Connectivity Reconstruction
Reconstruct functional networks using Pearson correlation and transfer entropy. Pearson correlation captures synchronous activity and transfer entropy captures directional information flow.

### Station 5: Network Characterization
Quantify small-world structure and identify functional hub neurons through graph-theoretical metrics.

## Performance
CalcCatch achieved a matching indicator of approximately **0.59**, more than doubling the benchmark Detect MATLAB toolbox result (**0.30**).

Experimental subnote:
- Baseline matching indicator: `0.593`
- Morphological filtering: `0.465`
- Consistency-based ROI rejection: `0.349`
- Rigid motion correction: `0.523`
- Conservative merge filter: `0.593`

These branch-level tests did not improve the baseline detection score on the real validation dataset. The conservative merge rule slightly reduced false positives, but it did not improve the matching indicator.

## File Setup
Place local inputs in `data/` and keep script paths pointed to project-relative files.

Default dummy paths:

```text
data/example_stack.tif
data/example_manual_roi_coordinates.xlsx
results/algorithm_rois.xlsx
```

## Python Port (Parity Baseline)
A Python implementation is included so the pipeline can run without a MATLAB license while mirroring the original MATLAB workflow.

### Implementation Notes
- The activity map uses pixel-wise standard deviation across frames.
- Watershed refinement uses a MATLAB-equivalent h-minima transformation (`h=0.5`) to suppress shallow minima and prevent over-segmentation in dense ENS neuron clusters.
- Validation uses contingency table matching with nearest-candidate centroid resolution.
- The ROI time series export is compatible with downstream transfer entropy and graph-theory stages.

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python calc_catch.py \
  --tiff data/example_stack.tif \
  --coords data/example_manual_roi_coordinates.xlsx \
  --output results/algorithm_rois_python.xlsx \
  --metrics-json results/metrics_python.json
```

## MATLAB Scripts
- `CalcCatch.m`: Current MATLAB pipeline.

## MATLAB Dependencies
As of March 28, 2026, the MATLAB pipeline does not require a code change for compatibility with the current MathWorks function behavior used here.

Required:
- MATLAB
- Image Processing Toolbox

Used from Image Processing Toolbox:
- `bwdist`
- `imhmin`
- `watershed`
- `imdilate`
- `strel`
- `bwconncomp`
- `regionprops`
- ROI drawing and masking for manual QC with `drawpolygon` and ROI `createMask`

Included with base MATLAB and used by this script:
- `imfinfo`
- `imread`
- `readtable`
- `writetable`
- plotting and UI functions such as `figure`, `imagesc`, `scatter`, `questdlg`, and `ginput`

Optional:
- Parallel Computing Toolbox for thread-based or GPU acceleration only. It is not required for correctness or reproducibility of the pipeline.

Compatibility notes:
- `readtable` with `VariableNamingRule` remains current and avoids the older `PreserveVariableNames` pattern.
- `bwdist` gained expanded GPU support for 3-D images in R2025a, but this script does not depend on GPU execution.
- The interactive manual curation block requires a desktop MATLAB session with graphics support because it uses ROI drawing and dialog functions.

## Python Scripts
- `calc_catch.py`: Python parity baseline with ROI extraction, Excel export, and contingency table matching.
- `requirements.txt`: Python dependencies.

## License
MIT. See `LICENSE`.


