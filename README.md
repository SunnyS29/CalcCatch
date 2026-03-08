# CalcCatch: A Statistical Pipeline for Automated ROI Detection and Functional Connectivity Analysis

CalcCatch is a MATLAB-based pipeline for automated ROI detection and functional connectivity analysis in Enteric Nervous System (ENS) calcium imaging datasets.

## The Problem
Manual ROI annotation for ENS neurons is a major bottleneck and can require up to 4 months of expert labor. CalcCatch was developed as a Master's project at the University of Melbourne to provide a scalable and objective alternative.

## The Workflow (The 5 Stations)

### Station 1: Data Ingestion
We load 3D TIFF stacks into structured matrices and generate time vectors for temporal alignment.

### Station 2: Activity Mapping
We apply sliding window thresholding to pixel-wise temporal standard deviation and identify consistently active regions.

### Station 3: Watershed Refinement
We use a negative distance transform with h-minima transformation before watershed segmentation to suppress noise and reduce over-segmentation in dense neuron clusters.

### Station 4: Connectivity Reconstruction
We reconstruct functional networks using Pearson Correlation and Transfer Entropy. Pearson Correlation captures synchronous activity and Transfer Entropy captures directional information flow.

### Station 5: Network Characterization
We quantify small-world structure and identify functional hub neurons through graph-theoretical metrics.

## Performance
CalcCatch achieved a matching indicator of approximately **0.59**, more than doubling the benchmark Detect MATLAB toolbox result (**0.30**).

## AI-Augmented Development
ChatGPT 4o was used in a limited capacity for debugging index errors, restructuring for readability, and adjusting figure formatting. We manually verified all data processing and analytical logic to ensure biological and mathematical accuracy.

## Data Integrity Protocol
This repository is maintained as a code-first, reproducible release. Do not upload proprietary raw imaging data.

- Raw `.tif` and related lab data should stay local and outside version control.
- Use the `data/` folder as a local mount point for your own input files.
- Keep file references in scripts pointed to dummy-style project paths, then adapt locally.

Example local paths to use:

```matlab
tiff_stack_path      = 'data/example_stack.tif';
original_coordinates = 'data/example_manual_roi_coordinates.xlsx';
output_excel_file    = 'results/algorithm_rois.xlsx';
```

## Reproducibility and Parameter Transparency
Key parameters are documented in the MATLAB scripts, including sliding window thresholds, consistency criteria, ROI size filtering, and watershed depth (`h`). This supports controlled tuning across experiments while preserving reproducibility.

## Repository Contents
- `ROI_recognition(CalcCatch)_annotated.m`: Automated ROI detection, consistency scoring, and ROI export.
- `metrics_algorithm.m`: Connectivity and graph metrics for algorithm-derived ROIs.
- `metrics_manual.m`: Connectivity and graph metrics for manually annotated ROIs.
- `binary_distance(graph_metric_test.m`: Binary distance and graph metric test script.
- `Final_Thesis_Copy.pdf`: Thesis reference for methods and benchmark context.
- `data/`: Placeholder directory for user-supplied local data only.

## Quick Start
1. Place your own TIFF stack and ROI coordinate spreadsheet in `data/`.
2. Adjust paths and tunable parameters in `ROI_recognition(CalcCatch)_annotated.m`.
3. Run ROI extraction.
4. Run connectivity and graph metric scripts for network analysis.
