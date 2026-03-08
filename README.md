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

## File Setup
Place your local input files in the data/ directory and point CalcCatch.m to those paths.

Default dummy paths in CalcCatch.m:

```matlab
tiff_stack_path      = 'data/example_stack.tif';
original_coordinates = 'data/example_manual_roi_coordinates.xlsx';
output_excel_file    = 'results/algorithm_rois.xlsx';
```

## Generate Synthetic Sample Data
Run the generator script in MATLAB:

```matlab
generate_sample_data
```

This creates:
- `data/example_stack.tif`
- `data/example_manual_roi_coordinates.xlsx` (sheet name: `xy coord`)

Then run:

```matlab
CalcCatch
```

## Reproducibility and Parameter Transparency
Key parameters are documented in the MATLAB scripts, including sliding window thresholds, consistency criteria, ROI size filtering, and watershed depth (`h`). This supports controlled tuning across experiments while preserving reproducibility.

## Repository Contents
- `CalcCatch.m`: Main ROI detection pipeline.
- `generate_sample_data.m`: Synthetic data generator for local testing.
- `README.md`: Project documentation.
- `LICENSE`: MIT license.
- `data/`: Local data placeholder directory with ignore rules.

## Quick Start
1. Run `generate_sample_data` in MATLAB.
2. Confirm sample files were created in `data/`.
3. Run `CalcCatch`.

