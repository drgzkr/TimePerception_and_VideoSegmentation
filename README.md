# Naturalistic Video Segmentation and Temporal Accumulation

This repository contains two Jupyter notebooks that together implement a pipeline for processing naturalistic videos using VGG-19 features and applying a perceptual accumulation model inspired by [Roseboom et al. (2019)](https://www.nature.com/articles/s41467-018-08194-7). The goal is to detect salient perceptual changes in videos and use those changes to simulate human-like temporal judgments.

---

## Notebooks

### 1. [Video_to_PerceptualChange_Timeline](https://colab.research.google.com/drive/1Wu3Tn9061w8A8n8AyEq5c5tNM8zodTiF?usp=sharing.ipynb)

This notebook performs the following steps:

#### Step 1: Video to Frames
- Extracts every `n`th frame (default: every frame) from `.avi` videos in a specified directory.
- Saves them as `.jpg` images in a dedicated folder per video.

#### Step 2: Frames to VGG-19 Features
- Loads a pretrained VGG-19 model from PyTorch.
- Extracts activation maps from selected convolutional layers (`[7, 14, 27, 40, 53]`) for each frame.
- Saves the 4D tensors as `.npy` files with shape `(num_frames, num_kernels, height, width)`.

#### Step 3: Feature Difference Timeline
- Computes `1 - Pearson correlation` between consecutive frames to quantify feature change.
- Normalizes features by z-scoring across time.
- Produces:
  - Heatmaps of z-scored activations.
  - Correlation matrices.
  - Histograms of feature changes.
  - Line plots of temporal changes.
- Analyzes compression artifacts and outlines causes (e.g., periodic spikes due to codec behavior).

---

### 2. [Perceptual_Change_Accumulators](https://colab.research.google.com/drive/1m7vVhVd2EB-kRTPlQl_iK0oyvo4UdLRk?usp=sharing)

This notebook applies a perceptual accumulation model to the output of the first notebook.

#### Core Component: Dynamic Attention Thresholding
- Implements the method from Roseboom et al.:
  - Models when perceptual change exceeds a threshold.
  - Accumulates these events as a proxy for perceived time.
- Two thresholding strategies:
  - **Static**: Threshold = `(min + max) / 2` for each layer.
  - **Dynamic**: Threshold decays over time with stochastic reset upon change detection.

#### Outputs:
- Change timelines overlaid with thresholds.
- Cumulative plots of change events per layer.
- Raw count of threshold crossings (perceptual events) per layer.

These values can be used as features for predicting perceived duration.

---

## Usage

### 1. [Video to Change Timeline Notebook (Colab)](https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID_HERE)

1. **Run Notebook 1**:
   - Extract frames.
   - Generate VGG-19 feature maps.
   - Calculate frame-to-frame feature differences.

### 2. [Perceptual Accumulators Notebook (Colab)](https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID_HERE)

2. **Run Notebook 2**:
   - Load saved difference timelines.
   - Apply static and dynamic thresholding.
   - Compute accumulated perceptual changes.
   - Use these as inputs for modeling perceived time.



## Notes

- Eye-tracking integration is possbile for personalized models
- Video compression artifacts (e.g., periodic gray/unchanged frames) affect results.
- For perceptual duration modeling, segment each video into variable-length clips (as done in Roseboom et al.) to increase sample diversity.
