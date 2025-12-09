---
license: mit
pipeline_tag: image-to-image
library_name: pytorch
tags:
  - computer-vision
  - image-restoration
  - image-enhancement
  - denoising
  - deblurring
  - comfyui
  - pytorch
  - nafnet
---

# JPG Artifacts & Noise Remover (V2)

This is a lightweight model designed to remove JPEG compression artifacts, digital noise, and slight blur from images. Ideally suited for anime/illustration art, but works reasonably well on photos. 

**Version 2 Update:**
*   **New Architecture:** Switched to a NAFNet-based UNet (State-of-the-Art blocks for restoration).
*   **Larger Dataset:** Trained on ~40,000 high-quality images from Danbooru2024.
*   **Improved Training:** Added Perceptual Loss, HFEN, and blur degradation handling.

## Examples

*Placeholder*

## How to use in ComfyUI

The model is designed to work with the **[JPG & Noise Remover ComfyUI Node](https://github.com/SnJake/SnJake_JPG_Artifacts_Noise_Cleaner)**.

1.  **Install the Node:** Follow instructions in the [GitHub Repository](https://github.com/SnJake/SnJake_JPG_Artifacts_Noise_Cleaner).
2.  **Download Weights:** Download the `.pt` or `.safetensors` v2 file from this repository.
3.  **Place Weights:** Put the file in `ComfyUI/models/artifacts_remover/`.
4.  **Select Model:** Select the new weight file in the node settings. Ensure `base_ch` is set to **64**.

## Training Details (V2)

The goal was to create a restorer that not only removes noise but preserves fine structural details without "plastic" smoothing.

### Dataset
Trained on **40,000 images** from the [Danbooru2024](https://huggingface.co/datasets/deepghs/danbooru2024) dataset (anime/illustration style).

### Architecture: NAFNet-based UNet
The model uses a U-Net structure but replaces standard convolutional blocks with **NAFBlocks** (Nonlinear Activation Free Blocks).
*   **SimpleGate:** Replaces complex activation functions with element-wise multiplication.
*   **LayerNorm2d:** Stabilizes training.
*   **Simplified Channel Attention (SCA):** captures global context efficiently.
*   **Base Channels:** 64

### Degradation Pipeline
The model is trained on "on-the-fly" generated pairs. The degradation pipeline is more aggressive than V1:
1.  **Blur:** Random downscale-upscale (probability 50%, scale down to 0.85x) to simulate soft blur.
2.  **JPEG Compression:** Quality 10 - 85.
3.  **Gaussian Noise:** Standard deviation 0 - 20.0.
4.  **Identity:** 2% of images are left clean to teach the model not to alter good images.

### Loss Function
A composite loss function was used to balance pixel accuracy and perceptual quality:
*   **Pixel Loss:** Charbonnier Loss (0.7) + MixL1SSIM (0.3).
*   **Perceptual Loss (VGG19):** Weight 0.05. Helps generating realistic textures.
*   **HFEN (High-Frequency Error Norm):** Weight 0.1. Enforces edge reconstruction.
*   **Gradient/Edge Loss:** Weight 0.2.
*   **Identity Loss:** Weight 0.02 (applied on clean images).

### Training Config
*   **Optimizer:** AdamW (`lr=2e-4`, `wd=1e-4`).
*   **Scheduler:** Cosine Annealing (2000 warmup steps).
*   **Precision:** BFloat16 (AMP).
*   **Patch Size:** 288x288.
*   **Batch Size:** 4 (Accumulated to effective 8).

## Limitations
*   Primarily trained on anime/2D art.
*   May struggle with extremely heavy motion blur (since it trained mostly on slight downscale blur).