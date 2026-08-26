# Digital Image Processing Coursework (Homeworks 1 – 6)

This repository contains complete, verified Python implementations and comprehensive theoretical reports for all six Digital Image Processing homework assignments.

Every algorithm in Homeworks 2 through 6 has been implemented **twice**:
1. **From Scratch**: Using pure NumPy array operations, explicit loops, and mathematical formulations without high-level library shortcuts.
2. **Library Standard**: Using OpenCV (`cv2`), SciPy, and Scikit-Image.
3. **Benchmarked**: Validated via Mean Squared Error ($\text{MSE}$), Structural Similarity Index ($\text{SSIM}$), high-precision execution timings (`time.perf_counter()`), and multi-panel visual comparison grids.

---

## Homework Navigation & Directory Structure

```text
hw/
├── hw1/                                <- Image Basics & Pixel Operations
│   ├── hw1_solution.py                 <- Executable solution script
│   ├── HW1_Report.md                   <- Comprehensive theoretical and visual report
│   └── outputs/                        <- HW1_Q1..Q4.bmp, color spaces grid, channel splits
│
├── hw2/                                <- Intensity Transformations & Histograms
│   ├── hw2_solution.py                 <- Negative, Gamma, Log, Piecewise, Equalization, Matching
│   ├── HW2_Report.md                   <- Full benchmark report (exact MSE=0.0 against OpenCV)
│   └── outputs/                        <- Visual comparison grids and output BMPs
│
├── hw3/                                <- Resolution, Interpolation & Geometric Transforms
│   ├── hw3_solution.py                 <- Sampling, Quantization, 4 Interpolators, Backward Warping
│   ├── HW3_Report.md                   <- Full analysis on pixelation, false contouring, affine warps
│   └── outputs/                        <- Multi-panel resolution grids and upscaled ROI comparisons
│
├── hw4/                                <- Spatial Domain Filtering
│   ├── hw4_solution.py                 <- Box, Gaussian, Median (10% Salt & Pepper), Laplacian, Sobel, Pipeline
│   ├── HW4_Report.md                   <- In-depth filtering theory and 6-stage combined pipeline report
│   └── outputs/                        <- Spatial filtering grids and denoised images
│
├── hw5/                                <- Frequency Domain Analysis
│   ├── hw5_solution.py                 <- 1D/2D DFT vs Radix-2 FFT, Phase/Mag, ILPF/BLPF/GLPF, HFE
│   ├── HW5_Report.md                   <- FFT complexity benchmarks (MSE ~ 10^-23) and Gibbs analysis
│   └── outputs/                        <- Centered spectra, phase reconstructions, filter grids
│
├── hw6/                                <- Frequency Applications & Advanced Transforms
│   ├── hw6_solution.py                 <- Butterworth Notch Reject, Homomorphic Filter, 2D DCT-II vs FFT
│   ├── HW6_Report.md                   <- Periodic noise removal, illumination normalization, JPEG DCT theory
│   └── outputs/                        <- Notch filter plots, homomorphic grids, 8x8 block DCT compaction
│
└── images/                             <- Standard test image dataset (Lena, Baboon, Peppers, Cameraman)
```

---

## Master Benchmark Summary Across All Homeworks

| Homework | Core Modules Implemented | Scratch Accuracy ($\text{MSE}$) | Execution Benchmark |
| :--- | :--- | :---: | :---: |
| **HW 1** | Array inspection, Center line, Diagonal line, ROI assignment, 12-channel color decompositions (RGB, HSV, CIELAB, YCrCb) | **Exact Match** | Real-time pixel ops |
| **HW 2** | Image Negative, Gamma transform ($\gamma=0.4, 0.67, 1.5, 2.5$), Log transform, Piecewise-linear stretching, Histogram Equalization, Histogram Matching (Gaussian & Natural) | **$\text{MSE} = 0.000000$** | $\sim 0.5 - 20\text{ ms}$ |
| **HW 3** | Spatial Downsample/Upsample ($k=8..4$), Intensity Quantization ($k=8..1$), Nearest / Bilinear / Bicubic / Lanczos-4 on $64\times 64$ ROI, Backward Warping (Translation, Scaling, Center Rotation $45^\circ$) | **$\text{MSE} \le 0.004$ (interp)** | $\sim 1 - 60\text{ ms}$ |
| **HW 4** | Box filters ($3\times 3, 5\times 5, 9\times 9$), Gaussian filters, Median filter vs Mean on 10% Salt & Pepper noise, Laplacian ($K_1, K_2$), Unsharp Masking ($k=1.0, 4.5$), Sobel Gradient Magnitude/Phase, 6-Stage Combined Pipeline | **$\text{MSE} = 0.000000$ (Lap/Sobel/Box)** | Denoising $\text{MSE}: 1864 \to 67.6$ |
| **HW 5** | 1D DFT $O(N^2)$, Radix-2 1D FFT $O(N \log N)$, 2D Separable FFT, Phase-only vs Magnitude-only reconstruction, Ideal / Butterworth / Gaussian Low-Pass ($D_0=10,30,60$), High-Frequency Emphasis ($H_{\text{HFE}} = 0.5 + 2.0 H_{\text{HP}}$) + HistEq | **$\text{MSE} \approx 10^{-23}$ vs compiled FFTW** | Log-time FFT scaling |
| **HW 6** | Sinusoidal Noise + Butterworth Notch Reject Filter, Homomorphic Illumination-Reflectance Filtering ($\gamma_L=0.25, \gamma_H=2.0$), 2D DCT-II / IDCT-II from scratch vs FFT Energy Compaction on $8\times 8$ block | **$\text{MSE} = 1.58 \times 10^{-12}$ (DCT)** | Noise removal: $\text{SSIM } 0.30 \to 0.96$ |
