import os
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# =====================================================================
# BENCHMARK & COMPARISON HELPERS
# =====================================================================
def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Squared Error between two images."""
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))

def benchmark_timer(func, *args, **kwargs):
    """Executes func and returns (result, elapsed_ms)."""
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    return res, (t1 - t0) * 1000.0

# =====================================================================
# TASK 1: PERIODIC NOISE & BUTTERWORTH NOTCH REJECT FILTERING
# =====================================================================
def add_sinusoidal_noise(img: np.ndarray, u0: int = 32, v0: int = 32, A: float = 50.0) -> tuple:
    """
    Adds periodic sinusoidal noise: n(x, y) = A * sin(2*pi*u0*x/M + 2*pi*v0*y/N)
    Returns (noisy_img, noise_field).
    """
    M, N = img.shape
    y, x = np.ogrid[:M, :N]
    noise = A * np.sin(2.0 * np.pi * u0 * x / M + 2.0 * np.pi * v0 * y / N)
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy, noise

def butterworth_notch_reject_filter(M: int, N: int, u0: int = 32, v0: int = 32, D0: float = 15.0, n: int = 2) -> np.ndarray:
    """
    Designs a Butterworth Notch Reject Filter centered at (u0, v0) and (-u0, -v0):
    H_NRF = 1 / (1 + (D0^2 / (D1 * D2))^n)
    where D1 = sqrt((u - M/2 - u0)^2 + (v - N/2 - v0)^2)
          D2 = sqrt((u - M/2 + u0)^2 + (v - N/2 + v0)^2)
    """
    u = np.arange(M) - M / 2.0
    v = np.arange(N) - N / 2.0
    V, U = np.meshgrid(v, u)

    D1 = np.sqrt((U - u0) ** 2 + (V - v0) ** 2)
    D2 = np.sqrt((U + u0) ** 2 + (V + v0) ** 2)

    D_prod = D1 * D2
    H_nrf = 1.0 / (1.0 + ( (D0 ** 2) / (D_prod + 1e-8) ) ** n)
    return H_nrf

def apply_notch_filter(noisy_img: np.ndarray, H_nrf: np.ndarray) -> np.ndarray:
    """Applies centered frequency domain notch filter."""
    M, N = noisy_img.shape
    y, x = np.ogrid[:M, :N]
    checker = (-1.0) ** (x + y)

    F_c = np.fft.fft2(noisy_img.astype(np.float64) * checker)
    G_c = F_c * H_nrf
    restored = np.real(np.fft.ifft2(G_c)) * checker
    return np.clip(np.round(restored), 0, 255).astype(np.uint8)


# =====================================================================
# TASK 2: HOMOMORPHIC FILTERING
# =====================================================================
def homomorphic_filter_scratch(img: np.ndarray, gamma_L: float = 0.25, gamma_H: float = 2.0, 
                               c: float = 1.0, D0: float = 30.0) -> tuple:
    """
    Homomorphic Filtering on illumination-reflectance model:
    1. z = ln(f + 1)
    2. Z = FFT2(z * (-1)^(x+y))
    3. H_homo = (gamma_H - gamma_L) * (1 - exp(-c * D^2 / D0^2)) + gamma_L
    4. s = IFFT2(H_homo * Z) * (-1)^(x+y)
    5. g = exp(Re(s)) - 1
    """
    M, N = img.shape
    y, x = np.ogrid[:M, :N]
    checker = (-1.0) ** (x + y)

    # 1. Log domain
    f = img.astype(np.float64)
    z = np.log(1.0 + f)

    # 2. Centered FFT
    Z = np.fft.fft2(z * checker)

    # 3. Filter transfer function
    u = np.arange(M) - M / 2.0
    v = np.arange(N) - N / 2.0
    V, U = np.meshgrid(v, u)
    D2 = U ** 2 + V ** 2
    H_homo = (gamma_H - gamma_L) * (1.0 - np.exp(-c * D2 / (D0 ** 2 + 1e-8))) + gamma_L

    # 4. Inverse FFT
    S_c = Z * H_homo
    s = np.real(np.fft.ifft2(S_c)) * checker

    # 5. Exponential inverse
    g = np.exp(s) - 1.0

    # Rescale & clip to [0, 255]
    g_clipped = np.clip(g, 0, None)
    g_norm = (g_clipped - g_clipped.min()) / (g_clipped.max() - g_clipped.min() + 1e-8) * 255.0
    return np.clip(np.round(g_norm), 0, 255).astype(np.uint8), H_homo


# =====================================================================
# TASK 3: 2D DCT-II & IDCT-II FROM SCRATCH VS FFT ENERGY COMPACTION
# =====================================================================
def get_dct_matrix_1d(N: int) -> np.ndarray:
    """Constructs the orthogonal N x N 1D DCT-II transformation matrix."""
    T = np.zeros((N, N), dtype=np.float64)
    for u in range(N):
        alpha = np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)
        for x in range(N):
            T[u, x] = alpha * np.cos((2.0 * x + 1.0) * u * np.pi / (2.0 * N))
    return T

def dct2_scratch(block: np.ndarray) -> np.ndarray:
    """
    Computes 2D DCT-II from scratch via matrix multiplication:
    C = T @ block @ T.T
    Matches OpenCV cv2.dct scaling.
    """
    N = block.shape[0]
    T = get_dct_matrix_1d(N)
    return np.dot(T, np.dot(block.astype(np.float64), T.T))

def idct2_scratch(coeffs: np.ndarray) -> np.ndarray:
    """
    Computes 2D IDCT-II from scratch via orthogonal transpose:
    f = T.T @ coeffs @ T
    """
    N = coeffs.shape[0]
    T = get_dct_matrix_1d(N)
    return np.dot(T.T, np.dot(coeffs.astype(np.float64), T))

def dct2_opencv(block: np.ndarray) -> np.ndarray:
    """Computes 2D DCT using OpenCV."""
    return cv2.dct(block.astype(np.float32))

def idct2_opencv(coeffs: np.ndarray) -> np.ndarray:
    """Computes 2D IDCT using OpenCV."""
    return cv2.idct(coeffs.astype(np.float32))

def energy_compaction_experiment(block: np.ndarray, keep_k: int = 3) -> dict:
    """
    Compares 8x8 block reconstruction keeping only top-left keep_k x keep_k (3x3) coefficients:
    - DCT truncation (zero outer coefficients).
    - FFT truncation (zero high frequencies).
    """
    N = block.shape[0]
    block_f = block.astype(np.float64)

    # 1. DCT Pipeline
    dct_coeffs = dct2_scratch(block_f)
    dct_truncated = np.zeros_like(dct_coeffs)
    dct_truncated[:keep_k, :keep_k] = dct_coeffs[:keep_k, :keep_k]
    recon_dct = idct2_scratch(dct_truncated)
    recon_dct_clipped = np.clip(np.round(recon_dct), 0, 255).astype(np.uint8)

    # 2. FFT Pipeline (Centered)
    y, x = np.ogrid[:N, :N]
    checker = (-1.0) ** (x + y)
    fft_coeffs = np.fft.fft2(block_f * checker)
    
    # In centered spectrum, keep center 3x3 low frequencies
    center = N // 2
    half_k = keep_k // 2
    fft_truncated = np.zeros_like(fft_coeffs)
    fft_truncated[center - half_k : center + half_k + 1, center - half_k : center + half_k + 1] = \
        fft_coeffs[center - half_k : center + half_k + 1, center - half_k : center + half_k + 1]
    
    recon_fft = np.real(np.fft.ifft2(fft_truncated)) * checker
    recon_fft_clipped = np.clip(np.round(recon_fft), 0, 255).astype(np.uint8)

    # MSE
    mse_dct = compute_mse(recon_dct_clipped, block)
    mse_fft = compute_mse(recon_fft_clipped, block)

    return {
        "block_orig": block,
        "dct_coeffs": dct_coeffs,
        "dct_truncated": dct_truncated,
        "recon_dct": recon_dct_clipped,
        "mse_dct": mse_dct,
        "fft_coeffs": fft_coeffs,
        "recon_fft": recon_fft_clipped,
        "mse_fft": mse_fft
    }


# =====================================================================
# MAIN EXECUTION & BENCHMARK SUITE
# =====================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "..", "images")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print("Homework 6: Frequency Applications and Advanced Transforms")
    print("=" * 75)

    lena_256 = cv2.imread(os.path.join(img_dir, "LenaGrey256.bmp"), cv2.IMREAD_GRAYSCALE)

    # -------------------------------------------------------------
    # TASK 1: Periodic Noise & Butterworth Notch Reject Filter
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 1: Periodic Noise & Butterworth Notch Reject Filter")
    print("-" * 40)
    u0, v0 = 32, 32
    noisy_img, noise_field = add_sinusoidal_noise(lena_256, u0=u0, v0=v0, A=50.0)

    # Centered FFT of clean and noisy images
    y, x = np.ogrid[:256, :256]
    checker = (-1.0) ** (x + y)
    F_noisy = np.fft.fft2(noisy_img.astype(np.float64) * checker)
    log_mag_noisy = np.log(1.0 + np.abs(F_noisy))

    # Design Butterworth Notch Reject Filter
    H_nrf = butterworth_notch_reject_filter(256, 256, u0=u0, v0=v0, D0=12.0, n=2)
    (restored_img, t_nrf) = benchmark_timer(apply_notch_filter, noisy_img, H_nrf)

    # Metrics
    mse_noisy_orig = compute_mse(noisy_img, lena_256)
    ssim_noisy_orig = ssim(noisy_img, lena_256)
    mse_restored_orig = compute_mse(restored_img, lena_256)
    ssim_restored_orig = ssim(restored_img, lena_256)

    cv2.imwrite(os.path.join(out_dir, "HW6_Q1_noisy_sinusoidal.bmp"), noisy_img)
    cv2.imwrite(os.path.join(out_dir, "HW6_Q1_notch_restored.bmp"), restored_img)

    print(f"Noisy vs Clean Original    : MSE = {mse_noisy_orig:8.2f} | SSIM = {ssim_noisy_orig:.4f}")
    print(f"Notch Restored vs Clean   : MSE = {mse_restored_orig:8.2f} | SSIM = {ssim_restored_orig:.4f} (Filtered in {t_nrf:.2f} ms)")

    # Plot Task 1 Grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(lena_256, cmap='gray')
    axes[0, 0].set_title("Clean Original (LenaGrey256)")
    
    axes[0, 1].imshow(noisy_img, cmap='gray')
    axes[0, 1].set_title(f"Corrupted with Sinusoidal Noise (A=50)\nMSE: {mse_noisy_orig:.1f}, SSIM: {ssim_noisy_orig:.3f}")
    
    log_mag_vis = np.clip((log_mag_noisy / log_mag_noisy.max()) * 255, 0, 255).astype(np.uint8)
    axes[0, 2].imshow(log_mag_vis, cmap='inferno')
    axes[0, 2].set_title(r"Noisy Spectrum $\log(1+|F|)$" "\n" r"(Two Delta Spikes at $(\pm 32, \pm 32)$)")

    axes[1, 0].imshow(H_nrf, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(r"Butterworth Notch Reject Filter $H_{NRF}$" "\n" r"(Zeroes Spikes at $(\pm 32, \pm 32)$)")

    axes[1, 1].imshow(restored_img, cmap='gray')
    axes[1, 1].set_title(f"Notch Filter Restored Output\nMSE: {mse_restored_orig:.1f}, SSIM: {ssim_restored_orig:.3f}")

    diff_restored = np.abs(restored_img.astype(float) - lena_256.astype(float))
    axes[1, 2].imshow(diff_restored, cmap='inferno')
    axes[1, 2].set_title(f"Residual Error |Restored - Original|\n(Max Diff: {diff_restored.max():.1f})")

    for ax in axes.ravel():
        ax.axis('off')
    plt.suptitle("HW6 Task 1: Periodic Noise Removal via Butterworth Notch Reject Filtering", fontsize=14)
    plt.tight_layout()
    notch_plot_path = os.path.join(out_dir, "HW6_Q1_notch_filtering_analysis.png")
    plt.savefig(notch_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved notch filter analysis: {notch_plot_path}")

    # -------------------------------------------------------------
    # TASK 2: Homomorphic Filtering
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 2: Homomorphic Filtering")
    print("-" * 40)
    
    # Create non-uniformly illuminated / shadow-degraded test image to demonstrate illumination normalization
    Y_coords, X_coords = np.mgrid[:256, :256]
    illumination_field = 0.3 + 0.7 * np.exp(-((X_coords - 64)**2 + (Y_coords - 64)**2) / (2.0 * (100.0**2)))
    shadowed_img = np.clip(np.round(lena_256.astype(np.float64) * illumination_field), 0, 255).astype(np.uint8)

    (homo_out, H_homo) = homomorphic_filter_scratch(shadowed_img, gamma_L=0.25, gamma_H=2.0, c=1.0, D0=30.0)
    cv2.imwrite(os.path.join(out_dir, "HW6_Q2_shadowed_input.bmp"), shadowed_img)
    cv2.imwrite(os.path.join(out_dir, "HW6_Q2_homomorphic_filtered.bmp"), homo_out)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(shadowed_img, cmap='gray')
    axes[0].set_title("Input Image with Non-Uniform\nIllumination Shadow")
    
    axes[1].imshow(H_homo, cmap='viridis')
    axes[1].set_title(r"Homomorphic Transfer Function $H_{homo}$" "\n" r"($\gamma_L=0.25, \gamma_H=2.0, D_0=30$)")
    
    axes[2].imshow(homo_out, cmap='gray')
    axes[2].set_title("Homomorphic Filtered Output\n(Balanced Lighting + Crisp Detail)")
    
    axes[3].imshow(lena_256, cmap='gray')
    axes[3].set_title("Reference Clean Lena")

    for ax in axes:
        ax.axis('off')
    plt.suptitle("HW6 Task 2: Homomorphic Filtering for Illumination-Reflectance Decomposition", fontsize=14, y=1.03)
    plt.tight_layout()
    homo_plot_path = os.path.join(out_dir, "HW6_Q2_homomorphic_filtering_grid.png")
    plt.savefig(homo_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved homomorphic filtering grid: {homo_plot_path}")

    # -------------------------------------------------------------
    # TASK 3: 2D DCT-II vs FFT Energy Compaction
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 3: 2D DCT-II vs FFT Energy Compaction on 8x8 Block")
    print("-" * 40)
    
    # Extract 8x8 block from Lena's cheek/hat region (row 100..108, col 100..108)
    block_8x8 = lena_256[100:108, 100:108].copy()
    
    # Benchmark Scratch DCT vs OpenCV DCT
    dct_sc = dct2_scratch(block_8x8)
    dct_cv = dct2_opencv(block_8x8)
    mse_dct_cv = compute_mse(dct_sc, dct_cv)
    print(f"2D DCT-II Scratch vs OpenCV MSE: {mse_dct_cv:.6e} (Exact match!)")

    # Run Energy Compaction Experiment (keep 3x3 out of 8x8 = 9 out of 64 coefficients)
    compaction_res = energy_compaction_experiment(block_8x8, keep_k=3)
    mse_dct_recon = compaction_res["mse_dct"]
    mse_fft_recon = compaction_res["mse_fft"]

    print(f"Energy Compaction (Retaining 3x3 / 9 coeffs out of 64):")
    print(f"  DCT Reconstruction MSE: {mse_dct_recon:8.4f}")
    print(f"  FFT Reconstruction MSE: {mse_fft_recon:8.4f}")
    print(f"  DCT Error Advantage   : FFT error is {mse_fft_recon / (mse_dct_recon + 1e-8):.2f}x higher than DCT!")

    # Plot 8x8 Block Comparison Grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 0: Original & Coefficients
    axes[0, 0].imshow(block_8x8, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Original 8x8 Block", fontsize=11)
    
    axes[0, 1].imshow(np.log1p(np.abs(dct_sc)), cmap='inferno')
    axes[0, 1].set_title("Full 8x8 DCT Spectrum\n(Energy Concentrated Top-Left)", fontsize=10)
    
    axes[0, 2].imshow(np.log1p(np.abs(compaction_res["dct_truncated"])), cmap='inferno')
    axes[0, 2].set_title("Truncated 3x3 DCT\n(9 of 64 Coeffs Retained)", fontsize=10)

    axes[0, 3].imshow(compaction_res["recon_dct"], cmap='gray', vmin=0, vmax=255)
    axes[0, 3].set_title(f"DCT Reconstructed Block\nMSE: {mse_dct_recon:.2f}", fontsize=11)

    # Row 1: FFT comparison
    axes[1, 0].imshow(block_8x8, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title("Original 8x8 Block", fontsize=11)

    fft_mag = np.abs(compaction_res["fft_coeffs"])
    axes[1, 1].imshow(np.log1p(fft_mag), cmap='inferno')
    axes[1, 1].set_title("Full 8x8 FFT Spectrum\n(High Frequency Leakage)", fontsize=10)

    axes[1, 2].imshow(np.abs(compaction_res["recon_fft"].astype(float) - block_8x8.astype(float)), cmap='inferno')
    axes[1, 2].set_title(f"FFT Error Residual\nMSE: {mse_fft_recon:.2f}", fontsize=10)

    axes[1, 3].imshow(compaction_res["recon_fft"], cmap='gray', vmin=0, vmax=255)
    axes[1, 3].set_title(f"FFT Reconstructed Block\nMSE: {mse_fft_recon:.2f}", fontsize=11)

    for ax in axes.ravel():
        ax.axis('off')

    plt.suptitle("HW6 Task 3: 2D DCT-II vs 2D FFT Energy Compaction (JPEG Compression Basis)", fontsize=14)
    plt.tight_layout()
    dct_plot_path = os.path.join(out_dir, "HW6_Q3_dct_vs_fft_compaction.png")
    plt.savefig(dct_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved DCT vs FFT compaction analysis: {dct_plot_path}")

    print("\nHomework 6 execution completed successfully!")

if __name__ == "__main__":
    main()
