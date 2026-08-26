import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add common directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.benchmark import compute_mse, benchmark_timer, plot_4panel_comparison

# =====================================================================
# TASK 1: 1D DFT, RADIX-2 1D FFT, AND 2D SEPARABLE TRANSFORMS
# =====================================================================
def dft_1d_scratch(x: np.ndarray) -> np.ndarray:
    """1D Discrete Fourier Transform O(N^2) from scratch."""
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return np.dot(W, x)

def idft_1d_scratch(X: np.ndarray) -> np.ndarray:
    """1D Inverse Discrete Fourier Transform O(N^2) from scratch."""
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W_inv = np.exp(2j * np.pi * k * n / N)
    return np.dot(W_inv, X) / N

def fft_1d_radix2_scratch(x: np.ndarray) -> np.ndarray:
    """
    1D Radix-2 Cooley-Tukey Fast Fourier Transform O(N log N).
    Requires len(x) to be a power of 2.
    """
    N = len(x)
    if N <= 1:
        return x.astype(np.complex128)
    if N % 2 != 0:
        return dft_1d_scratch(x)
    
    even = fft_1d_radix2_scratch(x[0::2])
    odd = fft_1d_radix2_scratch(x[1::2])
    
    k = np.arange(N // 2)
    twiddle = np.exp(-2j * np.pi * k / N) * odd
    return np.concatenate([even + twiddle, even - twiddle])

def ifft_1d_radix2_scratch(X: np.ndarray) -> np.ndarray:
    """1D Inverse Radix-2 FFT via conjugate property: IFFT(X) = conj(FFT(conj(X))) / N."""
    N = len(X)
    conj_X = np.conjugate(X)
    fft_conj = fft_1d_radix2_scratch(conj_X)
    return np.conjugate(fft_conj) / N

def fft2_scratch(img: np.ndarray, method: str = 'fft') -> np.ndarray:
    """
    2D Fourier Transform via row-column separability:
    1. 1D transform on all rows.
    2. 1D transform on all columns.
    method: 'fft' (radix-2) or 'dft' (matrix)
    """
    M, N = img.shape
    func_1d = fft_1d_radix2_scratch if method == 'fft' else dft_1d_scratch

    # Transform rows
    temp = np.zeros((M, N), dtype=np.complex128)
    for r in range(M):
        temp[r, :] = func_1d(img[r, :])
        
    # Transform columns
    out = np.zeros((M, N), dtype=np.complex128)
    for c in range(N):
        out[:, c] = func_1d(temp[:, c])
        
    return out

def ifft2_scratch(F: np.ndarray, method: str = 'fft') -> np.ndarray:
    """2D Inverse Fourier Transform via row-column separability."""
    M, N = F.shape
    ifunc_1d = ifft_1d_radix2_scratch if method == 'fft' else idft_1d_scratch

    # Inverse transform rows
    temp = np.zeros((M, N), dtype=np.complex128)
    for r in range(M):
        temp[r, :] = ifunc_1d(F[r, :])

    # Inverse transform columns
    out = np.zeros((M, N), dtype=np.complex128)
    for c in range(N):
        out[:, c] = ifunc_1d(temp[:, c])

    return out


# =====================================================================
# TASK 2: SPECTRUM, PHASE, AND RECONSTRUCTION
# =====================================================================
def get_centered_spectrum(img: np.ndarray) -> tuple:
    """
    Computes 2D FFT with spatial centering by multiplying by (-1)^(x+y).
    Returns (F_centered, log_magnitude, phase).
    """
    M, N = img.shape
    y, x = np.ogrid[:M, :N]
    checker = (-1.0) ** (x + y)
    
    centered_input = img.astype(np.float64) * checker
    F_centered = np.fft.fft2(centered_input)
    
    magnitude = np.abs(F_centered)
    log_magnitude = np.log(1.0 + magnitude)
    phase = np.angle(F_centered)
    
    return F_centered, log_magnitude, phase

def reconstruct_phase_only(phase: np.ndarray) -> np.ndarray:
    """
    Phase-only reconstruction: set |F| = 1, retain phase.
    f_phase = Re(IFFT(exp(j * phase)))
    """
    M, N = phase.shape
    y, x = np.ogrid[:M, :N]
    checker = (-1.0) ** (x + y)

    F_phase = np.exp(1j * phase)
    reconstructed = np.real(np.fft.ifft2(F_phase)) * checker
    
    # Normalize to [0, 255]
    r_min, r_max = reconstructed.min(), reconstructed.max()
    norm = (reconstructed - r_min) / (r_max - r_min + 1e-8) * 255.0
    return np.clip(np.round(norm), 0, 255).astype(np.uint8)

def reconstruct_magnitude_only(magnitude: np.ndarray) -> np.ndarray:
    """
    Magnitude-only reconstruction: set phase = 0, retain |F|.
    f_mag = Re(IFFT(|F|))
    """
    M, N = magnitude.shape
    y, x = np.ogrid[:M, :N]
    checker = (-1.0) ** (x + y)

    F_mag = magnitude.astype(np.complex128)
    reconstructed = np.real(np.fft.ifft2(F_mag)) * checker
    
    # Log-scale normalize for visualization
    r_log = np.log(1.0 + np.abs(reconstructed))
    r_min, r_max = r_log.min(), r_log.max()
    norm = (r_log - r_min) / (r_max - r_min + 1e-8) * 255.0
    return np.clip(np.round(norm), 0, 255).astype(np.uint8)


# =====================================================================
# TASK 3: FREQUENCY DOMAIN LOW-PASS FILTERS
# =====================================================================
def get_frequency_distance_grid(M: int, N: int) -> np.ndarray:
    """Returns grid of Euclidean frequency distances D(u, v) from center (M/2, N/2)."""
    u = np.arange(M) - M / 2.0
    v = np.arange(N) - N / 2.0
    V, U = np.meshgrid(v, u)
    return np.sqrt(U ** 2 + V ** 2)

def filter_ideal_lowpass(D: np.ndarray, D0: float) -> np.ndarray:
    """Ideal Low-Pass Filter: H(u,v) = 1 if D <= D0 else 0."""
    return (D <= D0).astype(np.float64)

def filter_butterworth_lowpass(D: np.ndarray, D0: float, n: int = 2) -> np.ndarray:
    """Butterworth Low-Pass Filter: H(u,v) = 1 / (1 + (D/D0)^(2n))."""
    return 1.0 / (1.0 + (D / (D0 + 1e-8)) ** (2 * n))

def filter_gaussian_lowpass(D: np.ndarray, D0: float) -> np.ndarray:
    """Gaussian Low-Pass Filter: H(u,v) = exp(-D^2 / (2 * D0^2))."""
    return np.exp(-(D ** 2) / (2.0 * (D0 ** 2) + 1e-8))

def apply_frequency_filter(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Applies frequency filter H to img via centered FFT:
    g(x, y) = Re(IFFT(F_centered * H)) * (-1)^(x+y)
    """
    M, N = img.shape
    y, x = np.ogrid[:M, :N]
    checker = (-1.0) ** (x + y)

    F_c = np.fft.fft2(img.astype(np.float64) * checker)
    G_c = F_c * H
    filtered = np.real(np.fft.ifft2(G_c)) * checker
    return np.clip(np.round(filtered), 0, 255).astype(np.uint8)


# =====================================================================
# TASK 4: HIGH-PASS & HIGH-FREQUENCY EMPHASIS (HFE)
# =====================================================================
def high_frequency_emphasis(img: np.ndarray, D0: float = 30.0, a: float = 0.5, b: float = 2.0) -> tuple:
    """
    High-Frequency Emphasis filtering:
    H_HP = 1 - H_Gaussian_LP
    H_HFE = a + b * H_HP
    Post-processes with histogram equalization.
    """
    M, N = img.shape
    D = get_frequency_distance_grid(M, N)
    
    H_LP = filter_gaussian_lowpass(D, D0)
    H_HP = 1.0 - H_LP
    H_HFE = a + b * H_HP

    # Apply pure highpass
    out_hp = apply_frequency_filter(img, H_HP)
    # Apply HFE
    out_hfe = apply_frequency_filter(img, H_HFE)
    # Apply Histogram Equalization
    out_hfe_eq = cv2.equalizeHist(out_hfe)

    return out_hp, out_hfe, out_hfe_eq, H_HP, H_HFE


# =====================================================================
# MAIN EXECUTION & BENCHMARK SUITE
# =====================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "..", "images")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print("Homework 5: Frequency Domain Analysis")
    print("=" * 75)

    lena_256 = cv2.imread(os.path.join(img_dir, "LenaGrey256.bmp"), cv2.IMREAD_GRAYSCALE)

    # -------------------------------------------------------------
    # TASK 1: DFT vs FFT Benchmark (N = 16, 32, 64, 128, 256)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 1: 1D/2D DFT vs Radix-2 FFT vs NumPy FFT Benchmark")
    print("-" * 40)
    n_sizes = [16, 32, 64, 128, 256]
    benchmark_data = []

    for N in n_sizes:
        sub_img = lena_256[:N, :N].astype(np.float64)
        
        # 1. 2D DFT Scratch
        (dft_res, t_dft) = benchmark_timer(fft2_scratch, sub_img, method='dft') if N <= 128 else (None, np.nan)
        # 2. 2D Radix-2 FFT Scratch
        (fft_res, t_fft) = benchmark_timer(fft2_scratch, sub_img, method='fft')
        # 3. NumPy FFT
        (np_res, t_np) = benchmark_timer(np.fft.fft2, sub_img)

        mse_fft_np = compute_mse(np.abs(fft_res), np.abs(np_res))
        mse_dft_np = compute_mse(np.abs(dft_res), np.abs(np_res)) if dft_res is not None else np.nan

        benchmark_data.append((N, t_dft, t_fft, t_np, mse_fft_np))
        print(f"N={N:3d}x{N:3d} | DFT: {t_dft:8.2f} ms | Radix-2 FFT: {t_fft:6.2f} ms | NumPy FFT: {t_np:5.3f} ms | FFT vs NP MSE: {mse_fft_np:.2e}")

    # Plot DFT vs FFT complexity benchmark curve
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_dft = [(N, t) for N, t, _, _, _ in benchmark_data if not np.isnan(t)]
    ax.plot([x[0] for x in valid_dft], [x[1] for x in valid_dft], 'ro-', label=r'2D DFT $O(N^3)$ (Scratch)', lw=2)
    ax.plot([x[0] for x in benchmark_data], [x[2] for x in benchmark_data], 'bs-', label=r'2D Radix-2 FFT $O(N^2 \log N)$ (Scratch)', lw=2)
    ax.plot([x[0] for x in benchmark_data], [x[3] for x in benchmark_data], 'g^-', label='NumPy C-FFT (PocketFFT)', lw=2)
    ax.set_xlabel('Matrix Dimension $N$', fontsize=11)
    ax.set_ylabel('Execution Time (ms)', fontsize=11)
    ax.set_title('HW5 Task 1: 2D Fourier Transform Algorithmic Complexity Benchmark', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    bench_plot_path = os.path.join(out_dir, "HW5_Q1_fft_dft_benchmark.png")
    plt.savefig(bench_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved benchmark plot: {bench_plot_path}")

    # -------------------------------------------------------------
    # TASK 2: Spectrum, Phase, and Reconstruction
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 2: Spectrum, Phase, and Reconstructions")
    print("-" * 40)
    F_c, log_mag, phase = get_centered_spectrum(lena_256)
    
    # Phase-only and magnitude-only reconstructions
    (recon_phase, t_phase) = benchmark_timer(reconstruct_phase_only, phase)
    (recon_mag, t_mag) = benchmark_timer(reconstruct_magnitude_only, np.abs(F_c))

    # Save images
    # Normalize log magnitude to 8-bit for saving
    log_mag_vis = np.clip(np.round((log_mag / log_mag.max()) * 255.0), 0, 255).astype(np.uint8)
    phase_vis = np.clip(np.round(((phase + np.pi) / (2.0 * np.pi)) * 255.0), 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, "HW5_Q2_log_magnitude_spectrum.bmp"), log_mag_vis)
    cv2.imwrite(os.path.join(out_dir, "HW5_Q2_phase_spectrum.bmp"), phase_vis)
    cv2.imwrite(os.path.join(out_dir, "HW5_Q2_phase_only_reconstruction.bmp"), recon_phase)
    cv2.imwrite(os.path.join(out_dir, "HW5_Q2_magnitude_only_reconstruction.bmp"), recon_mag)

    # Plot Task 2 4-panel visual comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(lena_256, cmap='gray')
    axes[0].set_title("Original (LenaGrey256)")
    axes[1].imshow(log_mag_vis, cmap='inferno')
    axes[1].set_title(r"Centered Log Spectrum $\log(1+|F|)$")
    axes[2].imshow(recon_phase, cmap='gray')
    axes[2].set_title("Phase-Only Reconstruction\n(Preserves Geometry & Edges!)")
    axes[3].imshow(recon_mag, cmap='gray')
    axes[3].set_title("Magnitude-Only Reconstruction\n(Loses Spatial Geometry)")

    for ax in axes:
        ax.axis('off')
    plt.suptitle("HW5 Task 2: Fourier Spectrum, Phase, and Spatial Reconstruction", fontsize=14, y=1.03)
    plt.tight_layout()
    phase_mag_path = os.path.join(out_dir, "HW5_Q2_phase_vs_magnitude_comparison.png")
    plt.savefig(phase_mag_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved phase vs magnitude comparison: {phase_mag_path}")

    # -------------------------------------------------------------
    # TASK 3: Frequency Domain Low-Pass Filters (ILPF, BLPF, GLPF)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 3: Ideal, Butterworth, and Gaussian Low-Pass Filters (D0 = 10, 30, 60)")
    print("-" * 40)
    D_grid = get_frequency_distance_grid(256, 256)
    d0_vals = [10, 30, 60]

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))

    for idx, D0 in enumerate(d0_vals):
        # 1. Ideal
        H_ideal = filter_ideal_lowpass(D_grid, D0)
        out_ideal = apply_frequency_filter(lena_256, H_ideal)
        cv2.imwrite(os.path.join(out_dir, f"HW5_Q3_ideal_lp_D0_{D0}.bmp"), out_ideal)

        # 2. Butterworth (n = 2)
        H_bw = filter_butterworth_lowpass(D_grid, D0, n=2)
        out_bw = apply_frequency_filter(lena_256, H_bw)
        cv2.imwrite(os.path.join(out_dir, f"HW5_Q3_butterworth_lp_D0_{D0}.bmp"), out_bw)

        # 3. Gaussian
        H_gauss = filter_gaussian_lowpass(D_grid, D0)
        out_gauss = apply_frequency_filter(lena_256, H_gauss)
        cv2.imwrite(os.path.join(out_dir, f"HW5_Q3_gaussian_lp_D0_{D0}.bmp"), out_gauss)

        # Row 0: Ideal, Row 1: Butterworth, Row 2: Gaussian
        axes[0, idx].imshow(out_ideal, cmap='gray')
        axes[0, idx].set_title(f"Ideal Low-Pass ($D_0={D0}$)\n(Visible Gibbs Ringing)")
        axes[0, idx].axis('off')

        axes[1, idx].imshow(out_bw, cmap='gray')
        axes[1, idx].set_title(f"Butterworth Low-Pass ($D_0={D0}, n=2$)\n(Minimal Ringing)")
        axes[1, idx].axis('off')

        axes[2, idx].imshow(out_gauss, cmap='gray')
        axes[2, idx].set_title(f"Gaussian Low-Pass ($D_0={D0}$)\n(Smooth / Zero Ringing)")
        axes[2, idx].axis('off')

        print(f"Computed D0={D0:2d} for Ideal, Butterworth, and Gaussian Low-Pass filters.")

    plt.suptitle("HW5 Task 3: Low-Pass Frequency Domain Filters and Ringing/Gibbs Artifacts", fontsize=14)
    plt.tight_layout()
    lpf_plot_path = os.path.join(out_dir, "HW5_Q3_lowpass_filters_grid.png")
    plt.savefig(lpf_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved low-pass filters grid: {lpf_plot_path}")

    # -------------------------------------------------------------
    # TASK 4: High-Pass & High-Frequency Emphasis (HFE)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 4: High-Frequency Emphasis (HFE) & Histogram Equalization")
    print("-" * 40)
    out_hp, out_hfe, out_hfe_eq, H_HP, H_HFE = high_frequency_emphasis(lena_256, D0=30.0, a=0.5, b=2.0)

    cv2.imwrite(os.path.join(out_dir, "HW5_Q4_highpass_D0_30.bmp"), out_hp)
    cv2.imwrite(os.path.join(out_dir, "HW5_Q4_hfe_a0.5_b2.0.bmp"), out_hfe)
    cv2.imwrite(os.path.join(out_dir, "HW5_Q4_hfe_equalized.bmp"), out_hfe_eq)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(lena_256, cmap='gray')
    axes[0].set_title("Original (LenaGrey256)")
    
    axes[1].imshow(out_hp, cmap='gray')
    axes[1].set_title(r"Pure High-Pass ($D_0=30$)" "\n(Zeroes DC Component)")
    
    axes[2].imshow(out_hfe, cmap='gray')
    axes[2].set_title(r"High-Frequency Emphasis" "\n($H_{HFE} = 0.5 + 2.0 H_{HP}$)")
    
    axes[3].imshow(out_hfe_eq, cmap='gray')
    axes[3].set_title("HFE + Histogram Equalization\n(Crisp Texture & High Contrast)")

    for ax in axes:
        ax.axis('off')
    plt.suptitle("HW5 Task 4: High-Frequency Emphasis (HFE) and Histogram Equalization", fontsize=14, y=1.03)
    plt.tight_layout()
    hfe_plot_path = os.path.join(out_dir, "HW5_Q4_high_frequency_emphasis_grid.png")
    plt.savefig(hfe_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved HFE comparison grid: {hfe_plot_path}")

    print("\nHomework 5 execution completed successfully!")

if __name__ == "__main__":
    main()
