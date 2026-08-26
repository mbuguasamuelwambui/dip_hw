import os
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

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

def plot_4panel_comparison(input_img: np.ndarray, scratch_img: np.ndarray, 
                           lib_img: np.ndarray, title: str, save_path: str,
                           is_color: bool = False, mse: float = 0.0,
                           scratch_time: float = 0.0, lib_time: float = 0.0):
    diff = np.abs(scratch_img.astype(np.float64) - lib_img.astype(np.float64))
    diff_vis = np.clip(diff * 10.0, 0, 255).astype(np.uint8) if diff.max() < 25 else np.clip(diff, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    if is_color:
        axes[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv2.cvtColor(scratch_img, cv2.COLOR_BGR2RGB))
        axes[2].imshow(cv2.cvtColor(lib_img, cv2.COLOR_BGR2RGB))
        axes[3].imshow(cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB) if len(diff_vis.shape) == 3 else diff_vis)
    else:
        axes[0].imshow(input_img, cmap='gray', vmin=0, vmax=255)
        axes[1].imshow(scratch_img, cmap='gray', vmin=0, vmax=255)
        axes[2].imshow(lib_img, cmap='gray', vmin=0, vmax=255)
        axes[3].imshow(diff_vis if len(diff_vis.shape) == 2 else diff_vis[:, :, 0], cmap='inferno')

    axes[0].set_title(f"Input Image\n({input_img.shape[0]}x{input_img.shape[1]})", fontsize=11)
    axes[1].set_title(f"Scratch Output\n({scratch_time:.3f} ms)", fontsize=11)
    axes[2].set_title(f"Library Output\n({lib_time:.3f} ms)", fontsize=11)
    axes[3].set_title(f"|Scratch - Lib| (Max Diff: {diff.max():.2f})\nMSE: {mse:.4e}", fontsize=11)

    for ax in axes:
        ax.axis('off')

    plt.suptitle(title, fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# =====================================================================
# CORE CONVOLUTION / FILTERING ENGINE (FROM SCRATCH)
# =====================================================================
def conv2d_scratch(img: np.ndarray, kernel: np.ndarray, border_mode: str = 'reflect') -> np.ndarray:
    """
    Computes 2D spatial correlation/convolution from scratch using pure NumPy sliding windows.
    Border padding matches OpenCV BORDER_REFLECT_101.
    """
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Pad image
    img_f = img.astype(np.float64)
    padded = np.pad(img_f, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # Create sliding windows view: shape (H, W, kh, kw)
    windows = sliding_window_view(padded, (kh, kw))
    
    # Element-wise product and sum over kernel dimensions
    out = np.einsum('ijkl,kl->ij', windows, kernel)
    return out

# =====================================================================
# TASK 1: LOW-PASS FILTERING (Box, Gaussian, Median)
# =====================================================================
def box_filter_scratch(img: np.ndarray, ksize: int) -> np.ndarray:
    """Applies a ksize x ksize box filter from scratch."""
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    filtered = conv2d_scratch(img, kernel)
    return np.clip(np.round(filtered), 0, 255).astype(np.uint8)

def box_filter_opencv(img: np.ndarray, ksize: int) -> np.ndarray:
    """Applies box filter using cv2.boxFilter / cv2.blur."""
    return cv2.blur(img, (ksize, ksize))

def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """Generates a normalized 2D Gaussian kernel summing to 1."""
    center = ksize // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return g / np.sum(g)

def gaussian_filter_scratch(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """Applies a 2D Gaussian filter from scratch."""
    kernel = gaussian_kernel_2d(ksize, sigma)
    filtered = conv2d_scratch(img, kernel)
    return np.clip(np.round(filtered), 0, 255).astype(np.uint8)

def gaussian_filter_opencv(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """Applies a 2D Gaussian filter using cv2.GaussianBlur."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)

def add_salt_and_pepper_noise(img: np.ndarray, noise_ratio: float = 0.10, seed: int = 42) -> np.ndarray:
    """Adds salt-and-pepper impulse noise (5% salt, 5% pepper)."""
    np.random.seed(seed)
    noisy = img.copy()
    num_noise = int(noise_ratio * img.size)
    
    # Half salt (255), half pepper (0)
    num_salt = num_noise // 2
    num_pepper = num_noise - num_salt
    
    # Salt
    coords_salt = (np.random.randint(0, img.shape[0], num_salt),
                   np.random.randint(0, img.shape[1], num_salt))
    noisy[coords_salt] = 255
    
    # Pepper
    coords_pepper = (np.random.randint(0, img.shape[0], num_pepper),
                     np.random.randint(0, img.shape[1], num_pepper))
    noisy[coords_pepper] = 0
    return noisy

def median_filter_scratch(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Applies a nonlinear median filter from scratch."""
    pad = ksize // 2
    padded = np.pad(img, pad, mode='reflect')
    windows = sliding_window_view(padded, (ksize, ksize))
    # windows shape: (H, W, ksize, ksize) -> flatten kernel dims to compute median
    flat_windows = windows.reshape(img.shape[0], img.shape[1], -1)
    med = np.median(flat_windows, axis=-1)
    return np.clip(np.round(med), 0, 255).astype(np.uint8)

def median_filter_opencv(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Applies median filter using cv2.medianBlur."""
    return cv2.medianBlur(img, ksize)


# =====================================================================
# TASK 2: HIGH-PASS / SHARPENING (Laplacian, Unsharp, Sobel)
# =====================================================================
def laplacian_sharpen_scratch(img: np.ndarray, kernel_type: str = 'K1') -> tuple:
    """
    Laplacian sharpening from scratch:
    K1: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] -> g = f - nabla^2 f
    K2: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]] -> g = f + nabla^2 f
    """
    if kernel_type == 'K1':
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        lap = conv2d_scratch(img, kernel)
        sharpened = img.astype(np.float64) - lap
    else:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
        lap = conv2d_scratch(img, kernel)
        sharpened = img.astype(np.float64) + lap

    sharpened_clipped = np.clip(np.round(sharpened), 0, 255).astype(np.uint8)
    return sharpened_clipped, lap

def laplacian_sharpen_opencv(img: np.ndarray, kernel_type: str = 'K1') -> np.ndarray:
    """Laplacian sharpening using OpenCV cv2.filter2D."""
    if kernel_type == 'K1':
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        lap = cv2.filter2D(img, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT_101)
        sharpened = img.astype(np.float64) - lap
    else:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
        lap = cv2.filter2D(img, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT_101)
        sharpened = img.astype(np.float64) + lap

    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8)

def unsharp_masking_scratch(img: np.ndarray, k: float = 1.0, ksize: int = 5, sigma: float = 1.0) -> tuple:
    """
    Unsharp masking & High-boost:
    gmask = f - fsmooth
    g = f + k * gmask
    """
    f = img.astype(np.float64)
    fsmooth = conv2d_scratch(img, gaussian_kernel_2d(ksize, sigma))
    gmask = f - fsmooth
    g = f + k * gmask
    return np.clip(np.round(g), 0, 255).astype(np.uint8), gmask, fsmooth

def unsharp_masking_opencv(img: np.ndarray, k: float = 1.0, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Unsharp masking using OpenCV."""
    f = img.astype(np.float64)
    fsmooth = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101).astype(np.float64)
    gmask = f - fsmooth
    g = f + k * gmask
    return np.clip(np.round(g), 0, 255).astype(np.uint8)

def sobel_gradients_scratch(img: np.ndarray) -> tuple:
    """
    Sobel gradient magnitude and phase from scratch:
    Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    Magnitude: M = sqrt(gx^2 + gy^2)
    Phase: alpha = arctan2(gy, gx)
    """
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = conv2d_scratch(img, Sx)
    gy = conv2d_scratch(img, Sy)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    phase = np.arctan2(gy, gx)  # radians [-pi, pi]

    mag_vis = np.clip(np.round(mag), 0, 255).astype(np.uint8)
    phase_vis = np.clip(np.round((phase + np.pi) * (255.0 / (2.0 * np.pi))), 0, 255).astype(np.uint8)

    return mag_vis, phase_vis, gx, gy, mag, phase

def sobel_gradients_opencv(img: np.ndarray) -> tuple:
    """Sobel gradient computation via OpenCV."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)

    mag, phase = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    mag_vis = np.clip(np.round(mag), 0, 255).astype(np.uint8)
    phase_vis = np.clip(np.round(phase * (255.0 / (2.0 * np.pi))), 0, 255).astype(np.uint8)

    return mag_vis, phase_vis, gx, gy


# =====================================================================
# TASK 3: COMBINED PROCESSING PIPELINE
# =====================================================================
def combined_enhancement_pipeline(img: np.ndarray) -> dict:
    """
    Gonzales & Woods combined enhancement pipeline:
    Input: Artificially scaled to [100, 150]
    (a) ga = f - nabla^2 f (Laplacian sharpening)
    (b) gb = sqrt(gx^2 + gy^2) (Sobel magnitude)
    (c) gc = box-filter(gb, 5x5)
    (d) gd = (ga * gc) normalized
    (e) ge = f + gd
    (f) gf = c * ge^0.5 (gamma transform)
    """
    # 0. Artificial contrast reduction: scale to [100, 150]
    f = 100.0 + (img.astype(np.float64) / 255.0) * 50.0

    # (a) Laplacian Sharpening
    K1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    lap = conv2d_scratch(f, K1)
    ga = f - lap

    # (b) Sobel Magnitude
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = conv2d_scratch(f, Sx)
    gy = conv2d_scratch(f, Sy)
    gb = np.sqrt(gx**2 + gy**2)

    # (c) Box Filter 5x5 on Sobel
    kernel_5x5 = np.ones((5, 5), dtype=np.float64) / 25.0
    gc = conv2d_scratch(gb, kernel_5x5)

    # (d) Mask Product: ga * gc (normalized to keep scale)
    gc_norm = gc / (np.max(gc) if np.max(gc) > 0 else 1.0)
    gd = ga * gc_norm

    # (e) ge = f + gd
    ge = f + gd

    # (f) gf = c * ge^0.5 (Gamma = 0.5)
    ge_norm = np.clip(ge, 0, 255) / 255.0
    gf = 255.0 * np.power(ge_norm, 0.5)

    return {
        "f_input": np.clip(np.round(f), 0, 255).astype(np.uint8),
        "ga_lap": np.clip(np.round(ga), 0, 255).astype(np.uint8),
        "gb_sobel": np.clip(np.round(gb), 0, 255).astype(np.uint8),
        "gc_smooth_sobel": np.clip(np.round(gc), 0, 255).astype(np.uint8),
        "gd_mask_product": np.clip(np.round(gd), 0, 255).astype(np.uint8),
        "ge_sharpened": np.clip(np.round(ge), 0, 255).astype(np.uint8),
        "gf_gamma_final": np.clip(np.round(gf), 0, 255).astype(np.uint8)
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
    print("Homework 4: Spatial Domain Filtering")
    print("=" * 75)

    lena_256 = cv2.imread(os.path.join(img_dir, "LenaGrey256.bmp"), cv2.IMREAD_GRAYSCALE)
    lena_512 = cv2.imread(os.path.join(img_dir, "LenaGrey512.bmp"), cv2.IMREAD_GRAYSCALE)

    # -------------------------------------------------------------
    # TASK 1: Low-pass Filtering (Box, Gaussian, Median)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 1A: Box Filters (3x3, 5x5, 9x9)")
    print("-" * 40)
    for ksize in [3, 5, 9]:
        (box_sc, t_box_sc) = benchmark_timer(box_filter_scratch, lena_256, ksize)
        (box_cv, t_box_cv) = benchmark_timer(box_filter_opencv, lena_256, ksize)
        mse_box = compute_mse(box_sc, box_cv)

        cv2.imwrite(os.path.join(out_dir, f"HW4_Q1_box_{ksize}x{ksize}_scratch.bmp"), box_sc)
        cv2.imwrite(os.path.join(out_dir, f"HW4_Q1_box_{ksize}x{ksize}_cv.bmp"), box_cv)
        print(f"Box {ksize}x{ksize}: MSE = {mse_box:.6e} | Scratch: {t_box_sc:.3f} ms | OpenCV: {t_box_cv:.3f} ms")

    print("\n" + "-" * 40)
    print("Task 1B: Gaussian Filters (5x5 sig=1.0, 11x11 sig=2.5)")
    print("-" * 40)
    gauss_configs = [(5, 1.0), (11, 2.5)]
    for ksize, sigma in gauss_configs:
        (g_sc, t_g_sc) = benchmark_timer(gaussian_filter_scratch, lena_256, ksize, sigma)
        (g_cv, t_g_cv) = benchmark_timer(gaussian_filter_opencv, lena_256, ksize, sigma)
        mse_g = compute_mse(g_sc, g_cv)

        cv2.imwrite(os.path.join(out_dir, f"HW4_Q1_gauss_{ksize}x{ksize}_sig{sigma}_scratch.bmp"), g_sc)
        cv2.imwrite(os.path.join(out_dir, f"HW4_Q1_gauss_{ksize}x{ksize}_sig{sigma}_cv.bmp"), g_cv)
        print(f"Gaussian {ksize}x{ksize} (sigma={sigma}): MSE = {mse_g:.6e} | Scratch: {t_g_sc:.3f} ms | OpenCV: {t_g_cv:.3f} ms")

    print("\n" + "-" * 40)
    print("Task 1C: Median Filter vs Mean Filter (10% Salt & Pepper Noise)")
    print("-" * 40)
    noisy_img = add_salt_and_pepper_noise(lena_256, noise_ratio=0.10)
    cv2.imwrite(os.path.join(out_dir, "HW4_Q1_noisy_sp10pct.bmp"), noisy_img)

    # Apply 3x3 mean filter and 3x3 median filter
    (mean_filtered, t_mean) = benchmark_timer(box_filter_scratch, noisy_img, 3)
    (med_sc, t_med_sc) = benchmark_timer(median_filter_scratch, noisy_img, 3)
    (med_cv, t_med_cv) = benchmark_timer(median_filter_opencv, noisy_img, 3)

    mse_noisy_orig = compute_mse(noisy_img, lena_256)
    mse_mean_orig = compute_mse(mean_filtered, lena_256)
    mse_med_orig = compute_mse(med_sc, lena_256)
    mse_med_scratch_cv = compute_mse(med_sc, med_cv)

    cv2.imwrite(os.path.join(out_dir, "HW4_Q1_mean_denoised.bmp"), mean_filtered)
    cv2.imwrite(os.path.join(out_dir, "HW4_Q1_median_denoised_scratch.bmp"), med_sc)
    cv2.imwrite(os.path.join(out_dir, "HW4_Q1_median_denoised_cv.bmp"), med_cv)

    print(f"Noisy vs Clean Original MSE   : {mse_noisy_orig:8.2f}")
    print(f"Mean Filter vs Clean Original : {mse_mean_orig:8.2f} (smears impulse noise)")
    print(f"Median Filter vs Clean Original: {mse_med_orig:8.2f} (successfully removes salt & pepper)")
    print(f"Median Scratch vs OpenCV MSE   : {mse_med_scratch_cv:.6e} | Scratch: {t_med_sc:.3f} ms | OpenCV: {t_med_cv:.3f} ms")

    # Plot Task 1 Lowpass Grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(lena_256, cmap='gray')
    axes[0, 0].set_title("Original (LenaGrey256)")
    axes[0, 1].imshow(box_filter_scratch(lena_256, 3), cmap='gray')
    axes[0, 1].set_title("Box Filter 3x3")
    axes[0, 2].imshow(box_filter_scratch(lena_256, 5), cmap='gray')
    axes[0, 2].set_title("Box Filter 5x5")
    axes[0, 3].imshow(box_filter_scratch(lena_256, 9), cmap='gray')
    axes[0, 3].set_title("Box Filter 9x9")

    axes[1, 0].imshow(gaussian_filter_scratch(lena_256, 5, 1.0), cmap='gray')
    axes[1, 0].set_title(r"Gaussian 5x5 ($\sigma=1.0$)")
    axes[1, 1].imshow(gaussian_filter_scratch(lena_256, 11, 2.5), cmap='gray')
    axes[1, 1].set_title(r"Gaussian 11x11 ($\sigma=2.5$)")
    axes[1, 2].imshow(noisy_img, cmap='gray')
    axes[1, 2].set_title(f"10% Salt & Pepper Noise\nMSE: {mse_noisy_orig:.1f}")
    axes[1, 3].imshow(med_sc, cmap='gray')
    axes[1, 3].set_title(f"Median 3x3 Denoised\nMSE to Orig: {mse_med_orig:.1f}")

    for ax in axes.ravel():
        ax.axis('off')
    plt.suptitle("HW4 Task 1: Low-Pass and Nonlinear Filtering Comparison", fontsize=14)
    plt.tight_layout()
    lowpass_path = os.path.join(out_dir, "HW4_Q1_lowpass_filtering_grid.png")
    plt.savefig(lowpass_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved low-pass grid: {lowpass_path}")

    # -------------------------------------------------------------
    # TASK 2: High-pass / Sharpening (Laplacian, Unsharp, Sobel)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 2A: Laplacian Sharpening (K1 & K2)")
    print("-" * 40)
    for k_name in ['K1', 'K2']:
        (lap_sc_res, t_lap_sc) = benchmark_timer(laplacian_sharpen_scratch, lena_256, k_name)
        lap_sc, lap_field = lap_sc_res
        (lap_cv, t_lap_cv) = benchmark_timer(laplacian_sharpen_opencv, lena_256, k_name)
        mse_lap = compute_mse(lap_sc, lap_cv)

        cv2.imwrite(os.path.join(out_dir, f"HW4_Q2_laplacian_{k_name}_scratch.bmp"), lap_sc)
        cv2.imwrite(os.path.join(out_dir, f"HW4_Q2_laplacian_{k_name}_cv.bmp"), lap_cv)
        print(f"Laplacian {k_name}: MSE = {mse_lap:.6e} | Scratch: {t_lap_sc:.3f} ms | OpenCV: {t_lap_cv:.3f} ms")

    print("\n" + "-" * 40)
    print("Task 2B: Unsharp Masking & High-Boost (k=1.0, k=4.5)")
    print("-" * 40)
    for k_val, label in [(1.0, "unsharp_k1.0"), (4.5, "highboost_k4.5")]:
        (unsharp_sc_res, t_un_sc) = benchmark_timer(unsharp_masking_scratch, lena_256, k_val)
        un_sc, gmask, fsmooth = unsharp_sc_res
        (un_cv, t_un_cv) = benchmark_timer(unsharp_masking_opencv, lena_256, k_val)
        mse_un = compute_mse(un_sc, un_cv)

        cv2.imwrite(os.path.join(out_dir, f"HW4_Q2_{label}_scratch.bmp"), un_sc)
        cv2.imwrite(os.path.join(out_dir, f"HW4_Q2_{label}_cv.bmp"), un_cv)
        print(f"Unsharp Masking (k={k_val:3.1f}): MSE = {mse_un:.6e} | Scratch: {t_un_sc:.3f} ms | OpenCV: {t_un_cv:.3f} ms")

    print("\n" + "-" * 40)
    print("Task 2C: Sobel Gradient Magnitude & Phase")
    print("-" * 40)
    (sobel_sc_res, t_sob_sc) = benchmark_timer(sobel_gradients_scratch, lena_256)
    mag_sc, phase_sc, gx_sc, gy_sc, raw_mag_sc, raw_phase_sc = sobel_sc_res
    (sobel_cv_res, t_sob_cv) = benchmark_timer(sobel_gradients_opencv, lena_256)
    mag_cv, phase_cv, gx_cv, gy_cv = sobel_cv_res
    mse_mag = compute_mse(mag_sc, mag_cv)

    cv2.imwrite(os.path.join(out_dir, "HW4_Q2_sobel_mag_scratch.bmp"), mag_sc)
    cv2.imwrite(os.path.join(out_dir, "HW4_Q2_sobel_phase_scratch.bmp"), phase_sc)
    print(f"Sobel Magnitude: MSE = {mse_mag:.6e} | Scratch: {t_sob_sc:.3f} ms | OpenCV: {t_sob_cv:.3f} ms")

    # Plot Task 2 Sharpening Grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(lena_256, cmap='gray')
    axes[0, 0].set_title("Original (LenaGrey256)")
    axes[0, 1].imshow(laplacian_sharpen_scratch(lena_256, 'K1')[0], cmap='gray')
    axes[0, 1].set_title("Laplacian Sharpened (K1)")
    axes[0, 2].imshow(laplacian_sharpen_scratch(lena_256, 'K2')[0], cmap='gray')
    axes[0, 2].set_title("Laplacian Sharpened (K2 - 8-neighbor)")
    axes[0, 3].imshow(unsharp_masking_scratch(lena_256, 1.0)[0], cmap='gray')
    axes[0, 3].set_title(r"Unsharp Masking ($k=1.0$)")

    axes[1, 0].imshow(unsharp_masking_scratch(lena_256, 4.5)[0], cmap='gray')
    axes[1, 0].set_title(r"High-Boost Filtering ($k=4.5$)")
    axes[1, 1].imshow(np.abs(gx_sc), cmap='gray')
    axes[1, 1].set_title(r"Sobel $|G_x|$ (Vertical Edges)")
    axes[1, 2].imshow(mag_sc, cmap='gray')
    axes[1, 2].set_title(r"Sobel Magnitude $M=\sqrt{g_x^2+g_y^2}$")
    axes[1, 3].imshow(phase_sc, cmap='hsv')
    axes[1, 3].set_title(r"Sobel Phase $\alpha=\arctan(g_y/g_x)$")

    for ax in axes.ravel():
        ax.axis('off')
    plt.suptitle("HW4 Task 2: High-Pass, Sharpening, and Gradient Filtering", fontsize=14)
    plt.tight_layout()
    sharpening_path = os.path.join(out_dir, "HW4_Q2_sharpening_grid.png")
    plt.savefig(sharpening_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved sharpening grid: {sharpening_path}")

    # -------------------------------------------------------------
    # TASK 3: Combined Processing Pipeline
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 3: Combined Enhancement Pipeline")
    print("-" * 40)
    (pipeline_res, t_pipe) = benchmark_timer(combined_enhancement_pipeline, lena_256)

    for stage_name, stage_img in pipeline_res.items():
        cv2.imwrite(os.path.join(out_dir, f"HW4_Q3_{stage_name}.bmp"), stage_img)
    print(f"Combined Pipeline executed in: {t_pipe:.3f} ms")

    # Plot 6-Stage Combined Pipeline
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    stages = [
        ("f_input", r"(0) Input (Contrast-Reduced [100, 150])"),
        ("ga_lap", r"(a) $g_a = f - \nabla^2 f$ (Laplacian Sharpened)"),
        ("gb_sobel", r"(b) $g_b = \sqrt{g_x^2 + g_y^2}$ (Sobel Magnitude)"),
        ("gc_smooth_sobel", r"(c) $g_c = $ BoxFilter 5x5 on $g_b$ (Smoothed Edge Mask)"),
        ("gd_mask_product", r"(d) $g_d = g_a \cdot g_c$ (Product Mask)"),
        ("gf_gamma_final", r"(f) $g_f = c \cdot g_e^{0.5}$ (Final Gamma Enhanced)")
    ]

    for idx, (k, title) in enumerate(stages):
        r = idx // 3
        c = idx % 3
        axes[r, c].imshow(pipeline_res[k], cmap='gray', vmin=0, vmax=255)
        axes[r, c].set_title(title, fontsize=11)
        axes[r, c].axis('off')

    plt.suptitle("HW4 Task 3: Multi-Stage Combined Image Enhancement Pipeline", fontsize=14)
    plt.tight_layout()
    combined_path = os.path.join(out_dir, "HW4_Q3_combined_pipeline_grid.png")
    plt.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved combined pipeline grid: {combined_path}")

    print("\nHomework 4 execution completed successfully!")

if __name__ == "__main__":
    main()
