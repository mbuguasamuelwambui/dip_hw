import os
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

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
# QUESTION 1: Image Negative (s = (L - 1) - r)
# =====================================================================
def negative_scratch(img: np.ndarray) -> np.ndarray:
    """Computes image negative using pure NumPy math: s = 255 - r."""
    return (255 - img).astype(np.uint8)

def negative_opencv(img: np.ndarray) -> np.ndarray:
    """Computes image negative using OpenCV bitwise_not."""
    return cv2.bitwise_not(img)

# =====================================================================
# QUESTION 2: Gamma Transform (s = c * r^gamma, r in [0, 1])
# =====================================================================
def gamma_transform_scratch(img: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """
    Computes power-law (gamma) transform from scratch:
    s = c * (r / 255.0)^gamma * 255.0
    """
    norm = img.astype(np.float64) / 255.0
    out = c * np.power(norm, gamma) * 255.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def gamma_transform_opencv(img: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """
    Computes power-law (gamma) transform using OpenCV Look-Up Table (LUT).
    """
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        val = c * ((i / 255.0) ** gamma) * 255.0
        lut[0, i] = np.clip(round(val), 0, 255)
    return cv2.LUT(img, lut)

# =====================================================================
# QUESTION 3: Log Transform & Piecewise-Linear Contrast Stretching
# =====================================================================
def log_transform_scratch(img: np.ndarray) -> np.ndarray:
    """
    Computes log transform: s = c * log(1 + r)
    c = 255 / log(1 + 255)
    """
    r = img.astype(np.float64)
    c = 255.0 / np.log(1.0 + 255.0)
    s = c * np.log(1.0 + r)
    return np.clip(np.round(s), 0, 255).astype(np.uint8)

def log_transform_opencv(img: np.ndarray) -> np.ndarray:
    """
    Computes log transform using OpenCV LUT.
    """
    c = 255.0 / np.log(1.0 + 255.0)
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        lut[0, i] = np.clip(round(c * np.log(1.0 + i)), 0, 255)
    return cv2.LUT(img, lut)

def piecewise_linear_scratch(img: np.ndarray, r1: int, s1: int, r2: int, s2: int) -> np.ndarray:
    """
    Piecewise linear contrast stretching from scratch:
    - 0 <= r < r1: s = (s1 / r1) * r
    - r1 <= r <= r2: s = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
    - r2 < r <= 255: s = ((255 - s2) / (255 - r2)) * (r - r2) + s2
    """
    out = np.zeros_like(img, dtype=np.float64)
    r = img.astype(np.float64)

    # Segment 1
    m1 = s1 / r1 if r1 > 0 else 0
    mask1 = (r < r1)
    out[mask1] = m1 * r[mask1]

    # Segment 2
    m2 = (s2 - s1) / (r2 - r1) if (r2 - r1) != 0 else 0
    mask2 = (r >= r1) & (r <= r2)
    out[mask2] = m2 * (r[mask2] - r1) + s1

    # Segment 3
    m3 = (255.0 - s2) / (255.0 - r2) if (255.0 - r2) != 0 else 0
    mask3 = (r > r2)
    out[mask3] = m3 * (r[mask3] - r2) + s2

    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def piecewise_linear_opencv(img: np.ndarray, r1: int, s1: int, r2: int, s2: int) -> np.ndarray:
    """
    Piecewise linear contrast stretching using OpenCV LUT.
    """
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        if i < r1:
            val = (s1 / r1) * i if r1 > 0 else 0
        elif i <= r2:
            val = ((s2 - s1) / (r2 - r1)) * (i - r1) + s1 if (r2 - r1) != 0 else s1
        else:
            val = ((255.0 - s2) / (255.0 - r2)) * (i - r2) + s2 if (255.0 - r2) != 0 else 255
        lut[0, i] = np.clip(round(val), 0, 255)
    return cv2.LUT(img, lut)

# =====================================================================
# QUESTION 4: Histogram Equalization
# =====================================================================
def calc_histogram_scratch(img: np.ndarray) -> np.ndarray:
    """Calculates 1D intensity histogram of length 256 from scratch."""
    hist = np.zeros(256, dtype=np.int64)
    flat = img.ravel()
    for val in flat:
        hist[val] += 1
    return hist

def hist_equalization_scratch(img: np.ndarray):
    """
    Computes Histogram Equalization from scratch:
    - PMF: pr(rk) = nk / (M * N)
    - CDF: T(rk) = round((L - 1) * sum_{j=0}^k pr(rj))
    Returns (equalized_img, hist_orig, hist_eq, cdf_mapping)
    """
    M, N = img.shape[:2]
    total_pixels = M * N
    
    # 1. Histogram
    hist_orig = np.bincount(img.ravel(), minlength=256)
    
    # 2. PMF
    pmf = hist_orig.astype(np.float64) / total_pixels
    
    # 3. CDF
    cdf = np.cumsum(pmf)
    
    # 4. Intensity Mapping T(rk)
    mapping = np.round(255.0 * cdf).astype(np.uint8)
    
    # 5. Map image pixels
    eq_img = mapping[img]
    hist_eq = np.bincount(eq_img.ravel(), minlength=256)
    
    return eq_img, hist_orig, hist_eq, mapping

def hist_equalization_opencv(img: np.ndarray) -> np.ndarray:
    """Library standard histogram equalization using cv2.equalizeHist."""
    return cv2.equalizeHist(img)

# =====================================================================
# QUESTION 5: Histogram Matching (Specification)
# =====================================================================
def hist_matching_scratch(source_img: np.ndarray, target_hist_or_img) -> np.ndarray:
    """
    Histogram Matching from scratch:
    - Compute input CDF S(rk)
    - Compute target CDF G(zq)
    - For each input intensity rk, find zq such that G(zq) is closest to S(rk).
    - Map pixels using inverse CDF lookup.
    """
    # 1. Source CDF
    src_hist = np.bincount(source_img.ravel(), minlength=256).astype(np.float64)
    src_cdf = np.cumsum(src_hist) / source_img.size

    # 2. Target CDF
    if isinstance(target_hist_or_img, np.ndarray) and target_hist_or_img.shape == (256,):
        # Target is directly given as a 256-element distribution
        tgt_hist = target_hist_or_img.astype(np.float64)
        tgt_cdf = np.cumsum(tgt_hist) / np.sum(tgt_hist)
    else:
        # Target is a reference image
        tgt_hist = np.bincount(target_hist_or_img.ravel(), minlength=256).astype(np.float64)
        tgt_cdf = np.cumsum(tgt_hist) / target_hist_or_img.size

    # 3. Mapping LUT: find closest target intensity
    lut = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(src_cdf[src_val] - tgt_cdf)
        lut[src_val] = np.argmin(diff)

    return lut[source_img]

def hist_matching_library(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    """Library standard histogram matching via scikit-image match_histograms."""
    matched = match_histograms(source_img, reference_img)
    return matched.astype(np.uint8)


# =====================================================================
# MAIN EXECUTION & BENCHMARK SUITE
# =====================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "..", "images")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print("Homework 2: Intensity Transformations and Histogram Processing")
    print("=" * 75)

    # -------------------------------------------------------------
    # TASK 1: Image Negative (Grayscale & Color)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 1: Image Negative")
    print("-" * 40)

    # 1A. Grayscale
    gray_img = cv2.imread(os.path.join(img_dir, "LenaGrey512.bmp"), cv2.IMREAD_GRAYSCALE)
    (neg_gray_sc, t_sc_g) = benchmark_timer(negative_scratch, gray_img)
    (neg_gray_cv, t_cv_g) = benchmark_timer(negative_opencv, gray_img)
    mse_neg_g = compute_mse(neg_gray_sc, neg_gray_cv)

    cv2.imwrite(os.path.join(out_dir, "HW2_Q1_negative_gray_scratch.bmp"), neg_gray_sc)
    cv2.imwrite(os.path.join(out_dir, "HW2_Q1_negative_gray_cv.bmp"), neg_gray_cv)
    plot_4panel_comparison(gray_img, neg_gray_sc, neg_gray_cv, 
                           "HW2 Task 1: Image Negative (Grayscale)",
                           os.path.join(out_dir, "HW2_Q1_negative_gray_comparison.png"),
                           is_color=False, mse=mse_neg_g, scratch_time=t_sc_g, lib_time=t_cv_g)

    print(f"Grayscale Negative: MSE = {mse_neg_g:.6e} | Scratch: {t_sc_g:.3f} ms | OpenCV: {t_cv_g:.3f} ms")

    # 1B. Color
    color_img = cv2.imread(os.path.join(img_dir, "LenaColor512.bmp"))
    (neg_color_sc, t_sc_c) = benchmark_timer(negative_scratch, color_img)
    (neg_color_cv, t_cv_c) = benchmark_timer(negative_opencv, color_img)
    mse_neg_c = compute_mse(neg_color_sc, neg_color_cv)

    cv2.imwrite(os.path.join(out_dir, "HW2_Q1_negative_color_scratch.bmp"), neg_color_sc)
    cv2.imwrite(os.path.join(out_dir, "HW2_Q1_negative_color_cv.bmp"), neg_color_cv)
    plot_4panel_comparison(color_img, neg_color_sc, neg_color_cv, 
                           "HW2 Task 1: Image Negative (Color)",
                           os.path.join(out_dir, "HW2_Q1_negative_color_comparison.png"),
                           is_color=True, mse=mse_neg_c, scratch_time=t_sc_c, lib_time=t_cv_c)

    print(f"Color Negative    : MSE = {mse_neg_c:.6e} | Scratch: {t_sc_c:.3f} ms | OpenCV: {t_cv_c:.3f} ms")

    # -------------------------------------------------------------
    # TASK 2: Gamma Transform (gamma in [0.4, 0.67, 1.5, 2.5])
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 2: Gamma Transform")
    print("-" * 40)
    gammas = [0.4, 0.67, 1.5, 2.5]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for idx, gamma in enumerate(gammas):
        (gamma_sc, t_sc) = benchmark_timer(gamma_transform_scratch, gray_img, gamma)
        (gamma_cv, t_cv) = benchmark_timer(gamma_transform_opencv, gray_img, gamma)
        mse = compute_mse(gamma_sc, gamma_cv)

        cv2.imwrite(os.path.join(out_dir, f"HW2_Q2_gamma_{gamma}_scratch.bmp"), gamma_sc)
        cv2.imwrite(os.path.join(out_dir, f"HW2_Q2_gamma_{gamma}_cv.bmp"), gamma_cv)

        # Plot row 0: Scratch, Row 1: OpenCV
        axes[0, idx].imshow(gamma_sc, cmap='gray', vmin=0, vmax=255)
        axes[0, idx].set_title(f"Scratch: $\\gamma={gamma}$\n({t_sc:.2f} ms)", fontsize=11)
        axes[0, idx].axis('off')

        axes[1, idx].imshow(gamma_cv, cmap='gray', vmin=0, vmax=255)
        axes[1, idx].set_title(f"OpenCV: $\\gamma={gamma}$\n(MSE: {mse:.2e})", fontsize=11)
        axes[1, idx].axis('off')

        print(f"Gamma {gamma:4.2f}: MSE = {mse:.6e} | Scratch: {t_sc:.3f} ms | OpenCV: {t_cv:.3f} ms")

    plt.suptitle("HW2 Task 2: Power-Law (Gamma) Transformations Comparison", fontsize=14)
    plt.tight_layout()
    gamma_plot_path = os.path.join(out_dir, "HW2_Q2_gamma_grid.png")
    plt.savefig(gamma_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved gamma grid: {gamma_plot_path}")

    # -------------------------------------------------------------
    # TASK 3: Log Transform & Piecewise-Linear Contrast Stretching
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 3: Log Transform & Piecewise-Linear Stretching")
    print("-" * 40)

    # 3A. Log Transform
    (log_sc, t_sc_log) = benchmark_timer(log_transform_scratch, gray_img)
    (log_cv, t_cv_log) = benchmark_timer(log_transform_opencv, gray_img)
    mse_log = compute_mse(log_sc, log_cv)

    cv2.imwrite(os.path.join(out_dir, "HW2_Q3_log_scratch.bmp"), log_sc)
    cv2.imwrite(os.path.join(out_dir, "HW2_Q3_log_cv.bmp"), log_cv)
    plot_4panel_comparison(gray_img, log_sc, log_cv,
                           "HW2 Task 3: Logarithmic Transformation",
                           os.path.join(out_dir, "HW2_Q3_log_comparison.png"),
                           is_color=False, mse=mse_log, scratch_time=t_sc_log, lib_time=t_cv_log)
    print(f"Log Transform: MSE = {mse_log:.6e} | Scratch: {t_sc_log:.3f} ms | OpenCV: {t_cv_log:.3f} ms")

    # 3B. Piecewise-Linear Contrast Stretching
    # Use control points (r1, s1) = (70, 20) and (r2, s2) = (180, 235) to boost dynamic range
    r1, s1, r2, s2 = 70, 20, 180, 235
    (pw_sc, t_sc_pw) = benchmark_timer(piecewise_linear_scratch, gray_img, r1, s1, r2, s2)
    (pw_cv, t_cv_pw) = benchmark_timer(piecewise_linear_opencv, gray_img, r1, s1, r2, s2)
    mse_pw = compute_mse(pw_sc, pw_cv)

    cv2.imwrite(os.path.join(out_dir, "HW2_Q3_piecewise_scratch.bmp"), pw_sc)
    cv2.imwrite(os.path.join(out_dir, "HW2_Q3_piecewise_cv.bmp"), pw_cv)
    plot_4panel_comparison(gray_img, pw_sc, pw_cv,
                           f"HW2 Task 3: Piecewise-Linear Contrast Stretching (r1={r1},s1={s1}, r2={r2},s2={s2})",
                           os.path.join(out_dir, "HW2_Q3_piecewise_comparison.png"),
                           is_color=False, mse=mse_pw, scratch_time=t_sc_pw, lib_time=t_cv_pw)
    print(f"Piecewise Linear: MSE = {mse_pw:.6e} | Scratch: {t_sc_pw:.3f} ms | OpenCV: {t_cv_pw:.3f} ms")

    # -------------------------------------------------------------
    # TASK 4: Histogram Equalization
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 4: Histogram Equalization")
    print("-" * 40)
    lena_256 = cv2.imread(os.path.join(img_dir, "LenaGrey256.bmp"), cv2.IMREAD_GRAYSCALE)

    (res_eq_sc, t_sc_eq) = benchmark_timer(hist_equalization_scratch, lena_256)
    eq_sc, hist_orig, hist_eq_sc, cdf_mapping = res_eq_sc

    (eq_cv, t_cv_eq) = benchmark_timer(hist_equalization_opencv, lena_256)
    mse_eq = compute_mse(eq_sc, eq_cv)

    cv2.imwrite(os.path.join(out_dir, "HW2_Q4_hist_eq_scratch.bmp"), eq_sc)
    cv2.imwrite(os.path.join(out_dir, "HW2_Q4_hist_eq_cv.bmp"), eq_cv)

    print(f"Histogram Equalization: MSE = {mse_eq:.6e} | Scratch: {t_sc_eq:.3f} ms | OpenCV: {t_cv_eq:.3f} ms")

    # Plot images and their respective histograms
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    # Images
    axes[0, 0].imshow(lena_256, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f"Original LenaGrey256", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(eq_sc, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f"Scratch Equalized ({t_sc_eq:.2f} ms)", fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(eq_cv, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f"OpenCV Equalized ({t_cv_eq:.2f} ms)\nMSE: {mse_eq:.2e}", fontsize=12)
    axes[0, 2].axis('off')

    # Histograms
    axes[1, 0].bar(range(256), hist_orig, color='steelblue', width=1.0)
    axes[1, 0].set_title("Original Histogram", fontsize=11)
    axes[1, 0].set_xlim([0, 255])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(range(256), hist_eq_sc, color='forestgreen', width=1.0)
    axes[1, 1].set_title("Scratch Equalized Histogram", fontsize=11)
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].grid(True, alpha=0.3)

    hist_cv = cv2.calcHist([eq_cv], [0], None, [256], [0, 256]).ravel()
    axes[1, 2].bar(range(256), hist_cv, color='darkorange', width=1.0)
    axes[1, 2].set_title("OpenCV Equalized Histogram", fontsize=11)
    axes[1, 2].set_xlim([0, 255])
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("HW2 Task 4: Histogram Equalization and Distribution Analysis", fontsize=14)
    plt.tight_layout()
    hist_plot_path = os.path.join(out_dir, "HW2_Q4_histogram_equalization_analysis.png")
    plt.savefig(hist_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram equalization analysis: {hist_plot_path}")

    # -------------------------------------------------------------
    # TASK 5: Histogram Matching (Specification)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 5: Histogram Matching")
    print("-" * 40)

    # 5A. Target 1: Synthetic Gaussian Target (mu = 128, sigma = 40)
    x = np.arange(256)
    mu, sigma = 128.0, 40.0
    gauss_target = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    gauss_target = gauss_target / np.sum(gauss_target)  # normalized target PMF

    (match_gauss_sc, t_sc_hm1) = benchmark_timer(hist_matching_scratch, lena_256, gauss_target)
    cv2.imwrite(os.path.join(out_dir, "HW2_Q5_hist_matched_gaussian_scratch.bmp"), match_gauss_sc)
    print(f"Gaussian Target Matching (Scratch): {t_sc_hm1:.3f} ms")

    # 5B. Target 2: Natural Reference Image (LenaColor512 converted to grayscale)
    ref_color_512 = cv2.imread(os.path.join(img_dir, "LenaColor512.bmp"))
    ref_gray_512 = cv2.cvtColor(ref_color_512, cv2.COLOR_BGR2GRAY)

    (match_nat_sc, t_sc_hm2) = benchmark_timer(hist_matching_scratch, lena_256, ref_gray_512)
    (match_nat_sk, t_sk_hm2) = benchmark_timer(hist_matching_library, lena_256, ref_gray_512)
    mse_hm2 = compute_mse(match_nat_sc, match_nat_sk)

    cv2.imwrite(os.path.join(out_dir, "HW2_Q5_hist_matched_natural_scratch.bmp"), match_nat_sc)
    cv2.imwrite(os.path.join(out_dir, "HW2_Q5_hist_matched_natural_skimage.bmp"), match_nat_sk)
    print(f"Natural Reference Matching: MSE = {mse_hm2:.6e} | Scratch: {t_sc_hm2:.3f} ms | Scikit-Image: {t_sk_hm2:.3f} ms")

    # Plot comprehensive Histogram Matching results
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # Row 1: Source Lena 256
    axes[0, 0].imshow(lena_256, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Source Image: LenaGrey256 (256x256)", fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].bar(range(256), np.bincount(lena_256.ravel(), minlength=256), color='steelblue', width=1.0)
    axes[0, 1].set_title("Source Histogram", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    src_cdf = np.cumsum(np.bincount(lena_256.ravel(), minlength=256)) / lena_256.size
    axes[0, 2].plot(range(256), src_cdf, color='steelblue', lw=2)
    axes[0, 2].set_title("Source CDF", fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Gaussian Matched
    axes[1, 0].imshow(match_gauss_sc, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(rf"Matched to Gaussian ($\mu=128, \sigma=40$)" f"\nTime: {t_sc_hm1:.2f} ms", fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].bar(range(256), np.bincount(match_gauss_sc.ravel(), minlength=256), color='mediumpurple', width=1.0)
    axes[1, 1].set_title("Output Histogram (Gaussian Target)", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    gauss_cdf = np.cumsum(gauss_target)
    out_gauss_cdf = np.cumsum(np.bincount(match_gauss_sc.ravel(), minlength=256)) / match_gauss_sc.size
    axes[1, 2].plot(range(256), gauss_cdf, 'r--', label='Target Gaussian CDF', lw=2)
    axes[1, 2].plot(range(256), out_gauss_cdf, 'b-', label='Matched Output CDF', lw=1.5)
    axes[1, 2].legend(loc='lower right', fontsize=9)
    axes[1, 2].set_title("Target vs. Output CDF", fontsize=11)
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Natural Reference Matched
    axes[2, 0].imshow(match_nat_sc, cmap='gray', vmin=0, vmax=255)
    axes[2, 0].set_title(f"Matched to Natural Reference (512x512)\nMSE: {mse_hm2:.2e}", fontsize=11)
    axes[2, 0].axis('off')

    axes[2, 1].bar(range(256), np.bincount(match_nat_sc.ravel(), minlength=256), color='forestgreen', width=1.0)
    axes[2, 1].set_title("Output Histogram (Natural Reference)", fontsize=11)
    axes[2, 1].grid(True, alpha=0.3)

    nat_cdf = np.cumsum(np.bincount(ref_gray_512.ravel(), minlength=256)) / ref_gray_512.size
    out_nat_cdf = np.cumsum(np.bincount(match_nat_sc.ravel(), minlength=256)) / match_nat_sc.size
    axes[2, 2].plot(range(256), nat_cdf, 'r--', label='Reference CDF', lw=2)
    axes[2, 2].plot(range(256), out_nat_cdf, 'g-', label='Matched Output CDF', lw=1.5)
    axes[2, 2].legend(loc='lower right', fontsize=9)
    axes[2, 2].set_title("Reference vs. Output CDF", fontsize=11)
    axes[2, 2].grid(True, alpha=0.3)

    plt.suptitle("HW2 Task 5: Histogram Matching (Specification) Analysis", fontsize=14)
    plt.tight_layout()
    match_plot_path = os.path.join(out_dir, "HW2_Q5_histogram_matching_analysis.png")
    plt.savefig(match_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram matching analysis: {match_plot_path}")

    print("\nHomework 2 execution completed successfully!")

if __name__ == "__main__":
    main()
