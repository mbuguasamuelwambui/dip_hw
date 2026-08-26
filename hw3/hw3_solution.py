import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add common directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.benchmark import compute_mse, benchmark_timer, plot_4panel_comparison

# =====================================================================
# TASK 1: Spatial Resolution (Downsampling & Upsampling)
# =====================================================================
def spatial_downsample_scratch(img: np.ndarray, target_k: int) -> np.ndarray:
    """Downsamples a 512x512 image to 2^k x 2^k using direct stride slicing."""
    target_size = 2 ** target_k
    step = 512 // target_size
    return img[::step, ::step].copy()

def spatial_downsample_opencv(img: np.ndarray, target_k: int) -> np.ndarray:
    """Downsamples using OpenCV nearest-neighbor resize."""
    target_size = 2 ** target_k
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

def spatial_upsample_scratch(img: np.ndarray, out_size: int = 512) -> np.ndarray:
    """Upsamples 2^k x 2^k back to out_size x out_size via pixel replication."""
    in_size = img.shape[0]
    factor = out_size // in_size
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)

def spatial_upsample_opencv(img: np.ndarray, out_size: int = 512) -> np.ndarray:
    """Upsamples using OpenCV nearest-neighbor resize."""
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


# =====================================================================
# TASK 2: Intensity Resolution (Quantization to 2^k levels)
# =====================================================================
def quantize_intensity_scratch(img: np.ndarray, k: int) -> np.ndarray:
    """
    Quantizes an 8-bit image to 2^k discrete intensity levels from scratch.
    Scales quantized levels across the full [0, 255] dynamic range.
    """
    if k == 8:
        return img.copy()
    levels = 2 ** k
    step = 256.0 / levels
    quant_idx = np.floor(img.astype(np.float64) / step)
    reconstructed = np.round(quant_idx * (255.0 / (levels - 1)))
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def quantize_intensity_opencv(img: np.ndarray, k: int) -> np.ndarray:
    """
    Quantizes an 8-bit image to 2^k discrete intensity levels using OpenCV LUT.
    """
    if k == 8:
        return img.copy()
    levels = 2 ** k
    step = 256.0 / levels
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        idx = int(np.floor(i / step))
        reconstructed = round(idx * (255.0 / (levels - 1)))
        lut[0, i] = np.clip(reconstructed, 0, 255)
    return cv2.LUT(img, lut)


# =====================================================================
# TASK 3: Interpolation Methods (Nearest, Bilinear, Bicubic, Lanczos)
# =====================================================================
def interp_nearest_scratch(img: np.ndarray, scale: float) -> np.ndarray:
    """Nearest neighbor interpolation from scratch."""
    H, W = img.shape[:2]
    out_H, out_W = int(round(H * scale)), int(round(W * scale))
    
    dst_y, dst_x = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')
    src_y = np.clip(np.floor((dst_y + 0.5) / scale).astype(int), 0, H - 1)
    src_x = np.clip(np.floor((dst_x + 0.5) / scale).astype(int), 0, W - 1)
    
    return img[src_y, src_x]

def interp_bilinear_scratch(img: np.ndarray, scale: float) -> np.ndarray:
    """Bilinear interpolation from scratch matching OpenCV geometric center convention."""
    H, W = img.shape[:2]
    out_H, out_W = int(round(H * scale)), int(round(W * scale))
    is_color = (img.ndim == 3)

    dst_y, dst_x = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')
    src_y = (dst_y + 0.5) / scale - 0.5
    src_x = (dst_x + 0.5) / scale - 0.5

    x1 = np.floor(src_x).astype(int)
    y1 = np.floor(src_y).astype(int)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = src_x - x1
    dy = src_y - y1

    x1_c = np.clip(x1, 0, W - 1)
    x2_c = np.clip(x2, 0, W - 1)
    y1_c = np.clip(y1, 0, H - 1)
    y2_c = np.clip(y2, 0, H - 1)

    if not is_color:
        Ia = img[y1_c, x1_c]
        Ib = img[y1_c, x2_c]
        Ic = img[y2_c, x1_c]
        Id = img[y2_c, x2_c]
        out = (Ia * (1 - dx) * (1 - dy) +
               Ib * dx * (1 - dy) +
               Ic * (1 - dx) * dy +
               Id * dx * dy)
    else:
        out = np.zeros((out_H, out_W, img.shape[2]), dtype=np.float64)
        for c in range(img.shape[2]):
            Ia = img[y1_c, x1_c, c]
            Ib = img[y1_c, x2_c, c]
            Ic = img[y2_c, x1_c, c]
            Id = img[y2_c, x2_c, c]
            out[:, :, c] = (Ia * (1 - dx) * (1 - dy) +
                            Ib * dx * (1 - dy) +
                            Ic * (1 - dx) * dy +
                            Id * dx * dy)

    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def bicubic_kernel(x: np.ndarray, a: float = -0.75) -> np.ndarray:
    """Keys' cubic convolution kernel matching OpenCV (a = -0.75)."""
    abs_x = np.abs(x)
    abs_x2 = abs_x ** 2
    abs_x3 = abs_x ** 3
    
    w = np.zeros_like(x, dtype=np.float64)
    mask1 = (abs_x <= 1.0)
    mask2 = (abs_x > 1.0) & (abs_x < 2.0)
    
    w[mask1] = (a + 2.0) * abs_x3[mask1] - (a + 3.0) * abs_x2[mask1] + 1.0
    w[mask2] = a * abs_x3[mask2] - 5.0 * a * abs_x2[mask2] + 8.0 * a * abs_x[mask2] - 4.0 * a
    return w

def interp_bicubic_scratch(img: np.ndarray, scale: float) -> np.ndarray:
    """Bicubic interpolation from scratch using 4x4 neighborhood convolution."""
    H, W = img.shape[:2]
    out_H, out_W = int(round(H * scale)), int(round(W * scale))
    is_color = (img.ndim == 3)

    dst_y, dst_x = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')
    src_y = (dst_y + 0.5) / scale - 0.5
    src_x = (dst_x + 0.5) / scale - 0.5

    x_base = np.floor(src_x).astype(int)
    y_base = np.floor(src_y).astype(int)

    out = np.zeros((out_H, out_W) if not is_color else (out_H, out_W, img.shape[2]), dtype=np.float64)
    total_weights = np.zeros((out_H, out_W), dtype=np.float64)

    for m in range(-1, 3):
        wy = bicubic_kernel(src_y - (y_base + m))
        y_coord = np.clip(y_base + m, 0, H - 1)
        for n in range(-1, 3):
            wx = bicubic_kernel(src_x - (x_base + n))
            x_coord = np.clip(x_base + n, 0, W - 1)
            weight = wy * wx
            total_weights += weight
            
            if not is_color:
                out += img[y_coord, x_coord] * weight
            else:
                for c in range(img.shape[2]):
                    out[:, :, c] += img[y_coord, x_coord, c] * weight

    if not is_color:
        out = out / total_weights
    else:
        for c in range(img.shape[2]):
            out[:, :, c] = out[:, :, c] / total_weights

    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def lanczos_kernel(x: np.ndarray, a: int = 4) -> np.ndarray:
    """Lanczos-4 windowed sinc kernel."""
    abs_x = np.abs(x)
    w = np.zeros_like(x, dtype=np.float64)
    zero_mask = (abs_x < 1e-7)
    w[zero_mask] = 1.0

    valid_mask = (abs_x >= 1e-7) & (abs_x < a)
    xv = abs_x[valid_mask]
    w[valid_mask] = (a * np.sin(np.pi * xv) * np.sin(np.pi * xv / a)) / ((np.pi * xv) ** 2)
    return w

def interp_lanczos_scratch(img: np.ndarray, scale: float, a: int = 4) -> np.ndarray:
    """Lanczos-4 interpolation from scratch."""
    H, W = img.shape[:2]
    out_H, out_W = int(round(H * scale)), int(round(W * scale))
    is_color = (img.ndim == 3)

    dst_y, dst_x = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')
    src_y = (dst_y + 0.5) / scale - 0.5
    src_x = (dst_x + 0.5) / scale - 0.5

    x_base = np.floor(src_x).astype(int)
    y_base = np.floor(src_y).astype(int)

    out = np.zeros((out_H, out_W) if not is_color else (out_H, out_W, img.shape[2]), dtype=np.float64)
    total_weights = np.zeros((out_H, out_W), dtype=np.float64)

    for m in range(-a + 1, a + 1):
        wy = lanczos_kernel(src_y - (y_base + m), a)
        y_coord = np.clip(y_base + m, 0, H - 1)
        for n in range(-a + 1, a + 1):
            wx = lanczos_kernel(src_x - (x_base + n), a)
            x_coord = np.clip(x_base + n, 0, W - 1)
            weight = wy * wx
            total_weights += weight

            if not is_color:
                out += img[y_coord, x_coord] * weight
            else:
                for c in range(img.shape[2]):
                    out[:, :, c] += img[y_coord, x_coord, c] * weight

    safe_weights = np.where(total_weights == 0, 1.0, total_weights)
    if not is_color:
        out = out / safe_weights
    else:
        for c in range(img.shape[2]):
            out[:, :, c] = out[:, :, c] / safe_weights

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# =====================================================================
# TASK 4: Geometric Transformations via Backward Mapping
# =====================================================================
def backward_warp_bilinear_scratch(img: np.ndarray, inv_matrix_2x3: np.ndarray, out_shape: tuple) -> np.ndarray:
    """
    General backward mapping engine with bilinear interpolation & explicit zero-padding boundary handling.
    inv_matrix_2x3 maps destination coords (xd, yd) -> source coords (xs, ys):
    xs = M_inv[0,0]*xd + M_inv[0,1]*yd + M_inv[0,2]
    ys = M_inv[1,0]*xd + M_inv[1,1]*yd + M_inv[1,2]
    """
    out_H, out_W = out_shape
    in_H, in_W = img.shape[:2]
    is_color = (img.ndim == 3)

    yd, xd = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')
    xs = inv_matrix_2x3[0, 0] * xd + inv_matrix_2x3[0, 1] * yd + inv_matrix_2x3[0, 2]
    ys = inv_matrix_2x3[1, 0] * xd + inv_matrix_2x3[1, 1] * yd + inv_matrix_2x3[1, 2]

    # In-bounds mask (OpenCV boundary convention)
    valid_mask = (xs >= 0) & (xs <= in_W - 1) & (ys >= 0) & (ys <= in_H - 1)

    x1 = np.floor(xs).astype(int)
    y1 = np.floor(ys).astype(int)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = xs - x1
    dy = ys - y1

    x1_c = np.clip(x1, 0, in_W - 1)
    x2_c = np.clip(x2, 0, in_W - 1)
    y1_c = np.clip(y1, 0, in_H - 1)
    y2_c = np.clip(y2, 0, in_H - 1)

    if not is_color:
        out = np.zeros((out_H, out_W), dtype=np.float64)
        Ia = img[y1_c, x1_c]
        Ib = img[y1_c, x2_c]
        Ic = img[y2_c, x1_c]
        Id = img[y2_c, x2_c]
        interpolated = (Ia * (1 - dx) * (1 - dy) +
                        Ib * dx * (1 - dy) +
                        Ic * (1 - dx) * dy +
                        Id * dx * dy)
        out[valid_mask] = interpolated[valid_mask]
    else:
        out = np.zeros((out_H, out_W, img.shape[2]), dtype=np.float64)
        for c in range(img.shape[2]):
            Ia = img[y1_c, x1_c, c]
            Ib = img[y1_c, x2_c, c]
            Ic = img[y2_c, x1_c, c]
            Id = img[y2_c, x2_c, c]
            interpolated = (Ia * (1 - dx) * (1 - dy) +
                            Ib * dx * (1 - dy) +
                            Ic * (1 - dx) * dy +
                            Id * dx * dy)
            out[valid_mask, c] = interpolated[valid_mask]

    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def transform_translation_scratch(img: np.ndarray, tx: float = 40, ty: float = 40) -> np.ndarray:
    """Backward mapping translation by (tx, ty)."""
    inv_M = np.array([
        [1.0, 0.0, -tx],
        [0.0, 1.0, -ty]
    ], dtype=np.float64)
    return backward_warp_bilinear_scratch(img, inv_M, (img.shape[0], img.shape[1]))

def transform_scaling_scratch(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Backward mapping scaling by (sx, sy)."""
    inv_M = np.array([
        [1.0 / sx, 0.0, 0.0],
        [0.0, 1.0 / sy, 0.0]
    ], dtype=np.float64)
    out_H = int(round(img.shape[0] * sy))
    out_W = int(round(img.shape[1] * sx))
    return backward_warp_bilinear_scratch(img, inv_M, (out_H, out_W))

def transform_rotation_scratch(img: np.ndarray, angle_deg: float = 45.0) -> np.ndarray:
    """Backward mapping rotation by angle_deg around image center."""
    H, W = img.shape[:2]
    # Forward rotation matrix matching cv2.getRotationMatrix2D
    M_fwd = cv2.getRotationMatrix2D(( (W - 1) / 2.0, (H - 1) / 2.0 ), angle_deg, 1.0)
    inv_M = cv2.invertAffineTransform(M_fwd)
    return backward_warp_bilinear_scratch(img, inv_M, (H, W))


# =====================================================================
# MAIN EXECUTION & BENCHMARK SUITE
# =====================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "..", "images")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print("Homework 3: Resolution, Interpolation, and Geometric Transformations")
    print("=" * 75)

    lena_color = cv2.imread(os.path.join(img_dir, "LenaColor512.bmp"))
    lena_gray = cv2.imread(os.path.join(img_dir, "LenaGrey512.bmp"), cv2.IMREAD_GRAYSCALE)

    # -------------------------------------------------------------
    # TASK 1: Spatial Resolution (2^k x 2^k for k = 8, 7, 6, 5, 4)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 1: Spatial Resolution Downsampling & Upsampling")
    print("-" * 40)
    k_vals = [8, 7, 6, 5, 4]
    fig, axes = plt.subplots(2, len(k_vals), figsize=(18, 7.5))

    for idx, k in enumerate(k_vals):
        size = 2 ** k
        (down_sc, t_down_sc) = benchmark_timer(spatial_downsample_scratch, lena_color, k)
        (down_cv, t_down_cv) = benchmark_timer(spatial_downsample_opencv, lena_color, k)
        mse_down = compute_mse(down_sc, down_cv)

        (up_sc, t_up_sc) = benchmark_timer(spatial_upsample_scratch, down_sc, 512)
        (up_cv, t_up_cv) = benchmark_timer(spatial_upsample_opencv, down_cv, 512)
        mse_up = compute_mse(up_sc, up_cv)

        cv2.imwrite(os.path.join(out_dir, f"HW3_Q1_down_k{k}_{size}x{size}.bmp"), down_sc)
        cv2.imwrite(os.path.join(out_dir, f"HW3_Q1_upsampled_k{k}_512x512.bmp"), up_sc)

        axes[0, idx].imshow(cv2.cvtColor(down_sc, cv2.COLOR_BGR2RGB))
        axes[0, idx].set_title(f"$2^{k} \\times 2^{k}$ ({size}$\\times${size})\n(MSE: {mse_down:.1e})", fontsize=11)
        axes[0, idx].axis('off')

        axes[1, idx].imshow(cv2.cvtColor(up_sc, cv2.COLOR_BGR2RGB))
        axes[1, idx].set_title(f"Re-upsampled to $512\\times 512$\n(Pixelation factor: {512//size}$\\times$)", fontsize=10)
        axes[1, idx].axis('off')

        print(f"k={k} ({size:3d}x{size:3d}): Downsample MSE = {mse_down:.2e} | Upsample MSE = {mse_up:.2e}")

    plt.suptitle("HW3 Task 1: Spatial Resolution Reduction and Pixelation Progression", fontsize=14)
    plt.tight_layout()
    spatial_plot_path = os.path.join(out_dir, "HW3_Q1_spatial_resolution_grid.png")
    plt.savefig(spatial_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial resolution grid: {spatial_plot_path}")

    # -------------------------------------------------------------
    # TASK 2: Intensity Resolution (2^k levels for k = 8, 7, ..., 1)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 2: Intensity Resolution Quantization (k = 8..1)")
    print("-" * 40)
    k_levels = list(range(8, 0, -1))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))

    for idx, k in enumerate(k_levels):
        row = idx // 4
        col = idx % 4
        levels = 2 ** k
        (q_sc, t_q_sc) = benchmark_timer(quantize_intensity_scratch, lena_gray, k)
        (q_cv, t_q_cv) = benchmark_timer(quantize_intensity_opencv, lena_gray, k)
        mse_q = compute_mse(q_sc, q_cv)

        cv2.imwrite(os.path.join(out_dir, f"HW3_Q2_quantized_k{k}_{levels}levels.bmp"), q_sc)

        axes[row, col].imshow(q_sc, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f"$k={k}$ ({levels} levels)\nMSE: {mse_q:.2e} | {t_q_sc:.2f} ms", fontsize=11)
        axes[row, col].axis('off')

        print(f"k={k:1d} ({levels:3d} levels): MSE = {mse_q:.6e} | Scratch: {t_q_sc:.3f} ms | OpenCV: {t_q_cv:.3f} ms")

    plt.suptitle("HW3 Task 2: Intensity Resolution Quantization and False Contouring", fontsize=14)
    plt.tight_layout()
    quant_plot_path = os.path.join(out_dir, "HW3_Q2_intensity_resolution_grid.png")
    plt.savefig(quant_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved intensity resolution grid: {quant_plot_path}")

    # -------------------------------------------------------------
    # TASK 3: Interpolation Comparison (Crop 64x64 ROI -> 2x and 4x)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 3: Interpolation Methods Benchmark (64x64 ROI -> 2x, 4x)")
    print("-" * 40)
    roi_64 = lena_gray[240:304, 240:304].copy()
    cv2.imwrite(os.path.join(out_dir, "HW3_Q3_roi_64x64.bmp"), roi_64)

    methods = [
        ("Nearest Neighbor", interp_nearest_scratch, cv2.INTER_NEAREST),
        ("Bilinear", interp_bilinear_scratch, cv2.INTER_LINEAR),
        ("Bicubic", interp_bicubic_scratch, cv2.INTER_CUBIC),
        ("Lanczos-4", interp_lanczos_scratch, cv2.INTER_LANCZOS4)
    ]

    for scale in [2.0, 4.0]:
        target_size = int(64 * scale)
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))

        for idx, (name, sc_func, cv_flag) in enumerate(methods):
            (out_sc, t_sc) = benchmark_timer(sc_func, roi_64, scale)
            (out_cv, t_cv) = benchmark_timer(cv2.resize, roi_64, (target_size, target_size), interpolation=cv_flag)
            mse = compute_mse(out_sc, out_cv)

            cv2.imwrite(os.path.join(out_dir, f"HW3_Q3_interp_{name.replace(' ', '_').lower()}_{int(scale)}x_scratch.bmp"), out_sc)
            cv2.imwrite(os.path.join(out_dir, f"HW3_Q3_interp_{name.replace(' ', '_').lower()}_{int(scale)}x_cv.bmp"), out_cv)

            axes[0, idx].imshow(out_sc, cmap='gray', vmin=0, vmax=255)
            axes[0, idx].set_title(f"Scratch: {name}\n({t_sc:.2f} ms)", fontsize=11)
            axes[0, idx].axis('off')

            axes[1, idx].imshow(out_cv, cmap='gray', vmin=0, vmax=255)
            axes[1, idx].set_title(f"OpenCV: {name}\n({t_cv:.2f} ms, MSE: {mse:.2f})", fontsize=11)
            axes[1, idx].axis('off')

            print(f"Scale {scale}x | {name:16s}: MSE = {mse:8.4f} | Scratch: {t_sc:7.2f} ms | OpenCV: {t_cv:6.3f} ms")

        plt.suptitle(f"HW3 Task 3: Interpolation Methods Comparison ({int(scale)}x Upscaling: 64x64 -> {target_size}x{target_size})", fontsize=14)
        plt.tight_layout()
        interp_plot_path = os.path.join(out_dir, f"HW3_Q3_interpolation_{int(scale)}x_comparison.png")
        plt.savefig(interp_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved {int(scale)}x interpolation comparison: {interp_plot_path}")

    # -------------------------------------------------------------
    # TASK 4: Geometric Transformations (Translation, Scaling, Rotation)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Task 4: Geometric Transformations via Backward Mapping")
    print("-" * 40)

    # 4A. Translation by (40, 40)
    (trans_sc, t_trans_sc) = benchmark_timer(transform_translation_scratch, lena_gray, 40, 40)
    M_trans = np.float32([[1, 0, 40], [0, 1, 40]])
    (trans_cv, t_trans_cv) = benchmark_timer(cv2.warpAffine, lena_gray, M_trans, (512, 512), flags=cv2.INTER_LINEAR)
    mse_trans = compute_mse(trans_sc, trans_cv)

    cv2.imwrite(os.path.join(out_dir, "HW3_Q4_translation_scratch.bmp"), trans_sc)
    cv2.imwrite(os.path.join(out_dir, "HW3_Q4_translation_cv.bmp"), trans_cv)
    plot_4panel_comparison(lena_gray, trans_sc, trans_cv,
                           "HW3 Task 4: Backward Mapping Translation (tx=40, ty=40)",
                           os.path.join(out_dir, "HW3_Q4_translation_comparison.png"),
                           is_color=False, mse=mse_trans, scratch_time=t_trans_sc, lib_time=t_trans_cv)
    print(f"Translation (40, 40): MSE = {mse_trans:.6e} | Scratch: {t_trans_sc:.3f} ms | OpenCV: {t_trans_cv:.3f} ms")

    # 4B. Scaling by 2.0 and 0.5
    for s_factor in [2.0, 0.5]:
        out_sz = int(512 * s_factor)
        (scale_sc, t_scale_sc) = benchmark_timer(transform_scaling_scratch, lena_gray, s_factor, s_factor)
        M_scale = np.float32([[s_factor, 0, 0], [0, s_factor, 0]])
        (scale_cv, t_scale_cv) = benchmark_timer(cv2.warpAffine, lena_gray, M_scale, (out_sz, out_sz), flags=cv2.INTER_LINEAR)
        mse_scale = compute_mse(scale_sc, scale_cv)

        cv2.imwrite(os.path.join(out_dir, f"HW3_Q4_scale_{s_factor}x_scratch.bmp"), scale_sc)
        cv2.imwrite(os.path.join(out_dir, f"HW3_Q4_scale_{s_factor}x_cv.bmp"), scale_cv)
        print(f"Scaling {s_factor}x ({out_sz}x{out_sz}): MSE = {mse_scale:.6e} | Scratch: {t_scale_sc:.3f} ms | OpenCV: {t_scale_cv:.3f} ms")

    # 4C. Rotation by 45 degrees about image center
    (rot_sc, t_rot_sc) = benchmark_timer(transform_rotation_scratch, lena_gray, 45.0)
    M_rot = cv2.getRotationMatrix2D((255.5, 255.5), 45.0, 1.0)
    (rot_cv, t_rot_cv) = benchmark_timer(cv2.warpAffine, lena_gray, M_rot, (512, 512), flags=cv2.INTER_LINEAR)
    mse_rot = compute_mse(rot_sc, rot_cv)

    cv2.imwrite(os.path.join(out_dir, "HW3_Q4_rotation_45deg_scratch.bmp"), rot_sc)
    cv2.imwrite(os.path.join(out_dir, "HW3_Q4_rotation_45deg_cv.bmp"), rot_cv)
    plot_4panel_comparison(lena_gray, rot_sc, rot_cv,
                           "HW3 Task 4: Center Rotation by 45° with Bilinear Interpolation",
                           os.path.join(out_dir, "HW3_Q4_rotation_comparison.png"),
                           is_color=False, mse=mse_rot, scratch_time=t_rot_sc, lib_time=t_rot_cv)
    print(f"Rotation 45 deg: MSE = {mse_rot:.6e} | Scratch: {t_rot_sc:.3f} ms | OpenCV: {t_rot_cv:.3f} ms")

    print("\nHomework 3 execution completed successfully!")

if __name__ == "__main__":
    main()
