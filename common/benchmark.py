import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between two images.
    MSE = (1 / (M * N * C)) * sum((img1 - img2)^2)
    """
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))

def benchmark_timer(func, *args, **kwargs):
    """
    Executes func(*args, **kwargs) and returns (result, elapsed_time_ms).
    Uses time.perf_counter() for high precision.
    """
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    return res, elapsed_ms

def plot_4panel_comparison(input_img: np.ndarray, scratch_img: np.ndarray, 
                           lib_img: np.ndarray, title: str, save_path: str,
                           is_color: bool = False, mse: float = 0.0,
                           scratch_time: float = 0.0, lib_time: float = 0.0):
    """
    Generates a 4-panel visual comparison:
    [Input Image | Scratch Output | Library Output | Absolute Difference (x10 amplified)]
    """
    diff = np.abs(scratch_img.astype(np.float64) - lib_img.astype(np.float64))
    # Amplify difference for visual clarity if diff is small
    diff_vis = np.clip(diff * 10.0, 0, 255).astype(np.uint8) if diff.max() < 25 else np.clip(diff, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    if is_color:
        in_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        sc_rgb = cv2.cvtColor(scratch_img, cv2.COLOR_BGR2RGB)
        lib_rgb = cv2.cvtColor(lib_img, cv2.COLOR_BGR2RGB)
        diff_show = cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB) if len(diff_vis.shape) == 3 else diff_vis
        
        axes[0].imshow(in_rgb)
        axes[1].imshow(sc_rgb)
        axes[2].imshow(lib_rgb)
        axes[3].imshow(diff_show)
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
