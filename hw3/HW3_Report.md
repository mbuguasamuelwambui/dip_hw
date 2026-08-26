# Homework 3: Resolution, Interpolation, and Geometric Transformations Report

**Course**: Digital Image Processing  
**Dataset**: `LenaColor512.bmp`, `LenaGrey512.bmp`  
**Dual Implementation**: From Scratch (`NumPy`) vs. Library Standard (`OpenCV`)  
**Evaluation Metrics**: Mean Squared Error ($\text{MSE}$), Visual Multi-Panel Grids, Execution Timings (`time.perf_counter()`).

---

## 1. Summary Benchmark Matrix

| Task | Operation / Configuration | Scratch Time | OpenCV Time | MSE ($\text{Scratch} \text{ vs } \text{Lib}$) |
| :--- | :--- | :---: | :---: | :---: |
| **Task 1** | Spatial Downsample & Upsample ($k=8 \dots 4$) | $\sim 0.8\text{ ms}$ | $\sim 0.1\text{ ms}$ | **$0.000000$** |
| **Task 2** | Intensity Quantization ($k=8 \dots 1$) | $\sim 7.5\text{ ms}$ | $\sim 5.5\text{ ms}$ | **$0.000000$** |
| **Task 3A**| Interpolation 2x: Nearest Neighbor | $0.92\text{ ms}$ | $0.04\text{ ms}$ | **$0.000000$** |
| **Task 3B**| Interpolation 2x: Bilinear | $4.16\text{ ms}$ | $0.07\text{ ms}$ | **$0.080800$** |
| **Task 3C**| Interpolation 2x: Bicubic (Keys $a=-0.75$) | $22.30\text{ ms}$ | $0.18\text{ ms}$ | **$0.000000$** |
| **Task 3D**| Interpolation 2x: Lanczos-4 ($a=4$) | $91.03\text{ ms}$ | $0.28\text{ ms}$ | **$0.004500$** |
| **Task 3E**| Interpolation 4x: Nearest Neighbor | $2.25\text{ ms}$ | $0.06\text{ ms}$ | **$0.000000$** |
| **Task 3F**| Interpolation 4x: Bilinear | $8.26\text{ ms}$ | $0.05\text{ ms}$ | **$0.107300$** |
| **Task 3G**| Interpolation 4x: Bicubic (Keys $a=-0.75$) | $169.28\text{ ms}$ | $0.16\text{ ms}$ | **$0.000000$** |
| **Task 3H**| Interpolation 4x: Lanczos-4 ($a=4$) | $422.21\text{ ms}$ | $0.72\text{ ms}$ | **$0.004000$** |
| **Task 4A**| Translation by $(40, 40)$ | $54.94\text{ ms}$ | $0.75\text{ ms}$ | **$0.000000$** |
| **Task 4B**| Scaling $0.5\times$ ($256 \times 256$) | $17.68\text{ ms}$ | $0.17\text{ ms}$ | **$0.000000$** |
| **Task 4C**| Scaling $2.0\times$ ($1024 \times 1024$) | $228.20\text{ ms}$ | $2.83\text{ ms}$ | **$8.684045$** |
| **Task 4D**| Center Rotation $45^\circ$ (Bilinear) | $63.24\text{ ms}$ | $0.72\text{ ms}$ | **$12.151573$** |

---

## 2. Theoretical Breakdown & Implementation Details

### Task 1: Spatial Resolution and Pixelation
* **Mechanism**: Downsampling `LenaColor512.bmp` to spatial dimensions $2^k \times 2^k$ for $k \in \{8, 7, 6, 5, 4\}$ ($256, 128, 64, 32, 16$).
* **Reconstruction**: Upsampled back to $512 \times 512$ via zero-order hold (pixel replication).
* **Observation**:
  * $k=8$ ($256 \times 256$): Retains nearly all visual semantics with minimal perceptual degradation.
  * $k=6$ ($64 \times 64$): Fine facial textures (eyelashes, feathers) disappear into visible blocks.
  * $k=4$ ($16 \times 16$): Severe **pixelation (checkerboard artifact)**; individual macro-blocks dominate, losing facial identity.
* **Saved Visual Grid**: [HW3_Q1_spatial_resolution_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw3/outputs/HW3_Q1_spatial_resolution_grid.png)

---

### Task 2: Intensity Resolution and False Contouring
* **Mechanism**: Reducing the number of available gray levels from $2^8 = 256$ down to $2^1 = 2$ levels (binary thresholding) while maintaining full dynamic range $[0, 255]$.
  $$q(x, y) = \text{round}\left( \left\lfloor \frac{f(x, y)}{256 / 2^k} \right\rfloor \cdot \frac{255}{2^k - 1} \right)$$
* **Observation & Analysis**:
  * $k=8 \dots 5$ ($256 \to 32$ levels): The human eye can barely perceive difference due to Weber's law ($>32$ levels generally suffice for continuous tone perception under normal illumination).
  * $k=4 \dots 3$ ($16 \to 8$ levels): Prominent **false contouring (banding)** appears across smooth gradient regions (cheeks, forehead, shoulder). The continuous gradients degrade into distinct stepped staircase bands.
  * $k=1$ ($2$ levels): Extreme binarization; all subtle gradations are replaced by sharp black-and-white silhouettes.
* **Saved Visual Grid**: [HW3_Q2_intensity_resolution_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw3/outputs/HW3_Q2_intensity_resolution_grid.png)

---

### Task 3: Interpolation Benchmark on Cropped $64 \times 64$ ROI
A $64 \times 64$ ROI surrounding Lena's right eye and facial features ($y \in [240, 304], x \in [240, 304]$) was extracted and upscaled by $2\times$ and $4\times$:

1. **Nearest Neighbor**: 0-order hold. Fast ($0.92\text{ ms}$), but produces harsh jagged edges and blocky artifacts ($\text{MSE} = 0.0$ vs OpenCV).
2. **Bilinear Interpolation**: 1st-order 2D linear weighted blending. Eliminates blockiness but softens fine textures slightly ($\text{MSE} \approx 0.08$ due to sub-pixel floating point rounding).
3. **Bicubic Interpolation**: 3rd-order convolution across a $4 \times 4$ neighborhood using Keys' cubic spline kernel with $a = -0.75$:
   $$W(x) = \begin{cases} (a+2)|x|^3 - (a+3)|x|^2 + 1 & |x| \le 1 \\ a|x|^3 - 5a|x|^2 + 8a|x| - 4a & 1 < |x| < 2 \\ 0 & \text{otherwise} \end{cases}$$
   * Produces crisp, smooth curves with excellent edge sharpness ($\text{MSE} = 0.000000$ exact match to OpenCV `cv2.INTER_CUBIC`).
4. **Lanczos-4 Interpolation**: Sinc-windowed sinc filter over an $8 \times 8$ neighborhood ($a=4$). Superior frequency response and detail retention without excessive blurring ($\text{MSE} = 0.0040$ vs `cv2.INTER_LANCZOS4`).
* **Saved Comparisons**:
  * [HW3_Q3_interpolation_2x_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw3/outputs/HW3_Q3_interpolation_2x_comparison.png)
  * [HW3_Q3_interpolation_4x_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw3/outputs/HW3_Q3_interpolation_4x_comparison.png)

---

### Task 4: Geometric Transformations via Backward Mapping
* **Backward Mapping Principle**: Iterating through destination coordinates $(x_d, y_d)$, mapping back to continuous source coordinates $(x_s, y_s)$ via inverse transformation matrix $A^{-1}$, interpolating with bilinear weights, and setting out-of-boundary regions to 0.
* **Transforms Evaluated**:
  1. **Translation $(40, 40)$**: Displaces image by $40$ pixels right and down ($\text{MSE} = 0.000000$).
  2. **Scaling ($0.5\times$ and $2.0\times$)**: Downscaling and upscaling with explicit coordinate scaling.
  3. **Rotation $45^\circ$ about image center**:
     $$M_{\text{fwd}} = \begin{bmatrix} \cos 45^\circ & \sin 45^\circ & (1-\cos 45^\circ)c_x - \sin 45^\circ c_y \\ -\sin 45^\circ & \cos 45^\circ & \sin 45^\circ c_x + (1-\cos 45^\circ)c_y \end{bmatrix}, \quad M_{\text{inv}} = M_{\text{fwd}}^{-1}$$
     * Center: $((W-1)/2, (H-1)/2) = (255.5, 255.5)$.
     * Generates rotated image with black boundary fill and smooth anti-aliased edges.
* **Saved Comparisons**:
  * [HW3_Q4_translation_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw3/outputs/HW3_Q4_translation_comparison.png)
  * [HW3_Q4_rotation_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw3/outputs/HW3_Q4_rotation_comparison.png)
