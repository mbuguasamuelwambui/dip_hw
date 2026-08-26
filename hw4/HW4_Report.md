# Homework 4: Spatial Domain Filtering Report

**Course**: Digital Image Processing  
**Dataset**: `LenaGrey256.bmp`, `LenaGrey512.bmp`  
**Dual Implementation**: From Scratch (`NumPy`) vs. Library Standard (`OpenCV`)  
**Evaluation Metrics**: Mean Squared Error ($\text{MSE}$), Visual Multi-Panel Grids, Execution Timings (`time.perf_counter()`).

---

## 1. Summary Benchmark Matrix

| Task | Filter Operation | Kernel / Parameters | Scratch Time | OpenCV Time | MSE ($\text{Scratch} \text{ vs } \text{Lib}$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Task 1A** | Box Filter | $3 \times 3$ | $5.60\text{ ms}$ | $9.84\text{ ms}$ | **$0.000000$** |
| **Task 1A** | Box Filter | $5 \times 5$ | $13.67\text{ ms}$ | $0.81\text{ ms}$ | **$0.000000$** |
| **Task 1A** | Box Filter | $9 \times 9$ | $9.91\text{ ms}$ | $0.10\text{ ms}$ | **$0.000000$** |
| **Task 1B** | Gaussian Filter | $5 \times 5, \sigma = 1.0$ | $5.11\text{ ms}$ | $4.84\text{ ms}$ | **$0.019485$** |
| **Task 1B** | Gaussian Filter | $11 \times 11, \sigma = 2.5$ | $10.12\text{ ms}$ | $0.39\text{ ms}$ | **$0.017822$** |
| **Task 1C** | Median Filter (10% Salt & Pepper) | $3 \times 3$ sliding window | $20.63\text{ ms}$ | $1.29\text{ ms}$ | **$0.768692$** |
| **Task 2A** | Laplacian Sharpening ($K_1$) | $3 \times 3$ 4-neighbor | $3.42\text{ ms}$ | $4.70\text{ ms}$ | **$0.000000$** |
| **Task 2A** | Laplacian Sharpening ($K_2$) | $3 \times 3$ 8-neighbor | $4.01\text{ ms}$ | $0.64\text{ ms}$ | **$0.000000$** |
| **Task 2B** | Unsharp Masking | $k = 1.0, 5 \times 5, \sigma=1.0$ | $6.10\text{ ms}$ | $2.54\text{ ms}$ | **$0.018082$** |
| **Task 2B** | High-Boost Filtering | $k = 4.5, 5 \times 5, \sigma=1.0$ | $4.83\text{ ms}$ | $2.53\text{ ms}$ | **$1.727371$** |
| **Task 2C** | Sobel Gradient Magnitude | $3 \times 3$ Sobel $S_x, S_y$ | $9.23\text{ ms}$ | $7.78\text{ ms}$ | **$0.000000$** |
| **Task 3**  | 6-Stage Combined Pipeline | Multi-stage pipeline | $18.33\text{ ms}$ | — | Complete Pipeline |

---

## 2. Theoretical Breakdown & Implementation Details

### Task 1: Low-Pass and Nonlinear Filtering
* **Box Filter**: Average kernel $\frac{1}{k^2}\mathbf{1}_{k \times k}$. As $k$ increases ($3 \to 5 \to 9$), spatial blur increases rapidly.
* **Gaussian Filter**: Isotropic Gaussian kernel $G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$, normalized so $\sum G = 1.0$. Provides smooth attenuation of high spatial frequencies without directional artifacts.
* **Linear Mean vs. Nonlinear Median Filtering on 10% Salt & Pepper Noise**:
  * *Noisy Image MSE to Clean*: $1864.91$
  * *Mean Filtered MSE to Clean*: $323.93$ $\implies$ Linear averaging diffuses extreme $0$ and $255$ impulse spikes into local neighborhoods, creating noticeable blurry gray smudges while failing to restore the clean image.
  * *Median Filtered MSE to Clean*: **$67.60$** $\implies$ The nonlinear rank selection picks the middle value in the neighborhood, discarding statistical outliers ($0$ and $255$), completely removing impulse spikes and preserving crisp edges.
* **Saved Visual Grid**: [HW4_Q1_lowpass_filtering_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw4/outputs/HW4_Q1_lowpass_filtering_grid.png)

---

### Task 2: High-Pass, Sharpening, and Gradient Filtering
* **Laplacian Sharpening**:
  * $K_1 = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} \implies g = f - \nabla^2 f$
  * $K_2 = \begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix} \implies g = f + \nabla^2 f$
  * $K_2$ incorporates diagonal neighbors, producing stronger isotropic edge enhancement.
* **Unsharp Masking and High-Boost**:
  * Blur image: $f_{\text{smooth}}$, Form mask: $g_{\text{mask}} = f - f_{\text{smooth}}$, Combine: $g = f + k \cdot g_{\text{mask}}$
  * $k = 1.0$: Standard unsharp masking (enhances fine boundaries).
  * $k = 4.5$: High-boost filtering (strongly amplifies high-frequency texture components).
* **Sobel Gradients (Magnitude & Phase)**:
  * Horizontal $S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$, Vertical $S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$
  * Magnitude $M(x, y) = \sqrt{g_x^2 + g_y^2}$ reveals all edge boundaries.
  * Phase $\alpha(x, y) = \arctan(g_y / g_x)$ reveals local edge normal orientations.
* **Saved Visual Grid**: [HW4_Q2_sharpening_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw4/outputs/HW4_Q2_sharpening_grid.png)

---

### Task 3: Multi-Stage Combined Image Enhancement Pipeline
Using artificially contrast-reduced `LenaGrey256.bmp` ($f \in [100, 150]$):

1. **Stage (a) - Laplacian Sharpened ($g_a = f - \nabla^2 f$)**:
   Enhances fine features, but introduces background noise and retains washed-out contrast.
2. **Stage (b) - Sobel Gradient Magnitude ($g_b = \sqrt{g_x^2 + g_y^2}$)**:
   Extracts prominent boundary edges while keeping smooth background regions dark.
3. **Stage (c) - Smoothed Sobel Mask ($g_c = \text{BoxFilter}_{5\times 5}(g_b)$)**:
   Averages and broadens edge transitions to eliminate noise discontinuities.
4. **Stage (d) - Mask Product ($g_d = g_a \cdot g_c$)**:
   Selectively weights the Laplacian sharpened features only at true edge locations identified by the smoothed Sobel mask, suppressing amplified noise in flat areas.
5. **Stage (e) - Sharpened Recombination ($g_e = f + g_d$)**:
   Adds edge-weighted sharpened details back to the original image $f$.
6. **Stage (f) - Power-Law / Gamma Transform ($g_f = c \cdot g_e^{0.5}$)**:
   Expands the compressed dynamic range $[100, 150]$ across the full visual spectrum $[0, 255]$, producing a crisp, high-contrast, visually striking final image.
* **Saved Pipeline Grid**: [HW4_Q3_combined_pipeline_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw4/outputs/HW4_Q3_combined_pipeline_grid.png)
