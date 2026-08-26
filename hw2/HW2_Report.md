# Homework 2: Intensity Transformations and Histogram Processing Report

**Course**: Digital Image Processing  
**Dataset**: `LenaGrey512.bmp`, `LenaColor512.bmp`, `LenaGrey256.bmp`  
**Dual Implementation**: From Scratch (`NumPy`) vs. Library Standard (`OpenCV` / `Scikit-Image`)  
**Evaluation Metrics**: Mean Squared Error ($\text{MSE}$), Visual 4-Panel Comparison, Execution Timings (`time.perf_counter()`).

---

## 1. Summary Benchmark Matrix

| Task | Operation | Input Image | Scratch Time (ms) | Library Time (ms) | MSE ($\text{Scratch} \text{ vs } \text{Lib}$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Task 1A** | Image Negative (Grayscale) | `LenaGrey512.bmp` | $0.524$ | $0.232$ | **$0.000000$** |
| **Task 1B** | Image Negative (Color) | `LenaColor512.bmp` | $1.743$ | $0.272$ | **$0.000000$** |
| **Task 2A** | Gamma ($\gamma = 0.40$) | `LenaGrey512.bmp` | $25.039$ | $9.524$ | **$0.000000$** |
| **Task 2B** | Gamma ($\gamma = 0.67$) | `LenaGrey512.bmp` | $21.210$ | $7.262$ | **$0.000000$** |
| **Task 2C** | Gamma ($\gamma = 1.50$) | `LenaGrey512.bmp` | $19.449$ | $5.216$ | **$0.000000$** |
| **Task 2D** | Gamma ($\gamma = 2.50$) | `LenaGrey512.bmp` | $23.154$ | $4.289$ | **$0.000000$** |
| **Task 3A** | Logarithmic Transform | `LenaGrey512.bmp` | $16.278$ | $14.308$ | **$0.000000$** |
| **Task 3B** | Piecewise-Linear Stretching | `LenaGrey512.bmp` | $12.166$ | $6.858$ | **$0.000000$** |
| **Task 4** | Histogram Equalization | `LenaGrey256.bmp` | $1.158$ | $0.208$ | **$0.000000$** |
| **Task 5A** | Hist Matching (Gaussian $\mu=128, \sigma=40$) | `LenaGrey256.bmp` | $2.304$ | — | — |
| **Task 5B** | Hist Matching (Natural Reference) | `LenaGrey256.bmp` | $6.321$ | $4.091$ | **$0.542526$** |

---

## 2. Theoretical Breakdown & Implementation Details

### Task 1: Image Negative
* **Mathematical Model**:
  $$s = (L - 1) - r = 255 - r$$
* **Scratch Formulation**: Direct element-wise subtraction `(255 - img).astype(np.uint8)`.
* **Library Formulation**: `cv2.bitwise_not(img)`.
* **Findings**: Exactly equivalent ($\text{MSE} = 0.0$). Inverted grayscale brings out shadow details, while inverting color maps complementary colors (e.g., cyan $\leftrightarrow$ red, yellow $\leftrightarrow$ blue).
* **Saved Outputs**:
  * [HW2_Q1_negative_gray_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw2/outputs/HW2_Q1_negative_gray_comparison.png)
  * [HW2_Q1_negative_color_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw2/outputs/HW2_Q1_negative_color_comparison.png)

---

### Task 2: Power-Law (Gamma) Transform
* **Mathematical Model**:
  $$s = c \cdot r^\gamma, \quad r \in [0, 1] \implies s_{\text{scaled}} = \text{round}\left(255 \cdot \left(\frac{r}{255}\right)^\gamma\right)$$
* **Behavior Analysis**:
  * $\gamma < 1$ ($\gamma = 0.4, 0.67$): Expands dark shadow levels into higher brightness values (brightens underexposed regions).
  * $\gamma > 1$ ($\gamma = 1.5, 2.5$): Compresses mid-tones towards dark levels (increases contrast in bright/washed-out regions while darkening backgrounds).
* **Scratch vs. OpenCV LUT**: Both produce identical rounded 8-bit integer mappings ($\text{MSE} = 0.0$).
* **Saved Grid**: [HW2_Q2_gamma_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw2/outputs/HW2_Q2_gamma_grid.png)

---

### Task 3: Log Transform & Piecewise-Linear Contrast Stretching
* **Logarithmic Transformation**:
  $$s = c \cdot \ln(1 + r), \quad c = \frac{255}{\ln(1 + 255)} \approx 45.9859$$
  * *Effect*: Dramatically expands dark pixel values while compressing high-intensity highlights. Widely used when displaying Fourier spectra.
* **Piecewise-Linear Contrast Stretching**:
  * Defined with control points $(r_1, s_1) = (70, 20)$ and $(r_2, s_2) = (180, 235)$:
    $$s = \begin{cases} \frac{s_1}{r_1} r & 0 \le r < r_1 \\ \frac{s_2 - s_1}{r_2 - r_1}(r - r_1) + s_1 & r_1 \le r \le r_2 \\ \frac{255 - s_2}{255 - r_2}(r - r_2) + s_2 & r_2 < r \le 255 \end{cases}$$
  * *Effect*: Selectively stretches the dynamic range of the mid-tones where primary information resides.
* **Saved Visualizations**:
  * [HW2_Q3_log_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw2/outputs/HW2_Q3_log_comparison.png)
  * [HW2_Q3_piecewise_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw2/outputs/HW2_Q3_piecewise_comparison.png)

---

### Task 4: Histogram Equalization
* **Formulation**:
  1. Histogram frequency: $n_k = \sum_{x,y} \mathbf{1}(f(x,y) = k)$.
  2. Probability Mass Function: $p_r(r_k) = \frac{n_k}{M \cdot N}$.
  3. Cumulative Distribution Function: $C(k) = \sum_{j=0}^{k} p_r(r_j)$.
  4. Transformation mapping: $s_k = \lfloor (L - 1) C(k) + 0.5 \rfloor$.
* **Scratch vs. OpenCV (`cv2.equalizeHist`)**:
  * Identical output with $\text{MSE} = 0.0$.
  * Equalization linearizes the CDF, spreading concentrated pixel modes across all 256 gray levels, dramatically maximizing global image contrast.
* **Saved Analysis**: [HW2_Q4_histogram_equalization_analysis.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw2/outputs/HW2_Q4_histogram_equalization_analysis.png)

---

### Task 5: Histogram Matching (Specification)
* **Goal**: Force the input image distribution to approximate a desired target distribution (synthetic Gaussian or natural image reference).
* **Algorithm**:
  1. Calculate input image CDF: $S(r_k) = \sum_{j=0}^{k} p_{\text{in}}(j)$.
  2. Calculate target distribution CDF: $G(z_q) = \sum_{j=0}^{q} p_{\text{target}}(j)$.
  3. Construct monotonic mapping $T: r_k \to z_q$ by finding $z_q = \arg\min_{z} |S(r_k) - G(z)|$.
  4. Remap input pixels using the inverse mapping lookup table.
* **Target 1 (Synthetic Gaussian $\mu = 128, \sigma = 40$)**: Produces a bell-curve histogram with high tonal smoothness.
* **Target 2 (Natural Reference $512 \times 512$)**: Successfully transfers the tonal balance from a higher-resolution reference to a $256 \times 256$ input image ($\text{MSE} = 0.54$ vs scikit-image).
* **Saved Analysis**: [HW2_Q5_histogram_matching_analysis.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw2/outputs/HW2_Q5_histogram_matching_analysis.png)
