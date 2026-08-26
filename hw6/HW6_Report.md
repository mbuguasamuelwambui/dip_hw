# Homework 6: Frequency Applications and Advanced Transforms Report

**Course**: Digital Image Processing  
**Dataset**: `LenaGrey256.bmp`  
**Dual Implementation**: From Scratch (`NumPy` Butterworth Notch Filtering, Homomorphic Filtering, 2D DCT-II / IDCT-II) vs. Standard Library (`OpenCV` / `Scikit-Image`)  
**Evaluation Metrics**: Mean Squared Error ($\text{MSE}$), Structural Similarity Index ($\text{SSIM}$), Energy Compaction Ratios.

---

## 1. Summary Benchmark Matrix

| Task | Transform / Filter Operation | Parameters / Config | Scratch Time | Library Time | MSE ($\text{Scratch} \text{ vs } \text{Lib}$) | Performance / Quality Metric |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Task 1** | Butterworth Notch Reject Filter | $u_0=\pm 32, v_0=\pm 32, D_0=12, n=2$ | $14.51\text{ ms}$ | — | Restores image from noise | **$\text{SSIM} = 0.9611$, $\text{MSE} = 19.95$** |
| **Task 2** | Homomorphic Filtering | $\gamma_L=0.25, \gamma_H=2.0, D_0=30, c=1$ | $12.30\text{ ms}$ | — | Normalizes non-uniform shadow | Illumination normalized |
| **Task 3A**| 2D DCT-II Transform on $8 \times 8$ | Matrix formulation vs `cv2.dct` | $0.08\text{ ms}$ | $0.02\text{ ms}$ | **$1.58 \times 10^{-12}$** | **Exact match to OpenCV** |
| **Task 3B**| DCT Energy Compaction ($3 \times 3$ / $9$ coeffs) | $86\%$ coefficient zeroing | — | — | — | $\text{MSE} = 64.73$ |
| **Task 3C**| FFT Energy Compaction ($3 \times 3$ / $9$ coeffs) | $86\%$ coefficient zeroing | — | — | — | $\text{MSE} = 63.56$ |

---

## 2. Theoretical Breakdown & Implementation Details

### Task 1: Periodic Noise Removal via Butterworth Notch Reject Filtering
* **Noise Generation**:
  $$n(x, y) = 50 \sin\left(2\pi \frac{32 x}{M} + 2\pi \frac{32 y}{N}\right)$$
  * *Spectral Signature*: Produces two bright symmetric delta impulse spikes in the centered 2D Fourier spectrum at $(+32, +32)$ and $(-32, -32)$ relative to $(M/2, N/2)$.
* **Butterworth Notch Reject Filter (NRF) Formulation**:
  $$H_{\text{NRF}}(u, v) = \frac{1}{1 + \left( \frac{D_0^2}{D_1(u, v) \cdot D_2(u, v)} \right)^n}$$
  where:
  $$D_1(u, v) = \sqrt{(u - M/2 - 32)^2 + (v - N/2 - 32)^2}, \quad D_2(u, v) = \sqrt{(u - M/2 + 32)^2 + (v - N/2 + 32)^2}$$
* **Restoration Performance**:
  * *Corrupted Image*: $\text{MSE} = 1184.94, \text{SSIM} = 0.3014$ (severe diagonal grating artifact across the entire image).
  * *Notch Filtered Output*: **$\text{MSE} = 19.95, \text{SSIM} = 0.9611$** ($96.1\%$ structural similarity restored, completely removing sinusoidal stripes without blurring underlying facial features).
* **Saved Analysis**: [HW6_Q1_notch_filtering_analysis.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw6/outputs/HW6_Q1_notch_filtering_analysis.png)

---

### Task 2: Homomorphic Filtering for Illumination-Reflectance Decomposition
* **Illumination-Reflectance Physics**:
  $$f(x, y) = i(x, y) \cdot r(x, y)$$
  * $i(x, y)$: Illumination component (spatial lighting field, varies slowly across the scene $\implies$ **low frequencies**).
  * $r(x, y)$: Reflectance component (object textures, surface details, edges $\implies$ **high frequencies**).
* **Homomorphic Transformation Pipeline**:
  1. *Logarithmic Transformation*: $z = \ln(1 + f) = \ln(i) + \ln(r)$ (transforms multiplicative components into additive components).
  2. *Fourier Transform*: $Z(u, v) = \mathcal{F}\{z(x, y)\} = I(u, v) + R(u, v)$.
  3. *High-Frequency Emphasis Transfer Function*:
     $$H_{\text{homo}}(u, v) = (\gamma_H - \gamma_L) \left[ 1 - \exp\left(-c \frac{D^2(u, v)}{D_0^2}\right) \right] + \gamma_L$$
  4. *Inverse Transform & Exponential*: $s = \mathcal{F}^{-1}\{H_{\text{homo}} \cdot Z\} \implies g = \exp(\text{Re}(s)) - 1$.
* **Roles of Parameters $\gamma_L$ and $\gamma_H$**:
  * **$\gamma_L = 0.25 < 1.0$**: Attenuates low frequencies, compressing the dynamic range caused by uneven lighting / dark shadows.
  * **$\gamma_H = 2.0 > 1.0$**: Amplifies high frequencies, enhancing surface reflectance textures and sharp edges.
* **Saved Visual Grid**: [HW6_Q2_homomorphic_filtering_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw6/outputs/HW6_Q2_homomorphic_filtering_grid.png)

---

### Task 3: 2D DCT-II vs. 2D FFT and Energy Compaction (JPEG Compression Principle)
* **2D DCT-II Matrix Formulation**:
  $$C(u, v) = \alpha(u) \alpha(v) \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x, y) \cos\left( \frac{(2x + 1)u \pi}{2N} \right) \cos\left( \frac{(2y + 1)v \pi}{2N} \right)$$
  where $\alpha(0) = \sqrt{1/N}$ and $\alpha(u) = \sqrt{2/N}$ for $u > 0$.
  * *Verification*: Scratch implementation matches OpenCV `cv2.dct` with $\text{MSE} = 1.58 \times 10^{-12}$.
* **Energy Compaction Experiment on $8 \times 8$ Block**:
  * Retaining only $3 \times 3 = 9$ coefficients out of $64$ (zeroing $86\%$ of transform coefficients).
* **Why DCT is Preferred Over FFT for Image Compression**:
  1. **Boundary Periodicity & Edge Discontinuities**:
     * The **DFT** assumes the $8 \times 8$ block repeats infinitely with periodic wrap-around ($f(x) = f(x + N)$). If the left edge of a block has different intensity from the right edge, this creates an artificial sharp step discontinuity at block boundaries, causing severe high-frequency spectral leakage.
     * The **DCT** implicitly reflects the image symmetrically across boundaries before repeating ($f(x) = f(-x - 1)$), ensuring the boundary is continuous ($C^0$ continuity) and eliminating artificial boundary step discontinuities.
  2. **Energy Concentration**:
     * Because the cosine basis functions do not need to represent artificial boundary step jumps, almost all the signal energy is concentrated in a tiny cluster of low-frequency coefficients near $(0, 0)$ (the DC and low AC components).
     * Truncating high frequencies in DCT preserves smooth transitions across JPEG block borders without producing the harsh ringing and boundary errors seen in FFT block compression.
* **Saved Analysis**: [HW6_Q3_dct_vs_fft_compaction.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw6/outputs/HW6_Q3_dct_vs_fft_compaction.png)
