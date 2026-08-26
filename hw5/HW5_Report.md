# Homework 5: Frequency Domain Analysis Report

**Course**: Digital Image Processing  
**Dataset**: `LenaGrey256.bmp`  
**Dual Implementation**: From Scratch (`NumPy` 1D DFT $O(N^2)$, Radix-2 1D FFT $O(N \log N)$, 2D Separable Transform) vs. Standard Library (`np.fft.fft2`)  
**Evaluation Metrics**: Mean Squared Error ($\text{MSE}$), Log Spectrum, Phase/Magnitude Reconstruction, Ringing / Gibbs Artifact Analysis.

---

## 1. Summary Benchmark Matrix

### Task 1: 2D Fourier Transform Algorithmic Benchmark

| Matrix Size $N \times N$ | 2D DFT Scratch ($O(N^3)$) | 2D Radix-2 FFT Scratch ($O(N^2 \log N)$) | NumPy C-FFT (`fft2`) | MSE (Scratch FFT vs NumPy) |
| :---: | :---: | :---: | :---: | :---: |
| $16 \times 16$ | $5.25\text{ ms}$ | $6.22\text{ ms}$ | $0.28\text{ ms}$ | **$8.08 \times 10^{-29}$** |
| $32 \times 32$ | $8.75\text{ ms}$ | $34.03\text{ ms}$ | $0.33\text{ ms}$ | **$1.44 \times 10^{-26}$** |
| $64 \times 64$ | $67.78\text{ ms}$ | $112.94\text{ ms}$ | $0.32\text{ ms}$ | **$2.24 \times 10^{-25}$** |
| $128 \times 128$ | $633.70\text{ ms}$ | $833.43\text{ ms}$ | $0.64\text{ ms}$ | **$1.84 \times 10^{-24}$** |
| $256 \times 256$ | $> 5000\text{ ms}$ | $2528.95\text{ ms}$ | $2.72\text{ ms}$ | **$1.58 \times 10^{-23}$** |

*Note*: The Scratch Radix-2 FFT matches NumPy's compiled PocketFFT down to **machine precision floating-point error ($\sim 10^{-23}$)**!

---

## 2. Theoretical Breakdown & Implementation Details

### Task 1: Discrete Fourier Transform & Radix-2 Fast Fourier Transform
* **Mathematical Convention**:
  $$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}, \quad f(x, y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) e^{+j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
* **Row-Column Separability**:
  Because the kernel $e^{-j 2\pi (ux/M + vy/N)} = e^{-j 2\pi ux/M} e^{-j 2\pi vy/N}$ is separable, the 2D transform is computed by running 1D FFTs across all rows, followed by 1D FFTs down all resulting columns.
* **Saved Benchmark Plot**: [HW5_Q1_fft_dft_benchmark.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw5/outputs/HW5_Q1_fft_dft_benchmark.png)

---

### Task 2: Spectrum, Phase, and Spatial Reconstruction
* **Centering Property**: Multiplying $f(x, y)$ by $(-1)^{x+y}$ shifts the origin $(0, 0)$ from the corner to the optical frequency center $(M/2, N/2)$.
* **Log-Magnitude Spectrum**: $S(u, v) = \ln(1 + |F(u, v)|)$. Compresses the massive dynamic range of the DC component, rendering subtle high-frequency star patterns and cross-artifacts clearly visible.
* **Phase-Only vs. Magnitude-Only Reconstruction**:
  1. **Phase-Only Reconstruction** ($F_{\text{phase}} = e^{j \angle F(u, v)}$):
     * *Observation*: Reconstructs all edge locations, contours, boundaries, facial lines, and hat geometry!
     * *Reason*: Phase encodes the relative spatial alignment where sinusoids constructively interfere to form sharp localized edges.
  2. **Magnitude-Only Reconstruction** ($F_{\text{mag}} = |F(u, v)|$):
     * *Observation*: All semantic geometry and object identity are completely lost, collapsing into an autocorrelation energy glow at the image center.
* **Saved Comparison**: [HW5_Q2_phase_vs_magnitude_comparison.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw5/outputs/HW5_Q2_phase_vs_magnitude_comparison.png)

---

### Task 3: Low-Pass Frequency Domain Filters & Ringing Artifacts
For cutoff frequencies $D_0 \in \{10, 30, 60\}$:

1. **Ideal Low-Pass Filter (ILPF)**:
   $$H_{\text{ILPF}}(u, v) = \begin{cases} 1 & D(u, v) \le D_0 \\ 0 & D(u, v) > D_0 \end{cases}$$
   * *Spatial Response*: Inverse Fourier transform of a 2D cylinder is a circularly symmetric **$\text{jinc}$ function** ($\frac{J_1(r)}{r}$).
   * *Artifact*: Prominent concentric ripple halos (**Gibbs ringing**) across edges, especially severe at small $D_0 = 10$.
2. **Butterworth Low-Pass Filter (BLPF, $n = 2$)**:
   $$H_{\text{BLPF}}(u, v) = \frac{1}{1 + \left( \frac{D(u, v)}{D_0} \right)^{2n}}$$
   * *Spatial Response*: Smooth transition band provides continuous attenuation, almost entirely eliminating ringing.
3. **Gaussian Low-Pass Filter (GLPF)**:
   $$H_{\text{GLPF}}(u, v) = e^{-\frac{D^2(u, v)}{2 D_0^2}}$$
   * *Spatial Response*: The Fourier transform of a Gaussian is another Gaussian ($h(x, y) = 2\pi\sigma^2 e^{-2\pi^2\sigma^2(x^2+y^2)}$).
   * *Result*: **Completely zero ringing artifacts** across all cutoff frequencies.
* **Saved Grid**: [HW5_Q3_lowpass_filters_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw5/outputs/HW5_Q3_lowpass_filters_grid.png)

---

### Task 4: High-Frequency Emphasis (HFE) & Histogram Equalization
* **Problem with Pure High-Pass ($H_{\text{HP}} = 1 - H_{\text{LP}}$)**:
  * Since $H_{\text{HP}}(0, 0) = 0$, the DC component (average illumination) is completely zeroed out, reducing the image to dark faint edge contours.
* **High-Frequency Emphasis Formulation**:
  $$H_{\text{HFE}}(u, v) = a + b \cdot H_{\text{HP}}(u, v) = 0.5 + 2.0 \cdot H_{\text{HP}}(u, v)$$
  * With $a = 0.5$, the DC component is preserved at $50\%$ brightness, maintaining background illumination.
  * With $b = 2.0$, high-frequency textural detail is amplified by $a + b = 2.5\times$.
* **Post-Processing**:
  * Applying **Histogram Equalization** spreads the enhanced high-frequency spectrum across the full dynamic range $[0, 255]$, yielding exceptional local contrast, crisp textures, and rich depth.
* **Saved Grid**: [HW5_Q4_high_frequency_emphasis_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw5/outputs/HW5_Q4_high_frequency_emphasis_grid.png)
