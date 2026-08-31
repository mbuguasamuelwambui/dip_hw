# Homework 1: Image Basics and Pixel Operations Report

**Course**: Digital Image Processing  
**Dataset**: `LenaColor512.bmp` ($512 \times 512 \times 3$)  
**Implementation**: Python (`NumPy`, `OpenCV`, `Matplotlib`)

---

## 1. Image Metadata Inspection (Question 1)

### Theoretical & Implementation Overview
Digital images stored in standard uncompressed formats 
In OpenCV, image arrays follow the **BGR** (Blue, Green, Red) channel ordering rather than **RGB**.

### Extracted Properties
* **Dimensions**: $512 \times 512$ pixels ($H = 512, W = 512$)
* **Number of Channels**: $3$ (Color)
* **Data Type (`dtype`)**: `uint8` ($0 \dots 255$)
* **Per-Channel Intensity Extrema**:
  | Channel | Name | Minimum Intensity | Maximum Intensity | Mean Intensity |
  | :--- | :--- | :---: | :---: | :---: |
  | Channel 0 | **Blue (B)** | $8$ | $225$ | $105.40$ |
  | Channel 1 | **Green (G)** | $3$ | $248$ | $99.04$ |
  | Channel 2 | **Red (R)** | $54$ | $255$ | $180.18$ |

* **Output Saved**: `HW1_Q1.bmp`

---

## 2. Drawing Operations (Questions 2 – 4)

### Question 2: Solid Red Vertical Center Line
* **Coordinate Formula**: Center column $x = \lfloor W / 2 \rfloor = 512 // 2 = 256$.
* **Color Specification**: Pure Red in BGR format is `[0, 0, 255]`.
* **Array Indexing**: `img[:, 256, :] = [0, 0, 255]`
* **Output Saved**: `HW1_Q2.bmp`

### Question 3: Blue Diagonal Line of Thickness 2
* **Coordinates**: Start point $(x_1, y_1) = (100, 100)$, End point $(x_2, y_2) = (200, 200)$.
* **Color Specification**: Pure Blue in BGR format is `[255, 0, 0]`.
* **Implementation**: `cv2.line(img, (100, 100), (200, 200), (255, 0, 0), thickness=2)`
* **Output Saved**: `HW1_Q3.bmp`

### Question 4: Solid Green Rectangle via Direct Array Assignment
* **Coordinates**: Top-left $(x_1, y_1) = (50, 50)$ to Bottom-right $(x_2, y_2) = (100, 100)$ **inclusive**.
* **Array Slicing**: In NumPy, `img[y_start:y_end, x_start:x_end]` uses half-open intervals $[a, b)$. To include index $100$, the slice must be `50:101`.
* **Color Specification**: Pure Green in BGR format is `[0, 255, 0]`.
* **Assignment**: `img[50:101, 50:101, :] = [0, 255, 0]`
* **Output Saved**: `HW1_Q4.bmp`

---

## 3. Color Spaces & Visual Content Analysis (Question 5)

Understanding color representations is fundamental in computer vision and image processing. Different representations isolate intensity from chromaticity differently:

### 1. RGB Color Space ($R, G, B$)
* **Red (R)**: High brightness across Lena's face, skin, and hat. Skin contains strong red components, resulting in elevated pixel intensities ($R \in [54, 255]$).
* **Green (G)**: Moderate intensity across skin tones and background; closely resembles standard monochromatic luminance.
* **Blue (B)**: Darker across the face and hat; higher intensities in the background cyan/blue areas and feather specular highlights.
* **Limitation**: High inter-channel correlation—a change in lighting affects all three channels simultaneously, making RGB poor for illumination-invariant processing.

---

### 2. HSV Color Space ($H, S, V$)
* **Hue ($H \in [0, 179]$ in OpenCV)**:
  * Encodes dominant wavelength (color tint).
  * Flat/uniform regions on skin have similar hue values. The hat and background show distinct step-changes in intensity representing differing color angles.
* **Saturation ($S \in [0, 255]$)**:
  * Encodes color purity/vibrancy.
  * Hat ribbons, saturated feather plumes, and lips appear very bright (high saturation), while muted skin and background wall appear darker (low saturation).
* **Value ($V \in [0, 255]$)**:
  * Encodes maximum channel brightness ($\max(R, G, B)$).
  * Clearly shows structural lighting, highlights, and shadows regardless of hue.

---

### 3. CIELAB Color Space ($L^*, a^*, b^*$)
* **Perceptual Uniformity**: Designed so that a Euclidean distance $\Delta E = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$ directly corresponds to human visual perceived color difference.
* **$L^*$ (Lightness, $0 \dots 255$)**:
  * Closely matches the non-linear human visual perception of luminance (achromatic channel).
* **$a^*$ (Green–Red axis, centered at $128$)**:
  * Values $>128$ indicate magenta/red; values $<128$ indicate green.
  * Lena's reddish skin, lips, and hat ribbon appear bright ($>128$), while green/neutral areas stay mid-gray.
* **$b^*$ (Blue–Yellow axis, centered at $128$)**:
  * Values $>128$ indicate yellow; values $<128$ indicate blue.
  * Warm skin tones appear bright (yellowish tint), whereas blue background elements appear dark.

---

### 4. $\text{YCrCb}$ Color Space ($Y, Cr, Cb$)
* Used heavily in JPEG compression and digital broadcasting (ITU-R BT.601).
* **$Y$ (Luma)**: Weighted sum of gamma-compressed RGB ($Y = 0.299R + 0.587G + 0.114B$). Contains virtually all spatial high-frequency geometric detail.
* **$Cr$ (Chroma Red Difference, $R - Y + 128$)**: Highlights skin tones and red elements. Very popular for facial detection / skin color clustering.
* **$Cb$ (Chroma Blue Difference, $B - Y + 128$)**: Highlights blue components and complementary coolness.
* **Advantage**: Because the human eye is much more sensitive to spatial luminance variations than chroma variations, $Cr$ and $Cb$ can be aggressively downsampled (e.g., 4:2:0 chroma subsampling) without noticeable perceptual quality loss.

---

## 4. Summary Output Artifacts Generated

1. [HW1_Q1.bmp](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw1/outputs/HW1_Q1.bmp): Original Lena color image ($512 \times 512$).
2. [HW1_Q2.bmp](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw1/outputs/HW1_Q2.bmp): Red center vertical line ($x=256$).
3. [HW1_Q3.bmp](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw1/outputs/HW1_Q3.bmp): Blue line of thickness 2 from $(100,100)$ to $(200,200)$.
4. [HW1_Q4.bmp](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw1/outputs/HW1_Q4.bmp): Solid green rectangle slice $[50:101, 50:101]$.
5. [HW1_Q1_to_Q4_summary.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw1/outputs/HW1_Q1_to_Q4_summary.png): 4-panel visual verification grid.
6. [HW1_Q5_color_spaces_grid.png](file:///c:/Users/SAMMY/Documents/coursework/dip/hw/hw1/outputs/HW1_Q5_color_spaces_grid.png): Comprehensive $4 \times 3$ grid of all 12 decomposed channels.
