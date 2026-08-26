import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "..", "images")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = os.path.join(img_dir, "LenaColor512.bmp")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find input image: {input_path}")

    print("=" * 60)
    print("Homework 1: Image Basics and Pixel Operations")
    print("=" * 60)

    # -------------------------------------------------------------
    # Question 1: Load, inspect metadata, and save
    # -------------------------------------------------------------
    print("\n--- Question 1: Image Metadata Inspection ---")
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise ValueError("Failed to load image via cv2.imread")
    
    height, width, channels = img_bgr.shape
    dtype = img_bgr.dtype
    
    # Min/Max per channel in BGR and RGB terms
    # Channel 0: B, Channel 1: G, Channel 2: R
    b_min, b_max = img_bgr[:, :, 0].min(), img_bgr[:, :, 0].max()
    g_min, g_max = img_bgr[:, :, 1].min(), img_bgr[:, :, 1].max()
    r_min, r_max = img_bgr[:, :, 2].min(), img_bgr[:, :, 2].max()

    print(f"Height       : {height} px")
    print(f"Width        : {width} px")
    print(f"Channels     : {channels}")
    print(f"Data Type    : {dtype}")
    print(f"Blue  Channel: min = {b_min:3d}, max = {b_max:3d}")
    print(f"Green Channel: min = {g_min:3d}, max = {g_max:3d}")
    print(f"Red   Channel: min = {r_min:3d}, max = {r_max:3d}")

    q1_path = os.path.join(output_dir, "HW1_Q1.bmp")
    cv2.imwrite(q1_path, img_bgr)
    print(f"Saved: {q1_path}")

    # -------------------------------------------------------------
    # Question 2: Draw solid red vertical line at center column x = W // 2
    # -------------------------------------------------------------
    print("\n--- Question 2: Red Center Line ---")
    img_q2 = img_bgr.copy()
    center_x = width // 2  # 512 // 2 = 256
    # In BGR: Red is (0, 0, 255)
    img_q2[:, center_x, :] = [0, 0, 255]

    q2_path = os.path.join(output_dir, "HW1_Q2.bmp")
    cv2.imwrite(q2_path, img_q2)
    print(f"Drawn red vertical line at column x = {center_x}")
    print(f"Saved: {q2_path}")

    # -------------------------------------------------------------
    # Question 3: Draw blue line of thickness 2 from (100,100) to (200,200)
    # -------------------------------------------------------------
    print("\n--- Question 3: Blue Line (thickness 2) ---")
    img_q3 = img_q2.copy()
    # In OpenCV, cv2.line uses (x, y) coordinates. Blue in BGR is (255, 0, 0)
    # Drawing from (100, 100) to (200, 200) with thickness 2:
    cv2.line(img_q3, (100, 100), (200, 200), (255, 0, 0), thickness=2)

    q3_path = os.path.join(output_dir, "HW1_Q3.bmp")
    cv2.imwrite(q3_path, img_q3)
    print("Drawn blue line of thickness 2 from (100, 100) to (200, 200)")
    print(f"Saved: {q3_path}")

    # -------------------------------------------------------------
    # Question 4: Solid green rectangle top-left (50,50) to bottom-right (100,100) inclusive
    # -------------------------------------------------------------
    print("\n--- Question 4: Green Rectangle via Direct Array Assignment ---")
    img_q4 = img_q3.copy()
    # Inclusive range [50, 100] means slice 50:101 for rows (y) and cols (x)
    # Green in BGR is [0, 255, 0]
    img_q4[50:101, 50:101, :] = [0, 255, 0]

    q4_path = os.path.join(output_dir, "HW1_Q4.bmp")
    cv2.imwrite(q4_path, img_q4)
    print("Set rectangle [50:101, 50:101] to solid green [0, 255, 0] (BGR)")
    print(f"Saved: {q4_path}")

    # Save a summary plot of Q1-Q4 progression
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Q1: Original Lena (512x512)")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_q2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Q2: Red Center Line (x=256)")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img_q3, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Q3: + Blue Line (100,100)-(200,200)")
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(img_q4, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Q4: + Green Box (50,50)-(100,100)")
    axes[3].axis('off')

    plt.tight_layout()
    q1_4_plot_path = os.path.join(output_dir, "HW1_Q1_to_Q4_summary.png")
    plt.savefig(q1_4_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved summary comparison plot: {q1_4_plot_path}")

    # -------------------------------------------------------------
    # Question 5: Color Space Decompositions & Channel Visualizations
    # -------------------------------------------------------------
    print("\n--- Question 5: Color Spaces (RGB, HSV, CIELAB, YCrCb) ---")
    
    # 1. RGB Decomposition
    # Note: img_bgr has channels [B, G, R]
    b_chan = img_bgr[:, :, 0]
    g_chan = img_bgr[:, :, 1]
    r_chan = img_bgr[:, :, 2]

    # 2. HSV Decomposition
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan = img_hsv[:, :, 0]
    s_chan = img_hsv[:, :, 1]
    v_chan = img_hsv[:, :, 2]

    # 3. CIELAB Decomposition
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan = img_lab[:, :, 0]
    a_chan = img_lab[:, :, 1]
    b_lab_chan = img_lab[:, :, 2]

    # 4. YCrCb Decomposition
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y_chan = img_ycrcb[:, :, 0]
    cr_chan = img_ycrcb[:, :, 1]
    cb_chan = img_ycrcb[:, :, 2]

    # Create detailed 4x3 comparison grid
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    # Row 1: RGB
    axes[0, 0].imshow(r_chan, cmap='gray')
    axes[0, 0].set_title("RGB: Red (R) Channel")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(g_chan, cmap='gray')
    axes[0, 1].set_title("RGB: Green (G) Channel")
    axes[0, 1].axis('off')
    axes[0, 2].imshow(b_chan, cmap='gray')
    axes[0, 2].set_title("RGB: Blue (B) Channel")
    axes[0, 2].axis('off')

    # Row 2: HSV
    axes[1, 0].imshow(h_chan, cmap='gray')
    axes[1, 0].set_title("HSV: Hue (H) Channel [0..179]")
    axes[1, 0].axis('off')
    axes[1, 1].imshow(s_chan, cmap='gray')
    axes[1, 1].set_title("HSV: Saturation (S) Channel [0..255]")
    axes[1, 1].axis('off')
    axes[1, 2].imshow(v_chan, cmap='gray')
    axes[1, 2].set_title("HSV: Value/Brightness (V) Channel [0..255]")
    axes[1, 2].axis('off')

    # Row 3: CIELAB
    axes[2, 0].imshow(l_chan, cmap='gray')
    axes[2, 0].set_title("CIELAB: Lightness (L*) Channel [0..255]")
    axes[2, 0].axis('off')
    axes[2, 1].imshow(a_chan, cmap='gray')
    axes[2, 1].set_title("CIELAB: a* (Green-Red Axis) [0..255]")
    axes[2, 1].axis('off')
    axes[2, 2].imshow(b_lab_chan, cmap='gray')
    axes[2, 2].set_title("CIELAB: b* (Blue-Yellow Axis) [0..255]")
    axes[2, 2].axis('off')

    # Row 4: YCrCb
    axes[3, 0].imshow(y_chan, cmap='gray')
    axes[3, 0].set_title("YCrCb: Luma (Y) Channel")
    axes[3, 0].axis('off')
    axes[3, 1].imshow(cr_chan, cmap='gray')
    axes[3, 1].set_title("YCrCb: Cr (Red-difference Chroma)")
    axes[3, 1].axis('off')
    axes[3, 2].imshow(cb_chan, cmap='gray')
    axes[3, 2].set_title("YCrCb: Cb (Blue-difference Chroma)")
    axes[3, 2].axis('off')

    plt.tight_layout()
    q5_plot_path = os.path.join(output_dir, "HW1_Q5_color_spaces_grid.png")
    plt.savefig(q5_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved color space comparison grid: {q5_plot_path}")

    # Also save individual channel images for report reference
    channels_dict = {
        "R": r_chan, "G": g_chan, "B": b_chan,
        "HSV_H": h_chan, "HSV_S": s_chan, "HSV_V": v_chan,
        "LAB_L": l_chan, "LAB_A": a_chan, "LAB_B": b_lab_chan,
        "YCRCB_Y": y_chan, "YCRCB_CR": cr_chan, "YCRCB_CB": cb_chan
    }
    for name, c_img in channels_dict.items():
        cv2.imwrite(os.path.join(output_dir, f"HW1_channel_{name}.png"), c_img)

    print("\nHomework 1 execution completed successfully!")

if __name__ == "__main__":
    main()
