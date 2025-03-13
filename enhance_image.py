import os
import cv2
import numpy as np

# -- Tweakable Parameters --

# How much extra margin do you want around the detected text region? (in pixels)
TOP_MARGIN = 30
BOTTOM_MARGIN = 30
LEFT_MARGIN = 100
RIGHT_MARGIN = 80

# CLAHE parameters for “auto contrast”
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Optional sharpen kernel
SHARPEN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype="float32")

def auto_crop_and_enhance(input_path, output_path):
    # 1. Load the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not open {input_path}; skipping.")
        return
    
    # 2. Convert to grayscale for boundary detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Threshold to find text‐heavy regions (using OTSU or ADAPTIVE)
    #    Feel free to try adaptiveThreshold if OTSU is not robust enough
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # In many scans, text is dark on lighter background, so the “text area” might be the black region.
    # If your text is white on black, invert here:
    # thresh = cv2.bitwise_not(thresh)
    
    # 4. Find the biggest contour, which presumably corresponds to the main page/text block
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in {input_path}; skipping.")
        return
    
    # 5. Get bounding box of the LARGEST contour by area
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # 6. Apply some margins (so we don’t clip too close)
    x1 = max(0, x - LEFT_MARGIN)
    y1 = max(0, y - TOP_MARGIN)
    x2 = min(img.shape[1], x + w + RIGHT_MARGIN)
    y2 = min(img.shape[0], y + h + BOTTOM_MARGIN)
    
    # 7. Crop to that bounding box
    cropped = img[y1:y2, x1:x2]
    
    # 8. Convert to LAB and apply CLAHE on L‐channel for “auto contrast”
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 9. Optional: Sharpen the result
    # enhanced = cv2.filter2D(enhanced, ddepth=-1, kernel=SHARPEN_KERNEL)
    
    # 10. Save
    cv2.imwrite(output_path, enhanced)
    print(f"Saved to {output_path}")

def batch_process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
        auto_crop_and_enhance(in_path, out_path)

if __name__ == "__main__":
    input_dir = "enhance_input"
    output_dir = "enhanced_output"
    
    batch_process_images(input_dir, output_dir)
    print("All done!")
