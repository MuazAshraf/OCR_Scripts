import os
import cv2

def split_two_page_images(
    input_dir="two_page_images", 
    output_dir="split_pages"
):
    """
    Splits each two-page image in `input_dir` into two separate images
    (left and right), and saves them in `output_dir`.

    This script does NOT perform any automatic brightness/contrast or
    margin-cropping. You can do those steps manually afterward.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over files in the input directory
    for filename in os.listdir(input_dir):
        # Only process common image file extensions
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            continue
        
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Skip invalid images
        if image is None:
            print(f"Skipping invalid image: {filename}")
            continue

        h, w = image.shape[:2]

        # Compute midpoint
        mid = w // 2

        # Split into left and right
        left_page = image[:, :mid]
        right_page = image[:, mid:]

        # Build output paths
        base, ext = os.path.splitext(filename)
        left_path = os.path.join(output_dir, f"{base}_left{ext}")
        right_path = os.path.join(output_dir, f"{base}_right{ext}")

        # Save results
        cv2.imwrite(left_path, left_page)
        cv2.imwrite(right_path, right_page)

        print(f"Split {filename} -> {left_path}, {right_path}")

if __name__ == "__main__":
    split_two_page_images(
        input_dir="italian_images",   # <-- adjust as needed
        output_dir="output_split_pages"    # <-- adjust as needed
    )
