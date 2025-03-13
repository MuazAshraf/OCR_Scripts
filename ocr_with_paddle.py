import os
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR

def preprocess_image(image):
    """
    Preprocess the image to improve OCR results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    return denoised

def extract_text_with_paddle(image_path, output_file):
    """
    Extract text from broadcast images using PaddleOCR
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
            
        # Initialize PaddleOCR with Hindi and English support
        ocr = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False, 
                        show_log=False, enable_mkldnn=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Save preprocessed image for debugging (optional)
        debug_dir = os.path.dirname(output_file)
        cv2.imwrite(os.path.join(debug_dir, f"{Path(image_path).stem}_processed.jpg"), processed_image)
        
        # Run OCR on both original and processed images and combine results
        result_original = ocr.ocr(image, cls=True)
        result_processed = ocr.ocr(processed_image, cls=True)
        
        # Combine and sort results by vertical position
        all_results = []
        
        # Process original image results
        if result_original and result_original[0]:
            for line in result_original[0]:
                if len(line) >= 2 and line[1] and len(line[1]) >= 2:
                    text, confidence = line[1]
                    if confidence > 0.5 and text.strip():  # Filter by confidence
                        # Get vertical position (y-coordinate)
                        box = line[0]
                        y_position = sum(point[1] for point in box) / 4
                        all_results.append((text, y_position, confidence))
        
        # Process enhanced image results
        if result_processed and result_processed[0]:
            for line in result_processed[0]:
                if len(line) >= 2 and line[1] and len(line[1]) >= 2:
                    text, confidence = line[1]
                    if confidence > 0.5 and text.strip():  # Filter by confidence
                        # Get vertical position (y-coordinate)
                        box = line[0]
                        y_position = sum(point[1] for point in box) / 4
                        all_results.append((text, y_position, confidence))
        
        # Remove duplicates (keep highest confidence)
        unique_results = {}
        for text, y_pos, conf in all_results:
            text_lower = text.lower()
            if text_lower not in unique_results or conf > unique_results[text_lower][1]:
                unique_results[text_lower] = (text, conf, y_pos)
        
        # Sort by vertical position
        sorted_results = sorted(unique_results.values(), key=lambda x: x[2])
        
        # Format the results
        formatted_text = ""
        for text, conf, _ in sorted_results:
            formatted_text += f"{text}\n"
        
        # Save the results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        
        print(f"OCR completed for {image_path}. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def main():
    # Set default paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    default_input = os.path.join(parent_dir, "broadcast_images")
    default_output = os.path.join(parent_dir, "ocr_results")
    
    # Process input
    input_path = default_input
    output_path = default_output
    
    print(f"Processing images from: {input_path}")
    print(f"Saving results to: {output_path}")
    
    # Process input
    if os.path.isfile(input_path):
        # Process single file
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{Path(input_path).stem}_paddle_result.txt")
        extract_text_with_paddle(input_path, output_file)
    elif os.path.isdir(input_path):
        # Process directory
        os.makedirs(output_path, exist_ok=True)
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(input_path) 
                      if os.path.isfile(os.path.join(input_path, f)) 
                      and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if not image_files:
            print(f"No image files found in {input_path}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for file in image_files:
            print(f"\nProcessing {file}...")
            input_file = os.path.join(input_path, file)
            output_file = os.path.join(output_path, f"{Path(file).stem}_paddle_result.txt")
            extract_text_with_paddle(input_file, output_file)
            print(f"Completed {file}")
    else:
        print(f"Error: Input path {input_path} does not exist")
        print(f"Please create the folder {input_path} and add images to it")

if __name__ == "__main__":
    main()
