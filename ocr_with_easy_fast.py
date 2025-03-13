import easyocr
import cv2
import os
from pathlib import Path
import time

def extract_text_fast(image_path, output_file=None):
    """
    Extract text from broadcast images using EasyOCR with a fast approach
    """
    # Initialize the reader with multiple languages
    # Only load it once to save time
    global reader
    if 'reader' not in globals():
        print("Initializing EasyOCR (this will take a moment the first time)...")
        reader = easyocr.Reader(['en', 'hi'], gpu=True)  # Use GPU if available
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Resize large images to improve speed
    height, width = image.shape[:2]
    max_dimension = 1280  # Limit maximum dimension for speed
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
        print(f"Resized image from {width}x{height} to {new_width}x{new_height} for faster processing")
    
    # Single pass OCR with optimized settings
    print("Extracting text (single pass)...")
    start_time = time.time()
    
    # Use paragraph=True for faster processing
    results = reader.readtext(
        image, 
        paragraph=True,  # Group text into paragraphs for faster processing
        detail=0,        # Only return text, not bounding boxes
        batch_size=4,    # Increase batch size for faster processing
        decoder='greedy',# Use faster decoder
        beamWidth=5,     # Smaller beam width for faster processing
        contrast_ths=0.1,# Lower threshold to detect more text
        adjust_contrast=0.5,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4
    )
    
    end_time = time.time()
    print(f"Text extraction completed in {end_time - start_time:.2f} seconds")
    
    # Format the results
    final_text = '\n'.join(results)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
        print(f"Results saved to {output_file}")
    
    return final_text

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
        output_file = os.path.join(output_path, f"{Path(input_path).stem}_result.txt")
        extract_text_fast(input_path, output_file)
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
            output_file = os.path.join(output_path, f"{Path(file).stem}_result.txt")
            extract_text_fast(input_file, output_file)
            print(f"Completed {file}")
    else:
        print(f"Error: Input path {input_path} does not exist")
        print(f"Please create the folder {input_path} and add images to it")

if __name__ == "__main__":
    main()