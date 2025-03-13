from PIL import Image
import os
from pathlib import Path

def convert_jpg_to_pdf():
    # Define input and output directories
    input_dir = Path("enhanced_output")
    output_dir = Path("pdf_output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get all jpg files from input directory
    jpg_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG"))
    
    if not jpg_files:
        print("No JPG files found in enhanced_output directory")
        return
    
    # Process each jpg file
    for jpg_path in jpg_files:
        try:
            # Open the image
            with Image.open(jpg_path) as img:
                # Convert to RGB if necessary (in case of RGBA images)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create PDF filename
                pdf_path = output_dir / f"{jpg_path.stem}.pdf"
                
                # Save as PDF
                img.save(pdf_path, "PDF", resolution=100.0)
                print(f"Converted {jpg_path.name} to {pdf_path.name}")
                
        except Exception as e:
            print(f"Error converting {jpg_path.name}: {str(e)}")
    
    print(f"\nAll conversions completed. PDFs saved in {output_dir}")

if __name__ == "__main__":
    convert_jpg_to_pdf()