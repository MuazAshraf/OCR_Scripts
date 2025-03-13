import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Any

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# Configuration - Replace with your actual values
AZURE_ENDPOINT = "" 
AZURE_API_KEY = ""
INPUT_FOLDER = ""           
OUTPUT_FOLDER = ""                

class SimpleDictionaryOCR:
    def __init__(self, endpoint: str, api_key: str, output_dir: str = "ocr_results"):
        """
        Initialize the OCR client with Azure Form Recognizer credentials.
        
        Args:
            endpoint: Your Azure Form Recognizer endpoint
            api_key: Your Azure Form Recognizer API key
            output_dir: Directory to save OCR results
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.output_dir = output_dir
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(api_key)
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def process_image(self, image_path: str):
        """
        Process a single dictionary image and extract just the text.
        
        Args:
            image_path: Path to the dictionary image
        """
        print(f"Processing image: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Get file name without extension for output files
        file_name = Path(image_path).stem
        
        # Check if output file already exists (double-check)
        output_path = os.path.join(self.output_dir, f"{file_name}_for_ai.txt")
        if os.path.exists(output_path):
            print(f"Skipping {file_name} - already processed")
            return
        
        # Open the image file
        with open(image_path, "rb") as image_file:
            # Use Read model for recognition
            poller = self.document_analysis_client.begin_analyze_document(
                "prebuilt-read", image_file
            )
            
            result = poller.result()
        
        # Extract and save just the text content
        self._save_text_results(result, file_name)
            
    def process_directory(self, dir_path: str):
        """
        Process all images in a directory, skipping those already processed.
        
        Args:
            dir_path: Directory containing dictionary images
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created empty directory: {dir_path}")
            return
            
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_files = [f for f in os.listdir(dir_path) 
                      if os.path.isfile(os.path.join(dir_path, f)) 
                      and any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_files:
            print(f"No image files found in {dir_path}")
            return
            
        print(f"Found {len(image_files)} images in {dir_path}")
        
        # Filter out already processed files
        new_files = []
        skipped_files = []
        
        for file in image_files:
            file_path = os.path.join(dir_path, file)
            file_name = Path(file_path).stem
            output_path = os.path.join(self.output_dir, f"{file_name}_for_ai.txt")
            
            # Check if output file already exists
            if os.path.exists(output_path):
                skipped_files.append(file)
            else:
                new_files.append(file)
        
        if skipped_files:
            print(f"Skipping {len(skipped_files)} already processed files: {', '.join(skipped_files[:5])}")
            if len(skipped_files) > 5:
                print(f"... and {len(skipped_files) - 5} more")
        
        if not new_files:
            print("No new files to process.")
            return
        
        print(f"Processing {len(new_files)} new images...")
        
        success_count = 0
        for file in new_files:
            file_path = os.path.join(dir_path, file)
            try:
                self.process_image(file_path)
                success_count += 1
                print(f"✓ Successfully processed {file}")
            except Exception as e:
                print(f"✗ Error processing {file}: {str(e)}")
        
        print(f"\nSuccessfully processed {success_count} out of {len(new_files)} new images.")
    
    def _save_text_results(self, result, file_name: str):
        """
        Extract and save just the text content from OCR results.
        
        Args:
            result: OCR result from Form Recognizer
            file_name: Base name for the output file
        """
        # Extract just the text content by page
        all_text = ""
        
        for page_idx, page in enumerate(result.pages):
            page_text = ""
            
            # Collect all lines with their original formatting
            for line in page.lines:
                # Get text based on SDK version
                line_text = ""
                if hasattr(line, 'content'):
                    line_text = line.content
                elif hasattr(line, 'text'):
                    line_text = line.text
                
                if line_text:
                    page_text += line_text + "\n"
            
            all_text += page_text + "\n\n"
        
        # Save ONLY the OCR text file
        output_path = os.path.join(self.output_dir, f"{file_name}_for_ai.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(all_text)
        
        print(f"OCR text saved to: {output_path}")


# Main code - runs automatically when you execute the script
if __name__ == "__main__":
    print("Starting simple OCR processing for Italian dictionary images...")
    
    # Create OCR processor with your credentials
    ocr = SimpleDictionaryOCR(
        endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        output_dir=OUTPUT_FOLDER
    )
    
    # Process all images in the input folder
    print(f"Looking for images in {INPUT_FOLDER}...")
    ocr.process_directory(INPUT_FOLDER)
    
    print("\nOCR processing complete.") 