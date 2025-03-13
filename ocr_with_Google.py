import re
import json
import unicodedata
from google.cloud import vision
import os
from typing import List, Dict
from pdf2image import convert_from_path
import io

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Keys.json'

def clean_text(text):
    """Clean OCR text while preserving Italian characters."""
    text = unicodedata.normalize('NFKC', text)
    # Remove unwanted characters (keeping Italian letters, punctuation, and hyphens)
    text = re.sub(r'[^\w\s,.;:()\[\]à-ÿ-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_meanings(text: str) -> List[str]:
    """Extract meanings from entry text by splitting on common separators."""
    # Remove the lemma (assumes the first word is the lemma)
    first_space = text.find(' ')
    if first_space > 0:
        text = text[first_space:].strip()
    # Split by period, semicolon or pipe if a new capital letter starts a segment
    parts = re.split(r'[.;|](?=\s*[A-ZÀ-Ö])', text)
    meanings = []
    for part in parts:
        part = part.strip(' ,.;|')
        if len(part) > 2 and not part.startswith('-'):
            meanings.append(part)
    return meanings

def process_page(image_path: str) -> List[Dict]:
    """
    Process a single dictionary page using OCR blocks.
    This version uses a refined heuristic to detect headwords and 
    accumulates blocks until a new entry is detected.
    """
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    print("\nRaw OCR Text from {}:".format(image_path))
    print(response.full_text_annotation.text)

    print("\nBlock by block analysis:")
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_text = ' '.join(''.join(symbol.text for symbol in word.symbols)
                                  for paragraph in block.paragraphs
                                  for word in paragraph.words)
            print("\n---Block Start---\n{}\n---Block End---".format(block_text.strip()))

    # Updated block-based extraction
    entries = []
    current_entry = None
    current_content = []

    # This pattern looks for a headword:
    # Optional leading plus sign, optional whitespace, then a word starting with a letter (or accented letter)
    entry_match_pattern = re.compile(r'^(?:\+?\s*[A-ZÀ-Ö][a-zà-ö]+)', re.IGNORECASE)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_text = ' '.join(''.join(symbol.text for symbol in word.symbols)
                                  for paragraph in block.paragraphs
                                  for word in paragraph.words).strip()
            if not block_text:
                continue

            # If this block appears to start a new entry:
            if entry_match_pattern.match(block_text):
                tokens = block_text.split()
                # If the first token is just a stray symbol like "+" or "-", use the next token
                if tokens[0] in ['+', '-', '±'] and len(tokens) > 1:
                    lemma = tokens[1].rstrip(',.;')
                else:
                    lemma = tokens[0].rstrip(',.;')
                # Save previous entry if one exists
                if current_entry is not None:
                    combined_text = ' '.join(current_content)
                    entries.append({
                        'lemma': current_entry,
                        'ocr': combined_text,
                        'significati': extract_meanings(combined_text)
                    })
                current_entry = lemma
                current_content = [block_text]
                print("\nNew entry found: {}".format(current_entry))
                print("Block text: {}".format(block_text))
            else:
                if current_entry:
                    current_content.append(block_text)
                    print("Continuing {} with: {}".format(current_entry, block_text))

    # Save the last accumulated entry
    if current_entry:
        combined_text = ' '.join(current_content)
        entries.append({
            'lemma': current_entry,
            'ocr': combined_text,
            'significati': extract_meanings(combined_text)
        })

    print("\nFound {} entries in {}".format(len(entries), image_path))
    print("Entries found:", [e['lemma'] for e in entries])
    return entries

def process_dictionary(input_path: str, output_dir: str):
    """Process dictionary pages and write entries."""
    all_entries = []
    
    # Check if input is a directory of images or a PDF file
    if os.path.isdir(input_path):
        # Process image files in directory
        image_files = [f for f in os.listdir(input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in sorted(image_files):
            print(f"\nProcessing image {image_file}")
            try:
                image_path = os.path.join(input_path, image_file)
                page_entries = process_page(image_path)
                all_entries.extend(page_entries)
            except Exception as e:
                print(f"Error processing image {image_file}: {str(e)}")
    else:
        # Process PDF file
        try:
            # Check if poppler is available
            from pdf2image import pdfinfo_from_path
            # Process first few pages
            for page_num in range(1, 6):  # Process pages 1-5
                print(f"\nProcessing page {page_num}")
                try:
                    page_entries = process_pdf_page(input_path, page_num)
                    all_entries.extend(page_entries)
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
        except ImportError:
            print("Poppler not found. Please install poppler-utils for PDF processing.")
            print("For Windows: Download from http://blog.alivate.com.au/poppler-windows/")
            print("For Linux: sudo apt-get install poppler-utils")
            print("For Mac: brew install poppler")
            return

    # Write entries to JSONL file
    output_file = os.path.join(output_dir, "all_entries.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            json.dump(entry, ensure_ascii=False)
            f.write('\n')
    print(f"\nWritten {len(all_entries)} entries to {output_file}")

def process_pdf_page(pdf_path, page_number):
    """Process a single page from PDF"""
    # Convert PDF page to image
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    if not images:
        return []
    
    # Get the page image
    page_image = images[0]
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    page_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Use Google Vision API
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img_byte_arr)
    response = client.document_text_detection(image=image)

    # Updated block-based extraction
    entries = []
    current_entry = None
    current_content = []

    # Pattern for dictionary entries
    entry_match_pattern = re.compile(r'^(?:\+?\s*[A-ZÀ-Ö][a-zà-ö]+)', re.IGNORECASE)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_text = ' '.join(''.join(symbol.text for symbol in word.symbols)
                                for paragraph in block.paragraphs
                                for word in paragraph.words).strip()
            if not block_text:
                continue

            print(f"\n---Block Start---\n{block_text}\n---Block End---")

            if entry_match_pattern.match(block_text):
                tokens = block_text.split()
                lemma = tokens[1].rstrip(',.;') if tokens[0] in ['+', '-', '±'] and len(tokens) > 1 else tokens[0].rstrip(',.;')
                
                if current_entry is not None:
                    combined_text = ' '.join(current_content)
                    entries.append({
                        'lemma': current_entry,
                        'ocr': combined_text,
                        'significati': extract_meanings(combined_text)
                    })
                current_entry = lemma
                current_content = [block_text]
                print(f"\nNew entry found: {current_entry}")
                print(f"Block text: {block_text}")
            elif current_entry:
                current_content.append(block_text)
                print(f"Continuing {current_entry} with: {block_text}")

    # Save the last entry
    if current_entry:
        combined_text = ' '.join(current_content)
        entries.append({
            'lemma': current_entry,
            'ocr': combined_text,
            'significati': extract_meanings(combined_text)
        })

    print(f"\nFound {len(entries)} entries")
    print("Entries found:", [e['lemma'] for e in entries])
    return entries

def main():
    input_path = "italian_images_test"  # Can be either a directory of images or a PDF file
    output_dir = "output_json"
    
    if not os.path.exists(input_path):
        print(f"Input path not found: {input_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    process_dictionary(input_path, output_dir)

if __name__ == "__main__":
    main()
