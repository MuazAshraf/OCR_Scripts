import os
import re
import json
import logging
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
from pathlib import Path
import unicodedata

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string."""
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

def create_overlapping_chunks(text: str, max_tokens: int = 3000, overlap_tokens: int = 500) -> List[str]:
    """Create overlapping chunks of text, each with a maximum token count."""
    encoder = tiktoken.encoding_for_model("gpt-4")
    tokens = encoder.encode(text)
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # Calculate end index for this chunk
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # Decode chunk back to text
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = encoder.decode(chunk_tokens)
        
        # Add to chunks
        chunks.append(chunk_text)
        
        # Move start index for next chunk, ensuring overlap
        if end_idx == len(tokens):
            break
        
        start_idx = end_idx - overlap_tokens
    
    return chunks

def preprocess_ocr_text(text: str) -> str:
    """Apply preprocessing to fix common OCR issues before extraction."""
    processed_text = text
    
    # 1. Fix split words across line breaks (very common OCR issue)
    processed_text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', processed_text)
    
    # 2. Fix cases with multiple hyphens/line breaks that may fragment words
    processed_text = re.sub(r'(\w+)-\s*\n\s*-(\w+)', r'\1\2', processed_text)
    
    # 3. Normalize spacing around entry markers
    processed_text = re.sub(r'\|\s+', r'| ', processed_text)
    processed_text = re.sub(r'\s+\|', r' |', processed_text)
    
    # 4. Improve derivative marker formatting - critical for proper nesting
    processed_text = re.sub(r'\|\s*-', r'| -', processed_text)  # Standard derivative
    processed_text = re.sub(r'\|\s*\+', r'| +', processed_text)  # Alternative form
    processed_text = re.sub(r'\|\s*\|', r' || ', processed_text)  # Double derivative
    
    # 5. Ensure derivative chains are properly associated with main entries
    processed_text = re.sub(r'([\n\r])\s*-(\w+)', r'\1-\2', processed_text)  # Keep derivatives with entries
    processed_text = re.sub(r'^\s*-(\w+)', r'-\1', processed_text, flags=re.MULTILINE)  # Line-start derivatives
    
    # 6. Remove page numbers
    processed_text = re.sub(r'^\d+\s*$', '', processed_text, flags=re.MULTILINE)
    
    # 7. Fix common OCR symbol artifacts that confuse extraction
    processed_text = re.sub(r'[#@&]', '', processed_text)  # Common OCR artifacts
    processed_text = re.sub(r'\[\s*|\s*\]', ' ', processed_text)  # Remove brackets
    processed_text = re.sub(r'\{\s*|\s*\}', ' ', processed_text)  # Remove braces
    processed_text = re.sub(r'["""„«»]', '"', processed_text)  # Normalize quotes
    
    # 8. Fix spaces in verb patterns (common in Italian dictionaries)
    processed_text = re.sub(r'(\w+)\s+(\w{1,3})\s+ere\b', r'\1\2ere', processed_text)
    processed_text = re.sub(r'(\w+)\s+(\w{1,3})\s+ire\b', r'\1\2ire', processed_text)
    processed_text = re.sub(r'(\w+)\s+(\w{1,3})\s+are\b', r'\1\2are', processed_text)
    
    # 9. Fix lemmas with fragments (especially critical for "rentissimo" type issues)
    processed_text = re.sub(r'([a-zàèéìòóù]+)\s+([a-zàèéìòóù]+entissimo)', r'\1\2', processed_text)
    
    # 10. Normalize derivative chains (especially for entries like "acrocòro")
    processed_text = re.sub(r'\|\|\s*-(\w+)', r'| -\1', processed_text)  # Fix double pipe derivatives
    processed_text = re.sub(r'\|\s*-(\w+),\s*f\.\s*[*#]', r'| -\1, f. ', processed_text)  # Fix derivative marker spacing
    
    # 11. Fix "ofobia", "ografia", "omania" type derivatives to ensure proper association
    processed_text = re.sub(r'-\s*o(\w+),\s*[*#]', r'-o\1, ', processed_text)  # Fix o-prefixed derivatives
    
    # 12. Standardize punctuation and spacing around significati
    processed_text = re.sub(r'\.\s*\|\s*', r'. | ', processed_text)  # Add space after periods before pipe
    processed_text = re.sub(r',\s*\|\s*', r', | ', processed_text)  # Add space after commas before pipe
    
    # 13. Improve formatting for multiple meanings
    processed_text = re.sub(r'\|\s+(\d+\.)', r'| \1', processed_text)  # Fix numbered meanings
    processed_text = re.sub(r'\|\s+([\w])', r'| \1', processed_text)  # Fix start of meanings
    
    # 14. Normalize spacing for headwords (lemmas)
    processed_text = re.sub(r'(\n[a-zàèéìòóù]\w+),', r'\1, ', processed_text)
    
    # 15. CRITICAL FIX: Handle page-beginning definitions (avoid incorrect associations)
    # If the text starts with a continuing definition (no clear lemma at beginning)
    # Add a special marker to help the extraction process recognize this
    lines = processed_text.split('\n')
    if len(lines) > 0:
        first_line = lines[0].strip()
        # Check if first line looks like a continuation (no lemma pattern)
        if not re.match(r'^[a-zàèéìòóù]\w+\b', first_line) and '|' in first_line:
            # This looks like a continuation of a definition from previous page
            # Mark it specially so extraction can handle it
            processed_text = "###PAGE_CONTINUATION###\n" + processed_text
    
    return processed_text

def extract_entries_from_chunk(chunk_text: str, retry_count: int = 0) -> List[Dict[str, Any]]:
    """Extract dictionary entries from a chunk of text using GPT-4."""
    try:
        # Apply preprocessing to the chunk text
        chunk_text = preprocess_ocr_text(chunk_text)
        
        # Create a comprehensive system prompt
        system_prompt = """
        You are an expert linguist and dictionary analyst specialized in Italian dictionary texts.
        Extract all dictionary entries from the following OCR text of a dictionary page.

        IMPORTANT -> You must extract all entries from the text. Do not miss any no matter what happens. The entries should not be less then 15 for every individual ocr file PLEASE DO NOT MISS ANYTHING. 
        IMPORTANT -> Some Ocr files are croped and some are not. Please be careful and do not miss any entries.

        Format each entry as a JSON object with these fields:
        - lemma: The word or phrase being defined, including any grammatical notes
        - OCR: The complete original OCR text for the entry
        - significati: An array of numbered meanings (1., 2., 3., etc.)
        - derivati: An array of derivatives with their modificatore and significati

        CRITICAL RULES FOR EXTRACTION:
        1. ALWAYS number the significati sequentially (1., 2., 3., etc.)
        2. IF the original text has numbered meanings, preserve those exact numbers
        3. IF the original text does NOT have numbered meanings, ADD proper numbering (1., 2., 3.)
        4. NEVER create separate entries from derivatives - they must be nested within their main entry
        5. Each derivati must have both "modificatore" (the word form) and "significati" (meanings array)
        6. Once you have extracted the entry RE-SCAN the text to ensure that you have not missed anything. This is a very critical step Make sure you don't neglect this. 

        IMPORTANT FORMAT RULES:
        - Main entries begin with lowercase word at the start of a line 
        - Entry boundaries are indicated by new entries starting at the margin
        - Meanings are separated by | symbol or numbering
        - Derivatives are marked with "-", "+", or "||" symbols and belong to their preceding entry

        EXAMPLE OF CORRECT EXTRACTION:
        For input:
        "addolcire, a. (-isco). Far dolce. Liberare dell'amarezza. | Temperare. | metalli."

        Correct output:
        {
          "lemma": "addolcire",
          "ocr": "addolcire, a. (-isco). Far dolce. Liberare dell'amarezza. | Temperare. | metalli.",
          "significati": [
            "1. Far dolce. Liberare dell'amarezza.",
            "2. Temperare.",
            "3. metalli."
          ],
          "derivati": []
        }

        Response in STRICTLY VALID JSON with an "entries" array containing all extracted entries.
        """
        
        # Make API call with explicit JSON formatting
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract all dictionary entries from this OCR text and return ONLY valid JSON:\n\n{chunk_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse response with error handling (rest of the function remains the same)
        content = response.choices[0].message.content
        
        # Try to fix common JSON issues before parsing
        try:
            # First try direct parsing
            data = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to fix common JSON errors
            logger.info("Attempting to fix malformed JSON...")
            fixed_content = content
            
            # Replace incorrect escape sequences
            fixed_content = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', fixed_content)
            
            # Fix unterminated strings by adding missing quotes
            fixed_content = re.sub(r'("[^"\\]*(?:\\.[^"\\]*)*):(?!\s*[{[])', r'\1":', fixed_content)
            
            # Try parsing again with fixed content
            try:
                data = json.loads(fixed_content)
                logger.info("Successfully fixed and parsed JSON")
            except json.JSONDecodeError as e:
                logger.error(f"Still invalid JSON after fixes: {e}")
                
                # Ultimate fallback - try extracting entries with regex
                if retry_count >= 2:
                    logger.warning("Using regex fallback extraction")
                    return extract_with_regex(chunk_text)
                
                # Retry with more explicit instructions
                logger.info(f"Retrying extraction with simpler format (attempt {retry_count + 1})")
                time.sleep(2)
                return extract_entries_from_chunk(chunk_text, retry_count + 1)
        
        # Ensure we have an 'entries' key
        if 'entries' not in data:
            logger.warning(f"Response missing 'entries' key: {list(data.keys())}")
            return []
        
        return data['entries']
    
    except Exception as e:
        logger.error(f"Error in extraction: {e}")
        return []

def extract_with_regex(text: str) -> List[Dict[str, Any]]:
    """Emergency fallback method using regex to extract entries when JSON parsing fails."""
    entries = []
    
    # Improved pattern (captures more entry types):
    entry_pattern = r'([a-zàèéìòóù]+(?:\s*,\s*[+]?[a-zàèéìòóù-]+)?(?:\s+[a-zàèéìòóù]+)?)\s*,\s*(?:[a-z]+\.|\+[a-z]+\.|\w+\s+[a-z]+\.|[^,.]+\.)'
    
    matches = re.finditer(entry_pattern, text, re.DOTALL)
    
    for match in matches:
        lemma = match.group(1).strip()
        full_text = match.group(0).strip()
        
        # Extract meanings (after first | symbol)
        significati = []
        significati_text = re.split(r'\|', full_text)[1:] if '|' in full_text else []
        for sig in significati_text:
            if sig.strip():
                significati.append(sig.strip())
        
        # Look for derivatives
        derivati = []
        deriv_pattern = r'-(\w+),\s+([^|]+)'
        for deriv_match in re.finditer(deriv_pattern, full_text):
            derivati.append({
                "modificatore": f"-{deriv_match.group(1)}",
                "significati": [deriv_match.group(2).strip()]
            })
        
        entries.append({
            "lemma": lemma,
            "ocr": full_text,
            "significati": significati if significati else [""],
            "derivati": derivati
        })
    
    logger.info(f"Regex fallback extracted {len(entries)} entries")
    return entries

def extract_entries_fallback(chunk_text: str) -> List[Dict[str, Any]]:
    """Alternative extraction method for problematic chunks."""
    try:
        # Use a more structured, step-by-step approach with Claude model
        system_prompt = """
        You are an expert Italian lexicographer. Extract dictionary entries from the text below.
        IMPORTANT -> You must extract all entries from the text. Do not miss any no matter what happens. The entries should not be less then 15 for every individual ocr file PLEASE DO NOT MISS ANYTHING. 
        IMPORTANT -> Some Ocr files are croped and some are not. Please be careful and do not miss any entries.
        
        FOLLOW THESE STEPS EXACTLY:
        1. Identify each main entry (headword) in the text.
        2. For each entry, extract:
           - The headword and grammatical information
           - The complete original text
           - All meaning definitions
           - Any derivatives or related forms
        
        FORMAT YOUR RESPONSE AS VALID JSON:
        {
          "entries": [
            {
              "lemma": "word",
              "ocr": "original text",
              "significati": ["meaning 1", "meaning 2"],
              "derivati": [
                {
                  "modificatore": "-form, part of speech",
                  "significati": ["meaning"]
                }
              ]
            }
          ]
        }
        
        Be extremely careful with JSON formatting. Check all quotes, brackets and commas.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Carefully extract dictionary entries from this text and format as valid JSON:\n\n{chunk_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        try:
            data = json.loads(content)
            if 'entries' in data:
                return data['entries']
            else:
                logger.warning("Fallback extraction missing 'entries' key")
                return []
        except json.JSONDecodeError:
            logger.error("Fallback extraction also produced invalid JSON")
            # Last resort - create basic structure for the file
            return create_minimal_entries(chunk_text)
    
    except Exception as e:
        logger.error(f"Error in fallback extraction: {e}")
        return []

def create_minimal_entries(chunk_text: str) -> List[Dict[str, Any]]:
    """Create minimal entries using regex pattern matching as a last resort."""
    logger.info("Using regex pattern matching as last resort")
    
    # Basic pattern to identify potential dictionary entries
    # Looking for words at the beginning of lines that are likely to be headwords
    entry_pattern = r'(?m)^([a-zà-ù]+[^\n,.;:]*)[,.;:]'
    
    matches = re.finditer(entry_pattern, chunk_text)
    entries = []
    
    for match in matches:
        lemma = match.group(1).strip()
        # Find a reasonable amount of text following the lemma
        start_pos = match.start()
        end_pos = min(start_pos + 300, len(chunk_text))
        ocr_text = chunk_text[start_pos:end_pos]
        
        entries.append({
            "lemma": lemma,
            "ocr": ocr_text,
            "significati": ["Extracted via pattern matching - needs review"],
            "derivati": []
        })
    
    logger.info(f"Created {len(entries)} minimal entries via pattern matching")
    return entries

def merge_overlapping_entries(entries_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge entries from overlapping chunks, avoiding duplicates."""
    if not entries_lists:
        return []
    
    # Flatten the list of lists into a single list
    all_entries = []
    for entries in entries_lists:
        all_entries.extend(entries)
    
    # Group entries by lemma
    entries_by_lemma = {}
    for entry in all_entries:
        lemma = entry.get('lemma', '').lower()
        if not lemma:
            continue
            
        if lemma not in entries_by_lemma:
            entries_by_lemma[lemma] = []
        entries_by_lemma[lemma].append(entry)
    
    # For each lemma, select the most complete entry
    merged_entries = []
    for lemma, entries in entries_by_lemma.items():
        if len(entries) == 1:
            merged_entries.append(entries[0])
        else:
            # Select entry with most complete information
            best_entry = max(entries, key=lambda e: (
                len(e.get('ocr', '')), 
                len(e.get('significati', [])), 
                len(e.get('derivati', []))
            ))
            merged_entries.append(best_entry)
    
    return merged_entries

def validate_and_fix_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix an entry."""
    # Skip entries without a lemma
    if 'lemma' not in entry or not entry['lemma']:
        return None
    
    # Ensure all required fields exist
    if 'ocr' not in entry:
        entry['ocr'] = entry['lemma']
    
    if 'significati' not in entry:
        entry['significati'] = []
    
    if 'derivati' not in entry:
        entry['derivati'] = []
    
    # Fix lemma if it contains excessive information
    lemma = entry['lemma']
    if len(lemma) > 100:  # Likely contains extra text
        # Try to extract just the headword and basic grammatical info
        match = re.match(r'^([^,.;:]+(?:[,.] [a-z]+\.)?)', lemma)
        if match:
            entry['lemma'] = match.group(1).strip()
    
    # Fix significati format
    if isinstance(entry['significati'], str):
        entry['significati'] = [entry['significati']]
    
    # Fix derivati format and structure
    fixed_derivati = []
    for derivato in entry['derivati']:
        if isinstance(derivato, str):
            # Convert string to proper derivato structure
            fixed_derivati.append({
                'modificatore': derivato,
                'significati': []
            })
        elif isinstance(derivato, dict):
            # Ensure derivato has required fields
            if 'modificatore' not in derivato:
                derivato['modificatore'] = ""
            if 'significati' not in derivato:
                derivato['significati'] = []
            elif isinstance(derivato['significati'], str):
                derivato['significati'] = [derivato['significati']]
            
            fixed_derivati.append(derivato)
    
    entry['derivati'] = fixed_derivati
    
    return entry

def postprocess_entries(entries, filename):
    """
    Apply general postprocessing to entries to improve quality and fix common issues.
    Works for all dictionary files regardless of content.
    """
    processed_entries = []
    
    for entry in entries:
        # Skip empty entries
        if not entry.get("lemma"):
            continue
            
        # 2. Fix empty significati by extracting from OCR text
        if not entry.get("significati") or entry["significati"] == [""]:
            if "ocr" in entry:
                significati = []
                for part in entry["ocr"].split("|")[1:]:
                    if part.strip():
                        significati.append(part.strip())
                
                if significati:
                    entry["significati"] = significati
        
        # 3. Number significati if they don't have numbers
        fixed_significati = []
        for i, sig in enumerate(entry.get("significati", [])):
            if sig and not re.match(r'^\d+\.', sig):
                fixed_significati.append(f"{i+1}. {sig}")
            else:
                fixed_significati.append(sig)
        
        if fixed_significati:
            entry["significati"] = fixed_significati
        
        # 4. Fix common OCR errors in lemmas
        lemma = entry.get("lemma", "")
        # Remove unwanted characters that appear in OCR
        lemma = re.sub(r'[*#@&]', '', lemma)
        entry["lemma"] = lemma
        
        processed_entries.append(entry)
    
    logger.info(f"Postprocessed {len(processed_entries)} entries for {filename}")
    return processed_entries

def extract_entries_from_file(ocr_filepath: str) -> List[Dict[str, Any]]:
    """Extract dictionary entries from an OCR file."""
    logger.info(f"Processing {ocr_filepath}")
    
    try:
        with open(ocr_filepath, 'r', encoding='utf-8') as f:
            ocr_text = f.read()
        
        # Check if file is too large for single API call
        token_count = count_tokens(ocr_text)
        logger.info(f"File contains approximately {token_count} tokens")
        
        if token_count > 3000:
            # Create overlapping chunks to handle large files
            chunks = create_overlapping_chunks(ocr_text)
            logger.info(f"Split into {len(chunks)} chunks for processing")
            
            # Process each chunk and collect entries
            all_entries_lists = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                entries = extract_entries_from_chunk(chunk)
                all_entries_lists.append(entries)
                logger.info(f"Extracted {len(entries)} entries from chunk {i+1}")
            
            # Merge entries from overlapping chunks
            entries = merge_overlapping_entries(all_entries_lists)
            logger.info(f"Merged to {len(entries)} unique entries")
        else:
            # Process the entire file as a single chunk
            entries = extract_entries_from_chunk(ocr_text)
            logger.info(f"Extracted {len(entries)} entries")
        
        # Validate and fix entries
        validated_entries = []
        for entry in entries:
            fixed_entry = validate_and_fix_entry(entry)
            if fixed_entry:
                validated_entries.append(fixed_entry)
        
        # Fix cross-page issues
        final_entries = fix_cross_page_issues(validated_entries)
        
        # Apply postprocessing
        final_entries = postprocess_entries(final_entries, ocr_filepath)
        
        logger.info(f"Final count: {len(final_entries)} valid entries")
        return final_entries
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return []

def process_all_ocr_files(ocr_dir: str, output_dir: str, force_reprocess: bool = False):
    """Process all OCR files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all OCR files
    ocr_files = [f for f in os.listdir(ocr_dir) if f.endswith('_for_ai.txt')]
    logger.info(f"Found {len(ocr_files)} OCR files to process")
    
    for ocr_file in ocr_files:
        ocr_filepath = os.path.join(ocr_dir, ocr_file)
        
        # Determine output filename
        base_filename = ocr_file.replace('_for_ai.txt', '')
        output_filename = f"{base_filename}_output.json"
        output_filepath = os.path.join(output_dir, output_filename)
        
        # Skip if output file already exists (unless force_reprocess is True)
        if os.path.exists(output_filepath) and not force_reprocess:
            logger.info(f"Output file already exists: {output_filepath}, skipping")
            continue
        
        # Extract entries
        entries = extract_entries_from_file(ocr_filepath)
        
        # Save to JSON file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(entries)} entries to {output_filepath}")

def main():
    """Main function to process OCR files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract dictionary entries from OCR files')
    parser.add_argument('--ocr-dir', default='S_ocr_result', help='Directory containing OCR files')
    parser.add_argument('--output-dir', default='S_dictionary_entries', help='Directory to save output JSON files')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of already processed files')
    parser.add_argument('--file', help='Process a specific file only')
    
    args = parser.parse_args()
    
    if args.file:
        # Process a single file
        ocr_filepath = os.path.join(args.ocr_dir, args.file)
        if not os.path.exists(ocr_filepath):
            logger.error(f"File not found: {ocr_filepath}")
            return
        
        base_filename = args.file.replace('_for_ai.txt', '')
        output_filename = f"{base_filename}_output.json"
        output_filepath = os.path.join(args.output_dir, output_filename)

        # Add this check before processing each file
        if os.path.exists(output_filepath) and not args.force:
            logger.info(f"Skipping {args.file}, output already exists")
            return
        
        entries = extract_entries_from_file(ocr_filepath)
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(entries)} entries to {output_filepath}")
    else:
        # Process all files
        process_all_ocr_files(args.ocr_dir, args.output_dir, args.force)

def merge_letter_files(dictionary_dir, letter, output_path):
    """
    Merge all files for a given letter into a single JSON file and 
    sort the entries alphabetically by lemma.
    
    Args:
        dictionary_dir: Base directory containing letter-specific folders
        letter: Letter to merge (e.g., 'A', 'B', etc.)
        output_path: Path to save the merged JSON file
    """
    import os
    import json
    import re
    import unicodedata
    
    # Use the letter-specific directory
    letter_dir = f"{letter}_dictionary_entries"
    
    # Check if the letter directory exists
    if not os.path.exists(letter_dir):
        print(f"Directory not found: {letter_dir}")
        return
    
    # Get all files containing '_output.json' in the letter directory
    letter_files = [f for f in os.listdir(letter_dir) if '_output.json' in f]
    
    # Sort files by page number
    def get_file_number(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return 9999
    
    letter_files.sort(key=get_file_number)
    
    print(f"Merging {len(letter_files)} files for letter {letter}")
    
    # Collect all entries
    merged_entries = []
    for file in letter_files:
        file_path = os.path.join(letter_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'entries' in data:
                    # IMPORTANT: Filter entries to only include those starting with the correct letter
                    filtered_entries = []
                    for entry in data['entries']:
                        if 'lemma' in entry and entry['lemma']:
                            # Clean the lemma for comparison
                            clean_lemma = entry['lemma'].lower().strip()
                            # Remove any non-letter characters from the start
                            clean_lemma = re.sub(r'^[^a-zàèéìòóù]+', '', clean_lemma)
                            
                            # Check if the entry starts with the correct letter
                            if clean_lemma and clean_lemma[0].lower() == letter.lower():
                                filtered_entries.append(entry)
                    
                    merged_entries.extend(filtered_entries)
                    print(f"  Added {len(filtered_entries)} entries from {file} (filtered from {len(data['entries'])})")
        except Exception as e:
            print(f"  Error processing {file}: {e}")
    
    # IMPROVED ALPHABETICAL SORTING FUNCTION FOR ITALIAN
    def italian_sort_key(entry):
        if 'lemma' not in entry:
            return ""
            
        # Get the original lemma
        original_lemma = entry['lemma']
        
        # Create a clean version for sorting
        lemma = original_lemma.lower()
        
        # Remove ordinal indicators and numbers
        lemma = re.sub(r'\d+º\s*', '', lemma)
        
        # Remove any leading symbols for sorting purposes
        lemma = re.sub(r'^[^a-zàèéìòóù]+', '', lemma)
        
        # Split at commas and take first part as main lemma
        if ',' in lemma:
            lemma = lemma.split(',')[0].strip()
        
        # Handle Italian accents for proper sorting
        normalized = unicodedata.normalize('NFKD', lemma)
        
        # Remove any remaining non-alphabetic characters
        normalized = re.sub(r'[^a-z]', '', normalized)
        
        return normalized
    
    # Sort all entries alphabetically
    print(f"Sorting {len(merged_entries)} entries alphabetically...")
    merged_entries.sort(key=italian_sort_key)
    
    # Save the sorted entries
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"entries": merged_entries}, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuccessfully saved {len(merged_entries)} entries to {output_path}")
    
    # Print first and last few entries for verification
    if merged_entries:
        print("\nFirst 5 entries after sorting:")
        for i, entry in enumerate(merged_entries[:5]):
            print(f"{i+1}. {entry['lemma']}")
        
        print("\nLast 5 entries after sorting:")
        for i, entry in enumerate(merged_entries[-5:]):
            print(f"{len(merged_entries)-4+i}. {entry['lemma']}")

def read_ocr_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_entries(text):
    entries = []
    
    # Split the text into lines and clean up
    lines = text.split('\n')
    
    # Skip the first few lines which typically contain page numbers and headers
    start_idx = 0
    for i, line in enumerate(lines):
        if re.match(r'^\d+\|?\s*$', line.strip()):  # Skip line numbers and empty lines
            start_idx = i + 1
    
    # Combine lines to reconstruct the full text, removing line numbers
    full_text = ' '.join([re.sub(r'^\d+\|?\s*', '', line.strip()) for line in lines[start_idx:] if line.strip()])
    
    # Split the full text by dictionary entries
    # Look for patterns that indicate a new lemma/entry
    entry_pattern = r'([a-zàèéìòóù]+(?:\s*,\s*[+]?[a-zàèéìòóù-]+)?)\s*,\s*(?:[a-z]+\.\s*|[^,.]+\.\s*)'
    
    # Find all potential entry starts
    matches = list(re.finditer(entry_pattern, full_text))
    
    if not matches:
        return entries
    
    # Process each entry
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i < len(matches) - 1 else len(full_text)
        
        entry_text = full_text[start:end].strip()
        if not entry_text:
            continue
            
        # Extract the lemma
        lemma_match = re.match(r'([a-zàèéìòóù]+(?:\s*,\s*[+]?[a-zàèéìòóù-]+)?)', entry_text)
        if not lemma_match:
            continue
            
        lemma = lemma_match.group(1).strip()
        if '+' in lemma:
            lemma = lemma.split(',')[0].strip()  # Take only the main form if there are variants
        
        # Clean up the entry text
        entry_text = re.sub(r'^\s*' + re.escape(lemma_match.group(1)) + r'\s*,\s*', '', entry_text)
        
        # Split the entry into significati and derivati
        significati = []
        derivati = []
        
        # First, identify and extract significati
        significati_text = entry_text
        derivati_text = ""
        
        # Look for derivati markers like "-amento", "-ato", etc.
        derivati_markers = re.finditer(r'\|\s*([-+][\w]+,\s*(?:m|f|ag|av|pl|pt|cng|dm|acc|sup|peg|spr)\b\.?)', entry_text)
        
        for marker in derivati_markers:
            marker_pos = marker.start()
            # If we find a derivati marker, split the text
            if marker_pos > 0:
                significati_text = entry_text[:marker_pos]
                derivati_text = entry_text[marker_pos:]
                break
        
        # Extract significati
        significati_parts = significati_text.split('|')
        for i, part in enumerate(significati_parts):
            part = part.strip()
            if part and (i > 0 or 'Render' in part or part[0].isupper()):  # Skip lemma definition in first part
                significati.append(f"{len(significati) + 1}. {part}")
        
        # Extract derivati
        if derivati_text:
            derivati_parts = re.finditer(r'\|\s*([-+][\w]+,\s*(?:m|f|ag|av|pl|pt|cng|dm|acc|sup|peg|spr)\b\.?[^|]*)', derivati_text)
            
            for derivati_part in derivati_parts:
                derivati_part_text = derivati_part.group(1).strip()
                
                # Split by the first period or comma to get the modificatore
                mod_match = re.match(r'([-+][\w]+,\s*(?:m|f|ag|av|pl|pt|cng|dm|acc|sup|peg|spr)\b\.?)', derivati_part_text)
                if not mod_match:
                    continue
                    
                modificatore = mod_match.group(1).strip()
                
                # Get the significati for this derivati
                derivati_significati_text = derivati_part_text[len(modificatore):].strip()
                derivati_significati = []
                
                if derivati_significati_text:
                    derivati_significati_parts = derivati_significati_text.split('|')
                    for part in derivati_significati_parts:
                        part = part.strip()
                        if part:
                            derivati_significati.append(part)
                
                derivati.append({
                    "modificatore": modificatore,
                    "significati": derivati_significati
                })
        
        entries.append({
            "lemma": lemma,
            "ocr": entry_text,
            "significati": significati,
            "derivati": derivati
        })
    
    return entries

def fix_cross_page_issues(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fix issues with entries at page boundaries.
    - Remove incomplete first entries that are just continuation markers
    - Fix duplicate entries that might appear across page boundaries
    """
    if not entries:
        return entries
    
    # Check for empty or incomplete first entry
    if entries and (not entries[0].get('significati') or 
                    entries[0].get('ocr', '').startswith('###PAGE_CONTINUATION###')):
        # The first entry appears to be a continuation from previous page or incomplete
        # Remove it to avoid duplication
        fixed_entries = entries[1:]
        logger.info(f"Removed first entry '{entries[0].get('lemma', '')}' as it appears to be a page continuation")
        return fixed_entries
    
    # Check for duplicate entries
    unique_entries = []
    seen_lemmas = set()
    
    for entry in entries:
        lemma = entry.get('lemma', '').lower()
        if not lemma:
            continue
            
        if lemma in seen_lemmas:
            # This is a duplicate - merge with the existing entry
            for existing in unique_entries:
                if existing.get('lemma', '').lower() == lemma:
                    # Merge meanings and derivatives if needed
                    existing_sigs = existing.get('significati', [])
                    entry_sigs = entry.get('significati', [])
                    
                    # Add meanings that aren't already present
                    for sig in entry_sigs:
                        if sig not in existing_sigs:
                            existing_sigs.append(sig)
                    
                    # Update significati
                    existing['significati'] = existing_sigs
                    
                    # Merge derivatives
                    existing_derivs = existing.get('derivati', [])
                    entry_derivs = entry.get('derivati', [])
                    
                    # This is simplified - a more complex merge might be needed
                    for deriv in entry_derivs:
                        if deriv not in existing_derivs:
                            existing_derivs.append(deriv)
                    
                    # Update derivati
                    existing['derivati'] = existing_derivs
                    
                    logger.info(f"Merged duplicate entry for '{lemma}'")
                    break
        else:
            seen_lemmas.add(lemma)
            unique_entries.append(entry)
    
    if len(unique_entries) < len(entries):
        logger.info(f"Removed {len(entries) - len(unique_entries)} duplicate entries")
    
    return unique_entries

if __name__ == "__main__":
    main()