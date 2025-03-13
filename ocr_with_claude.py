import os
import json
from typing import List, Dict
import anthropic
from dotenv import load_dotenv
from PIL import Image
import base64
import re

load_dotenv('.env')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
class DictionaryExtractor:
    def __init__(self):
        self.client = client
        self.prompt = """
        CRITICAL: You are extracting entries from an Italian dictionary. Follow these instructions EXACTLY:

        VISUAL MARKERS TO IDENTIFY ENTRIES:
        1. Look for these visual cues that indicate NEW entries:
            • Larger or bold text at start of entry
            • Indentation changes
            • Paragraph breaks
            • Special symbols like +, ✩, ▽, ♥, ☞
            • Entry numbers or bullets
            • Different font styles
            
        2. Entry boundaries are indicated by:
            • Space between entries
            • Change in text size/style
            • New line with bold/large text
            • Punctuation marks like ‖, ║, |
            • Cross-reference markers (v., vedi, cf.)

        EXTRACTION RULES:
        1. Each VISUALLY DISTINCT headword starts a NEW entry
        2. Include ALL text until you see the next headword markers
        3. Keep entries COMPLETELY SEPARATE
        4. Maintain the EXACT order as on the page
        5. Preserve ALL special characters and formatting
        6. Include ALL variant forms in the lemma
        7. Capture ALL meanings and derivatives

        FORMAT:
        {
            "lemma": "EXACT headword with ALL variants and grammatical notes (e.g., 'abbagliare, v.tr.' or '+àbada, m.')",
            "ocr": "COMPLETE text from start of entry until next entry",
            "significati": [
                "each distinct meaning",
                "each usage or definition"
            ],
            "C": [
                {
                    "modificatore": "exact form (-ato, -mente, etc.)",
                    "significati": ["meanings for this form"]
                }
            ]
        }

        EXAMPLES OF PROPER ENTRY SEPARATION:
        • When you see: "abbagliare, v.tr. [definition] || abbaglio, m. [definition]"
           Create TWO entries: "abbagliare" and "abbaglio"
        
        • When you see: "+àbada, m. [definition] ‖ àbato, m. [definition]"
           Create TWO entries: "+àbada" and "àbato"

        VERIFICATION:
        1. Re-scan the page TOP to BOTTOM
        2. Check that each visually distinct headword is a separate entry
        3. Verify entry boundaries are correct
        4. Confirm no text is missed
        5. Validate all special characters are preserved

        CRITICAL: NEVER combine multiple headwords into one entry. Each visually distinct headword MUST be a separate entry.
        """

    def extract_entries(self, image_path: str) -> List[Dict]:
        """Extract dictionary entries using Claude vision capabilities"""
        print(f"Starting extraction for {image_path}")
        
        try:
            # Load and prepare image
            print("Reading image file...")
            with open(image_path, "rb") as f:
                image_data = f.read()
            print(f"Read {len(image_data)} bytes")
            
            # Convert image data to base64
            print("Converting to base64...")
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            print("Base64 conversion complete")
            
            try:
                # Get Claude's analysis
                print("Making API call to Claude...")
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4096,
                    system="You are a dictionary entry extractor. Extract entries precisely in the requested JSON format.",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_base64
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": self.prompt
                                }
                            ]
                        }
                    ]
                )
                
                print("Got response from Claude")
                print("Raw response:", response.content)
                
                # Parse Claude's response into structured data
                entries = self._parse_response(response.content)
                return entries
                
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return []
            
        except Exception as e:
            print(f"Error reading/processing image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
        
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse Claude's response into structured entries"""
        try:
            # Handle TextBlock response
            if isinstance(response, list):
                # If response is a list of TextBlocks, get the text from the first one
                text = response[0].text if response else ""
            elif hasattr(response, 'text'):
                text = response.text
            else:
                text = response
            
            # Remove markdown code block markers if present
            text = text.replace('```json', '').replace('```', '').strip()
            
            # Try to parse as JSON array first
            try:
                if text.strip().startswith('[') and text.strip().endswith(']'):
                    entries = json.loads(text)
                    valid_entries = [entry for entry in entries if self._is_valid_entry(entry)]
                    print(f"Successfully parsed {len(valid_entries)} entries from JSON array")
                    return valid_entries
            except json.JSONDecodeError as e:
                print(f"Failed to parse as JSON array: {str(e)}")
            
            # If that fails, try to find individual JSON objects
            entries = []
            depth = 0
            start = -1
            
            for i, char in enumerate(text):
                if char == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start != -1:
                        try:
                            json_str = text[start:i+1]
                            entry = json.loads(json_str)
                            if self._is_valid_entry(entry):
                                entries.append(entry)
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON object: {json_str[:100]}...")
                        start = -1
            
            print(f"Found {len(entries)} valid entries")
            return entries
        
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
        
    def _is_valid_entry(self, entry: Dict) -> bool:
        """Validate entry structure and content"""
        try:
            # Check required fields exist
            if not all(field in entry for field in ['lemma', 'ocr', 'significati']):
                print(f"Missing required fields in entry: {entry.get('lemma', 'unknown')}")
                return False
            
            # Check fields have valid content
            if not entry['lemma'] or not isinstance(entry['lemma'], str):
                print(f"Invalid lemma in entry: {entry}")
                return False
            
            if not entry['ocr'] or not isinstance(entry['ocr'], str):
                print(f"Invalid ocr in entry: {entry.get('lemma', 'unknown')}")
                return False
            
            if not isinstance(entry['significati'], list):
                print(f"Invalid significati in entry: {entry.get('lemma', 'unknown')}")
                return False
            
            # Check for mixed entries (entries containing multiple headwords)
            ocr_lower = entry['ocr'].lower()
            lemma_lower = entry['lemma'].lower()
            
            # Common entry separators in Italian dictionaries
            separators = [' v. ', ' vedi ', ' cf. ', ' cfr. ', '║', '‖', '\n\n']
            for sep in separators:
                if sep in ocr_lower and not sep in lemma_lower:
                    parts = ocr_lower.split(sep)
                    if len(parts) > 1 and not all(p.strip().startswith(lemma_lower) for p in parts):
                        print(f"Possible mixed entries detected in: {entry.get('lemma', 'unknown')}")
                        return False
            
            # Validate derivati if present
            if 'derivati' in entry:
                if not isinstance(entry['derivati'], list):
                    print(f"Invalid derivati list in entry: {entry.get('lemma', 'unknown')}")
                    return False
                for deriv in entry['derivati']:
                    if not isinstance(deriv, dict):
                        print(f"Invalid derivative in entry: {entry.get('lemma', 'unknown')}")
                        return False
                    if 'modificatore' not in deriv or 'significati' not in deriv:
                        print(f"Missing derivative fields in entry: {entry.get('lemma', 'unknown')}")
                        return False
                    if not isinstance(deriv['significati'], list):
                        print(f"Invalid derivative significati in entry: {entry.get('lemma', 'unknown')}")
                        return False
            
            return True
        
        except Exception as e:
            print(f"Error validating entry: {str(e)}")
            return False

def process_dictionary(input_dir: str, output_dir: str):
    """Process all dictionary pages"""
    print("\nInitializing DictionaryExtractor...")
    extractor = DictionaryExtractor()
    
    # Process each page
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            image_path = os.path.join(input_dir, filename)
            print(f"\nProcessing {filename}")
            print(f"Full path: {image_path}")
            
            try:
                print("Calling extract_entries...")
                entries = extractor.extract_entries(image_path)
                print(f"Got {len(entries)} entries back")
                
                # Write entries to appropriate files based on first letter
                for entry in entries:
                    first_letter = entry['lemma'][0].upper()
                    output_file = os.path.join(output_dir, f"{first_letter}.jsonl")
                    print(f"Writing to {output_file}")
                    
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write('\n')
                        
                print(f"Processed {len(entries)} entries")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                import traceback
                print(traceback.format_exc())

def main():
    input_dir = "italian_Images_test"
    output_dir = "output_json"
    
    print(f"Starting processing...")
    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found")
        return
        
    # Check for image files
    image_files = [f for f in sorted(os.listdir(input_dir)) 
                  if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    
    if not image_files:
        print(f"Error: No image files found in {input_dir}")
        return
        
    print(f"Found {len(image_files)} image files: {image_files}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check API key
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        return
    
    try:
        process_dictionary(input_dir, output_dir)
    except Exception as e:
        print(f"Error in process_dictionary: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 