import os
import string

def create_alphabet_folders(start_letter='C', base_dir='.'):
    """
    Create folders named A_ocr_result, B_ocr_result, ..., Z_ocr_result starting from a specified letter.
    
    Args:
        start_letter: The letter to start creating folders from (default is 'C').
        base_dir: The base directory where folders will be created (default is current directory).
    """
    # Get the index of the start letter in the alphabet
    start_index = string.ascii_uppercase.index(start_letter.upper())
    
    # Create folders from the start letter to 'Z'
    for letter in string.ascii_uppercase[start_index:]:
        folder_name = f"{letter}_ocr_result"
        folder_path = os.path.join(base_dir, folder_name)
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_name}")
        else:
            print(f"Folder already exists: {folder_name}")

# Example usage
create_alphabet_folders(start_letter='C')