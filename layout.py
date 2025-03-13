import layoutparser as lp
import cv2
import os
import numpy as np

try:
    print("Loading model...")
    # Use the correct model path without URL parameters
    model = lp.models.Detectron2LayoutModel(
        'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',  # Using official model path
        extra_config=["MODEL.DEVICE", "cpu"],             # Explicitly set to use CPU
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Figure"}
    )
    
    print("Model loaded successfully!")
    
    # Directory of images
    input_dir = "enhanced_output"
    
    # Output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through all images in the directory
    for filename in os.listdir(input_dir):
        # Construct the full path to the image
        image_path = os.path.join(input_dir, filename)
        
        # Read the image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping file '{filename}' because it is not a valid image.")
            continue
        
        # Convert BGR to RGB for LayoutParser
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"Processing image: {filename}")
        print("Detecting layout...")
        layout = model.detect(image_rgb)

        print("Drawing boxes...")
        viz = lp.draw_box(image_rgb, layout, box_width=3)

        # Convert to NumPy array for cv2
        viz_np = np.array(viz)
        viz_bgr = cv2.cvtColor(viz_np, cv2.COLOR_RGB2BGR)

        # Construct output path
        output_path = os.path.join(output_dir, f"layout_{filename}")
        
        # Save the result
        cv2.imwrite(output_path, viz_bgr)
        print(f"Output saved to {output_path}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Full error details:", e.__class__.__name__)
    import traceback
    traceback.print_exc()
