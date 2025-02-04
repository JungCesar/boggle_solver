import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from openai import OpenAI
import base64
from dotenv import load_dotenv
from google.cloud import vision

# Load environment variables
load_dotenv()

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Save preprocessed image for debugging
    cv2.imwrite('debug_preprocessed.jpg', thresh)
    
    return thresh, image.shape[1], image.shape[0]

def detect_grid(thresh, width, height):
    # Find the blue frame (assuming it's the largest contour)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(grid_contour)
    
    # Create a grid of 16 cells
    cell_w = w // 4
    cell_h = h // 4
    
    cells = []
    for row in range(4):
        for col in range(4):
            cell_x = x + (col * cell_w)
            cell_y = y + (row * cell_h)
            cells.append((cell_x, cell_y, cell_w, cell_h))
    
    return cells

def recognize_letter(image, cell):
    x, y, w, h = cell
    
    # Extract the cell
    cell_image = image[y:y+h, x:x+w]
    
    # Add padding around the cell
    padding = 2
    cell_image = cv2.copyMakeBorder(
        cell_image, 
        padding, padding, padding, padding, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )
    
    # Resize for better OCR
    cell_image = cv2.resize(cell_image, (32, 32))
    
    # Save individual cells for debugging
    cv2.imwrite(f'debug_cell_{x}_{y}.jpg', cell_image)
    
    # Configure Tesseract for single letter recognition
    config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter = pytesseract.image_to_string(cell_image, config=config).strip()
    
    return letter if letter else '?'

def main(image_path):
    # Preprocess the image
    thresh, width, height = preprocess_image(image_path)
    
    # Detect the grid and get cell coordinates
    cells = detect_grid(thresh, width, height)
    
    # Recognize letters in each cell
    letters = []
    for cell in cells:
        letter = recognize_letter(thresh, cell)
        letters.append(letter)
    
    # Combine letters into a string
    result = ''.join(letters)
    
    # Print results
    print(f"\nDetected letters: {result}")
    print("\nGrid visualization:")
    for i in range(0, 16, 4):
        print(result[i:i+4])
    
    return result

def test_all_examples():
    # Get all image files from grid_examples folder
    image_files = [f for f in os.listdir('grid_examples') 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join('grid_examples', image_file)
        print(f"\nProcessing {image_file}:")
        print("-" * 40)
        
        try:
            # Create a debug folder for this image
            debug_folder = f'debug_{os.path.splitext(image_file)[0]}'
            os.makedirs(debug_folder, exist_ok=True)
            
            # Process the image
            result = main(image_path)
            
            # Print results in a clear format
            print(f"\nResults for {image_file}:")
            print("Detected string:", result)
            print("\nGrid visualization:")
            for i in range(0, 16, 4):
                print(result[i:i+4])
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            print("-" * 40)

def get_gpt4_vision_result(image_base64):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """This is a 4x4 Boggle grid with a blue frame. 
                        1. The blue frame indicates the correct orientation.
                        2. Read ONLY the 16 letters from left to right, top to bottom.
                        3. Return just the 16 uppercase letters with no spaces or explanations.
                        4. Double-check your answer before responding."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=50
    )
    return ''.join(c for c in response.choices[0].message.content if c.isalpha()).upper()

def get_google_vision_result(image_data):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)
    response = client.text_detection(image=image)
    
    if response.text_annotations:
        # Get all detected text
        text = response.text_annotations[0].description
        # Clean up and get only letters
        letters = ''.join(c for c in text if c.isalpha()).upper()
        return letters[:16]
    return None

def test_image_recognition(image_path):
    try:
        # Read the image file
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            
        # Convert to base64 for GPT-4 Vision
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Get results from both models
        print("Getting results from multiple models...")
        
        gpt4_result = get_gpt4_vision_result(image_base64)
        print(f"GPT-4 Vision result: {gpt4_result}")
        
        google_result = get_google_vision_result(image_data)
        print(f"Google Vision result: {google_result}")
        
        # Compare results and handle discrepancies
        if gpt4_result == google_result and len(gpt4_result) == 16:
            final_result = gpt4_result
            confidence = "High (both models agree)"
        else:
            # If models disagree, prefer GPT-4 Vision result
            final_result = gpt4_result
            confidence = "Medium (models disagree)"
        
        print("\nFinal Results:")
        print(f"Letters: {final_result}")
        print(f"Confidence: {confidence}")
        
        # Display grid visualization
        print("\nGrid Visualization:")
        for i in range(0, 16, 4):
            print(final_result[i:i+4])
            
        return final_result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Set Tesseract path if needed
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
    
    # Test all examples
    test_all_examples()

    # Test with your image
    result = test_image_recognition('test_grid.jpg') 