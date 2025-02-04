from flask import Flask, request, render_template, jsonify
from solver import find_words, load_word_list

import cv2
import numpy as np

import pytesseract
# from PIL import Image
# import io
# from google.cloud import vision

import os
from openai import OpenAI
import base64
from dotenv import load_dotenv

app = Flask(__name__)

# Constant for the expected grid size
GRID_SIZE = 16
ROW_LENGTH = 4

# Add these configurations after the existing ones
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Load environment variables at the start of your app
load_dotenv()

# Test print (remove this after testing)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("API key loaded successfully!")
    print(f"First 5 characters of key: {api_key[:5]}...")
else:
    print("Warning: OPENAI_API_KEY not found in environment variables")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# def preprocess_image(image_data):
#     # Convert image data to a NumPy array
#     np_img = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#     )

#     # Apply morphological operations to remove noise and enhance the grid
#     kernel = np.ones((3, 3), np.uint8)
#     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     # Apply edge detection
#     edges = cv2.Canny(morph, 50, 150)

#     # Encode the processed image back to bytes
#     _, processed_img = cv2.imencode(".jpg", edges)
#     return processed_img.tobytes()


def process_boggle_image(image_data):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply binary thresholding
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        letter_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Adjust this threshold based on your image size
                x, y, w, h = cv2.boundingRect(contour)
                letter_contours.append((x, y, w, h, contour))
        
        # Sort contours by position (top to bottom, left to right)
        letter_contours.sort(key=lambda x: (x[1] // 50) * 4 + x[0] // 50)
        
        # Extract and recognize letters
        recognized_letters = ""
        custom_config = r'--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        for x, y, w, h, contour in letter_contours[:16]:  # Limit to 16 letters
            # Extract letter image
            letter_image = thresh[y:y+h, x:x+w]
            
            # Resize for better OCR
            letter_image = cv2.resize(letter_image, (28, 28))
            
            # Recognize letter
            letter = pytesseract.image_to_string(letter_image, config=custom_config).strip()
            if letter:
                recognized_letters += letter
        
        # Debug information
        print(f"Detected letters: {recognized_letters}")
        print(f"Number of letters detected: {len(recognized_letters)}")
        
        # Convert image to base64 for GPT-4 Vision
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Use GPT-4 Vision as a backup/verification
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a 4x4 Boggle grid. Return ONLY the 16 letters in reading order (left to right, top to bottom) as a single string of uppercase letters. No explanations or additional text."
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
        
        # Get GPT-4's verification/correction
        verified_letters = ''.join(c for c in response.choices[0].message.content if c.isalpha()).upper()
        
        print(f"GPT-4 verified letters: {verified_letters}")
        
        # Return the verified letters
        return verified_letters[:16]
        
    except Exception as e:
        print(f"Error in image processing: {str(e)}")
        raise


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        grid_input = request.form["grid"].upper().replace(",", "").strip()
        if len(grid_input) != GRID_SIZE:
            return render_template(
                "index.html", error=f"Please enter exactly {GRID_SIZE} letters."
            )
        grid = [
            list(grid_input[i : i + ROW_LENGTH])
            for i in range(0, GRID_SIZE, ROW_LENGTH)
        ]
        try:
            wordlist = load_word_list("words.txt")
            result = find_words(grid, wordlist)
            return render_template(
                "index.html",
                words=result["words"],
                count=result["count"],
                time=result["time"],
                longest_word=result["longest_word"],
                grid=grid,
            )
        except Exception as e:
            return render_template("index.html", error=str(e))
    return render_template("index.html")


@app.route("/sort", methods=["POST"])
def sort():
    try:
        words = request.json.get("words", [])
        sort_option = request.json.get("sort", "alphabetical")

        # Apply sorting based on the chosen option
        if sort_option == "points":
            sorted_words = sorted(
                words, key=lambda w: len(w), reverse=True
            )  # Example: sort by length
        elif sort_option == "alphabetical":
            sorted_words = sorted(words)
        else:
            sorted_words = words

        return {"sorted_words": sorted_words}, 200
    except Exception as e:
        return {"error": str(e)}, 400


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Read the image file
            image_data = file.read()
            letters = process_boggle_image(image_data)

            if len(letters) < 16:
                return jsonify({"error": "Could not detect 16 letters in image"}), 400

            return jsonify({"letters": letters}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
