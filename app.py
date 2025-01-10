from flask import Flask, request, render_template, jsonify
from solver import find_words, load_word_list
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

app = Flask(__name__)

# Constant for the expected grid size
GRID_SIZE = 16
ROW_LENGTH = 4

# Add these configurations after the existing ones
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_boggle_image(image_data):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Configure Tesseract parameters
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # Extract text
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    # Clean and format the text
    letters = ''.join(c for c in text.upper() if c.isalpha())
    return letters[:16]  # Return only first 16 letters

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


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read the image file
            image_data = file.read()
            letters = process_boggle_image(image_data)
            
            if len(letters) < 16:
                return jsonify({'error': 'Could not detect 16 letters in image'}), 400
                
            return jsonify({'letters': letters}), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
