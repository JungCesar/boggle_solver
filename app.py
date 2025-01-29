from flask import Flask, request, render_template, jsonify
from solver import find_words, load_word_list

# import cv2
# import numpy as np
# import pytesseract
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


def process_boggle_image(image_data):
    try:
        # Convert image data to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Initialize OpenAI client with API key from environment variable
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Prepare the API request with specific instructions
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a 4x4 Boggle grid. Read the letters from left to right, top to bottom, like reading a book. Return ONLY the 16 letters in a single string, no spaces or separators. The correct format should be exactly 16 uppercase letters.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=50,
        )

        # Prepare the API request with optimized instructions
        response = client.chat.completions.create(
            model="gpt-4o-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image of a 4x4 Boggle grid and return exactly 16 uppercase letters "
                                "in a single string with no spaces, punctuation, or separators. "
                                "The grid must be read from left to right, top to bottom, like reading a book. "
                                "Ensure proper alignment by identifying the text orientation—if the grid appears rotated, "
                                "correct its orientation before extracting the letters. "
                                "Return ONLY the 16 letters in the correct order. Do not include any additional text, "
                                "explanations, or formatting."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=50,
        )

        # Extract and clean the response
        letters = "".join(
            c for c in response.choices[0].message.content if c.isalpha()
        ).upper()

        # Debug information
        print(f"GPT-4 Vision detected letters: {letters}")
        print(f"Number of letters detected: {len(letters)}")

        if len(letters) != 16:
            raise ValueError(f"Detected {len(letters)} letters instead of 16")

        return letters

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
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
