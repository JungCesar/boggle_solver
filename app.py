from flask import Flask, request, render_template
from solver import find_words, load_word_list

app = Flask(__name__)

# Constant for the expected grid size
GRID_SIZE = 16
ROW_LENGTH = 4


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


if __name__ == "__main__":
    app.run(debug=True)
