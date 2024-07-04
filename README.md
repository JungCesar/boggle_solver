# Boggle Solver

This project is a Boggle solver implemented as a web application using Flask. The application allows users to input a string, representing the 4x4 Boggle grid and find all valid words from the grid based on a predefined word list (in Dutch).

## Features

- Input a Boggle grid through a web form.
- Find all possible words in the grid.
- Display the found words sorted alphabetically.
- Show the total count of found words.
- Measure and display the execution time of the search algorithm (Depft-First Search).
- Identify and display the longest word found.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Flask

### Installation

1. Clone the repository:

    >git clone <https://github.com/your-username/boggle_solver.git>
    >cd boggle_solver

2. Set up a virtual environment and activate it:

    >python -m venv venv
    >source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:

    >pip install Flask

4. A Dutch word list is provided. To use a different language, replace `words.txt` with your own word list.

### Running the Application

1. Start the Flask application:

    >python app.py

2. Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to use the Boggle solver.

## Project Structure

```python
boggle_solver/
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html
├── app.py
├── solver.py
├── words.txt
└── README.md
```

## Search Algorithm

The core of the Boggle solver is implemented in the `find_words` function in `solver.py`. Here's a detailed explanation of how it works:

### 1. Load Word List

The `load_word_list` function reads words from a file, converts them to uppercase, and filters out words shorter than 3 characters:

```python
def load_word_list(filepath: str) -> set:
    with open(filepath, "r", encoding="utf-8") as file:
        return set(word.strip().upper() for word in file if len(word.strip()) > 2)
```

### 2. Build Prefix Set

To optimize the search, a prefix set is created containing all prefixes of valid words:

```python
def build_prefix_set(word_list: set) -> set:
    prefix_set = set()
    for word in word_list:
        for i in range(1, len(word) + 1):
            prefix_set.add(word[:i])
    return prefix_set
```

This allows for early termination of searches that can't lead to valid words.

### 3. Depth-First Search (DFS)

The main search algorithm uses depth-first search to explore all possible paths in the grid:

```python
def find_words(grid: list, word_list: set) -> list:
    def dfs(x, y, word, visited):
        if len(word) > 2 and word in word_list:
            results.add(word)
        
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4 and (nx, ny) not in visited:
                new_word = word + grid[nx][ny]
                if new_word in prefix_set:
                    dfs(nx, ny, new_word, visited | {(nx, ny)})

    results = set()
    prefix_set = build_prefix_set(word_list)
    
    for i in range(4):
        for j in range(4):
            dfs(i, j, grid[i][j], {(i, j)})
    
    return sorted(results)
```

Key points of the algorithm:

- It starts from each cell in the grid.
- For each starting point, it explores all 3-8 neighboring cells.
- It keeps track of visited cells to avoid using the same letter twice in a word.
- It uses the prefix set to prune search paths that can't lead to valid words.
- When a valid word is found, it's added to the results set.

### 4. Optimization

The use of the prefix set significantly reduces the search space by terminating paths that can't form valid words early. This optimization is crucial for the algorithm's efficiency, especially for larger grids or word lists.

### Performance

The time complexity of the algorithm is O(N*M*8^L), where:

- N is the number of rows in the grid (4 in standard Boggle)
- M is the number of columns in the grid (4 in standard Boggle)
- L is the maximum word length in the dictionary

While this complexity is exponential, the prefix set optimization significantly reduces the actual runtime in practice.

### Future Improvements

- Implement a trie data structure for even more efficient prefix checking
- Add support for different grid sizes
- Implement a scoring system based on word length
- Create a RESTful API for the solver
