import time


def load_word_list(filepath: str) -> set:
    with open(filepath, "r", encoding="utf-8") as file:
        return set(word.strip().upper() for word in file if len(word.strip()) > 2)


def build_prefix_set(word_list: set) -> set:
    prefix_set = set()
    for word in word_list:
        for i in range(1, len(word) + 1):
            prefix_set.add(word[:i])
    return prefix_set


def find_words(grid: list, word_list: set) -> dict:
    def dfs(x, y, word, visited):
        if len(word) > 2 and word in word_list:
            results.add(word)

        for dx, dy in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4 and (nx, ny) not in visited:
                new_word = word + grid[nx][ny]
                if new_word in prefix_set:
                    dfs(nx, ny, new_word, visited | {(nx, ny)})

    start_time = time.time()
    results = set()
    word_list = {word.upper() for word in word_list}
    prefix_set = build_prefix_set(word_list)

    for i in range(4):
        for j in range(4):
            dfs(i, j, grid[i][j], {(i, j)})

    end_time = time.time()
    execution_time = end_time - start_time

    sorted_results_alph = sorted(results)
    sorted_results_len = sorted(results, key=len, reverse=True)

    return {
        "words": sorted_results_alph,
        "count": len(results),
        "time": execution_time,
        "longest_word": sorted_results_len[0] if sorted_results_len else None,
    }
