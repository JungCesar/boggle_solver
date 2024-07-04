$(document).ready(function () {
    function highlightWord(word) {
        // Reset all cells
        $('.boggle-cell').removeClass('highlighted');

        let grid = [];
        $('.boggle-row').each(function () {
            let row = [];
            $(this).find('.boggle-cell').each(function () {
                row.push($(this).text());
            });
            grid.push(row);
        });

        function findWord(x, y, remainingWord, path, used) {
            if (remainingWord.length === 0) {
                return path;
            }
            if (x < 0 || x >= 4 || y < 0 || y >= 4 || grid[x][y] !== remainingWord[0] || used[x][y]) {
                return null;
            }
            let newUsed = used.map(row => [...row]);
            newUsed[x][y] = true;
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    if (dx === 0 && dy === 0) continue;
                    let result = findWord(x + dx, y + dy, remainingWord.slice(1), [...path, [x, y]], newUsed);
                    if (result) return result;
                }
            }
            return null;
        }

        let path = null;
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let used = Array(4).fill().map(() => Array(4).fill(false));
                path = findWord(i, j, word, [], used);
                if (path) break;
            }
            if (path) break;
        }

        if (path) {
            path.forEach(([x, y]) => {
                $('.boggle-row').eq(x).find('.boggle-cell').eq(y).addClass('highlighted');
            });
        } else {
            console.log("Word not found in grid:", word);
        }
    }

    // Find the longest word
    let longestWord = '';
    $('.word[data-word]').each(function () {
        let word = $(this).data('word');
        if (word.length > longestWord.length) {
            longestWord = word;
        }
    });

    // Highlight longest word by default
    if (longestWord) {
        highlightWord(longestWord);
    }

    // Highlight word when clicked
    $('.word').click(function () {
        let word = $(this).data('word');
        highlightWord(word);
    });
});