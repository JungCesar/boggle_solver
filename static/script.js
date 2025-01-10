$(document).ready(function () {
    function initializeClickHandlers() {
        // Highlight word when clicked
        $('.word-item').off('click').on('click', function () {
            const word = $(this).text();
            highlightWord(word);
        });
    }

    $('#sort-select').change(function () {
        const sortOption = $(this).val(); // Get selected sort option
        const wordItems = [];

        // Collect current words from the word list
        $('#word-list .word-item').each(function () {
            wordItems.push($(this).text());
        });

        // Send AJAX POST request to the server
        $.ajax({
            url: '/sort',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                words: wordItems,
                sort: sortOption
            }),
            success: function (response) {
                // Update the word list with sorted words
                const sortedWords = response.sorted_words;
                const wordList = $('#word-list');
                wordList.empty(); // Clear the current list
                sortedWords.forEach(word => {
                    wordList.append(`<li class="word-item">${word}</li>`);
                });

                // Reinitialize click handlers after updating the word list
                initializeClickHandlers();
            },
            error: function (error) {
                console.error("Error sorting words:", error);
            }
        });
    });

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

    // Initialize click handlers on page load
    initializeClickHandlers();

    // Highlight the longest word by default (optional)
    const longestWord = $('.word-item').first().text();
    if (longestWord) {
        highlightWord(longestWord);
    }
});
