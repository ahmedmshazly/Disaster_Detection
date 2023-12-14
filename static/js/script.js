document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById('predictionForm');
    const resultDiv = document.getElementById('result');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const tweetsTable = document.getElementById('tweetsTable');

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        loadingIndicator.style.display = 'block';
        tweetsTable.innerHTML = ''; // Clear previous table rows

        const numTweets = document.getElementById('numTweets').value;

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/predict', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onload = function () {
            loadingIndicator.style.display = 'none';

            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);


                let disasterLevel = '';
                let indicatorClass = '';
                
                console.log(response)

                if (response.disaster_average < 0.3) {
                    disasterLevel = 'Green - All is Good';
                    indicatorClass = 'green';
                } else if (response.disaster_average < 0.6) {
                    disasterLevel = 'Yellow - Be Cautious';
                    indicatorClass = 'yellow';
                } else {
                    disasterLevel = 'Red - Disaster and Danger';
                    indicatorClass = 'red';
                }

                resultDiv.innerHTML = `
            <div class="disaster-indicator ${indicatorClass}"></div>
            <div class="result-text">${disasterLevel} (${(response.disaster_average * 100).toFixed(2)}% Probability)</div>
        `;
                // Create table headers
                let headerRow = tweetsTable.insertRow();
                headerRow.innerHTML = `<th>Tweet Text</th><th>Prediction</th>`;

                // Display each tweet and its prediction in a table row
                response.tweet_predictions.forEach(tweet => {
                    let row = tweetsTable.insertRow();
                    let textCell = row.insertCell(0);
                    let predictionCell = row.insertCell(1);
                    textCell.innerText = tweet.text;
                    predictionCell.innerText = tweet.prediction;
                });
            } else {
                resultDiv.innerHTML = `<p>Error occurred: ${xhr.statusText}</p>`;
            }
        };

        xhr.send(JSON.stringify({ num_tweets: numTweets }));
    });
});
