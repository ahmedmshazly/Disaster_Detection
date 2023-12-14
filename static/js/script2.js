document.getElementById('singlePredictionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const tweetText = document.getElementById('tweetText').value;
    const resultDiv = document.getElementById('singlePredictionResult');

    fetch('/predict_single', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tweet_text: tweetText }),
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.innerText = 'Prediction: ' + data.prediction;
    })
    .catch(error => {
        console.error('Error:', error);
        resultDiv.innerText = 'Error in prediction';
    });
});
