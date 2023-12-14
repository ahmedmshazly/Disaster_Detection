from flask import Flask, render_template, request, jsonify
import pandas as pd
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
import dill


app = Flask(__name__)

# Define global variables
vectorizer = None
model = "aa"

model_url = 'https://drive.google.com/uc?id=1r46OXcPY-YV1AM8GVmj8KMGs6t_NdLxG'
model_path = 'model.pkl'  # Choose a local path to save the model file
response = requests.get(model_url)
with open('model.pkl', 'rb') as file:
    vectorizer, model = dill.load(file)

@app.route('/upload')
def upload():
    return app.send_static_file('upload.html')

@app.route('/process_uploaded_file', methods=['POST'])
def process_uploaded_file():
    global vectorizer, model 
    try:
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Check if the uploaded file has a .csv extension
            if uploaded_file.filename.endswith('.csv'):
                # Read data from CSV
                data = pd.read_csv(uploaded_file)
                num_tweets = len(data)
                # Process the specified number of tweets
                processed_data = data.head(num_tweets)
                processed_data['cleaned_text'] = processed_data['text'].apply(clean_text).apply(lemmatize_text)
                vectorized_text = vectorizer.transform(processed_data['cleaned_text'])
                predictions = model.predict(vectorized_text)

                # Calculate the average prediction
                avg_prediction = predictions.mean()

                # Determine disaster level
                if avg_prediction < 0.3:
                    disaster_level = "Green"
                elif avg_prediction < 0.6:
                    disaster_level = "Yellow"
                else:
                    disaster_level = "Red"

                # Create a list of tweet texts and their predictions
                tweet_predictions = [{'text': text, 'prediction': 'Disaster' if pred == 1 else 'Neutral'} for text, pred in zip(processed_data['text'], predictions)]

                return jsonify({'status': disaster_level, 'processed_tweets': num_tweets, 'disaster_average': avg_prediction, 'tweet_predictions': tweet_predictions})

            else:
                return "Please upload a valid CSV file."
        else:
            return "No file uploaded. Please select a CSV file."

    except Exception as e:
        return str(e)

@app.route('/predict', methods=['POST'])
def predict():
    global vectorizer, model
    try:
        # Reading the number of tweets to process from the request
# Reading the number of tweets to process from the request
        num_tweets = request.json.get('num_tweets', None)

        # Convert num_tweets to an integer
        num_tweets = int(num_tweets) if num_tweets is not None else None
        
        # Read data from CSV
        data = pd.read_csv('data.csv')

        # Check if num_tweets is more than available tweets
        if num_tweets is None or num_tweets > len(data):
            num_tweets = len(data)

        # Process the specified number of tweets
        processed_data = data.head(num_tweets)
        processed_data['cleaned_text'] = processed_data['text'].apply(clean_text).apply(lemmatize_text)
        vectorized_text = vectorizer.transform(processed_data['cleaned_text'])
        predictions = model.predict(vectorized_text)

        # Calculate the average prediction
        avg_prediction = predictions.mean()

        # Determine disaster level
        if avg_prediction < 0.3:
            disaster_level = "Green"
        elif avg_prediction < 0.6:
            disaster_level = "Yellow"
        else:
            disaster_level = "Red"

        # Create a list of tweet texts and their predictions
        tweet_predictions = [{'text': text, 'prediction': 'Disaster' if pred == 1 else 'Neutral'} for text, pred in zip(processed_data['text'], predictions)]

        return jsonify({'status': disaster_level, 'processed_tweets': num_tweets, 'disaster_average': avg_prediction, 'tweet_predictions': tweet_predictions})

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({'error': str(e)})


@app.route('/predict_single_tweet')
def predict_single_tweet():
    return app.send_static_file('single.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    global vectorizer, model
    try:
        data = request.json
        tweet_text = data['tweet_text']

        prediction_result = predict_from_saved_model_single(tweet_text, 'model.pkl')
        # Convert numerical prediction to text
        prediction_text = "Disaster" if prediction_result == 1 else "Neutral"

        return jsonify({'prediction': prediction_text})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/about')
def about():
    return app.send_static_file('about.html')


#to clean text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#','', text)

    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    text_tokens = text.split()

    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(token) for token in text_tokens if token not in stopwords.words('english')]

    return ' '.join(text)


def lemmatize_text(text):
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Clean the text: Lowercasing, Removing non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Lemmatize each word in the text
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokens])

    return lemmatized_output

def predict_from_saved_model_dataframe(data, model_filename):
    """
    Predicts the classes for a DataFrame of texts using the saved model.

    Args:
        data (pd.DataFrame): The DataFrame containing the text to predict.
        model_filename (str): The filename of the saved model.

    Returns:
        A DataFrame with the predictions.
    """

    # Load the saved model and vectorizer
    with open(model_filename, 'rb') as file:
        vectorizer, model = pickle.load(file)

    # Preprocess the text in the DataFrame
    data['cleaned_text'] = data['text'].apply(clean_text).apply(lemmatize_text)

    # Transform the text using the loaded vectorizer
    vectorized_text = vectorizer.transform(data['cleaned_text'])

    # Make predictions
    predictions = model.predict(vectorized_text)

    # Add predictions to the DataFrame
    data['predictions'] = predictions
    return data

import pickle

def predict_from_saved_model_single(text, model_filename):
    """
    Predicts the class of a single text instance using the saved model.

    Args:
        text (str): The text instance to predict.
        model_filename (str): The filename of the saved model.

    Returns:
        The predicted class.
    """

    # Load the saved model and vectorizer
    with open(model_filename, 'rb') as file:
        vectorizer, model = pickle.load(file)

    # Preprocess the text (using your existing functions)
    cleaned_text = clean_text(text)  # Assuming clean_text function is already defined
    lemmatized_text = lemmatize_text(cleaned_text)  # Assuming lemmatize_text function is already defined

    # Transform the text using the loaded vectorizer
    vectorized_text = vectorizer.transform([lemmatized_text])

    # Make a prediction
    prediction = model.predict(vectorized_text)

    return prediction[0]  # Returning the first (and only) prediction


@app.route('/')
def home():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)