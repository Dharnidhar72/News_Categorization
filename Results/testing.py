import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn.model_selection import train_test_split




import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')  # This ensures that the stopwords are downloaded and available
stop_words = set(stopwords.words('english'))  # This initializes the stop_words variable

# Ensure you have these NLTK datasets downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Defining the stemmer
stemmer = SnowballStemmer("english")

# Updated stopwords list
stop_words = set(stopwords.words('english') + ['u', 'im', 'c'])


# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Function to remove stopwords
def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word not in stop_words)

# Function to stem text
def stemm_text(text):
    return ' '.join(stemmer.stem(word) for word in text.split())



# Function to preprocess data
def preprocess_data(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = stemm_text(text)
    return text




import joblib

import joblib

# Load the combined model and vectorizer from the single joblib file
combined_model = joblib.load('combined_model.joblib')  # Replace with the path to your single joblib file
vectorizer = joblib.load('tfidf_vectorizer.joblib')  # Replace with the path to your TF-IDF vectorizer joblib file

# Modify the predict_category function to include feature extraction
def predict_category(news_text):
    # Preprocess the input news text (if you have a specific preprocessing function)
    processed_text = preprocess_data(news_text)  # Uncomment and use your preprocessing function

    # Transform the text using the loaded vectorizer
    input_text_tfidf = vectorizer.transform([processed_text])  # Ensure input_text is correctly preprocessed if necessary

    predictions = {}
    for model_name, model in combined_model.items():
        # Make predictions using each algorithm
        predicted_category = model.predict(input_text_tfidf)

        # Store the prediction for the model
        predictions[model_name] = predicted_category[0]  # Assuming binary classification

    return predictions

# Example input news text
input_news_text = input("Enter the news you want to classify: ")
# Call the predict_category function
predictions = predict_category(input_news_text)

# Output the predicted categories from each algorithm
for model_name, prediction in predictions.items():
    print(f"{model_name} predicted category: {prediction}")
