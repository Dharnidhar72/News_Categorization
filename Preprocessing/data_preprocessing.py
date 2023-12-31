# -*- coding: utf-8 -*-
"""Data Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q8uLpt39gDnlmB408KXJbEtPS1OXmmxo
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import nltk
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud
from collections import Counter
import json

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

with open('/News_Category_Dataset_v3.json','r') as f:
    jdata = f.read()

jdata2  = [json.loads(line) for line in jdata.split('\n') if line]
df = pd.DataFrame.from_records(jdata2)

# Check for empty strings or empty lists/objects in each column
def check_empty_values(series):
    return (series.apply(lambda x: x == '' or x == [] or x == {})).sum()

empty_values_per_column = df.apply(check_empty_values)

# Load your dataframe here
# df = pd.read_csv('path_to_your_csv.csv')

# Replace empty descriptions with the headline
df['short_description'] = df.apply(
    lambda row: row['headline'] if pd.isnull(row['short_description']) or row['short_description'] == '' else row['short_description'],
    axis=1
)

# Load your dataframe here
# df = pd.read_csv('path_to_your_csv.csv')

# Replace empty strings with NaN
df['short_description'].replace('', pd.NA, inplace=True)

# Drop rows where 'short_description' is NaN
df.dropna(subset=['short_description'], inplace=True)

# Now df no longer contains rows with empty 'short_description' entries.

# Find the top 10 categories
top_categories = df['category'].value_counts().nlargest(10).index

# Filter the DataFrame to include only the top 8 categories
df_filtered = df[df['category'].isin(top_categories)]

# Drop all other columns except 'category' and 'short_description'
df_final = df_filtered[['category', 'short_description']]

# You can now work with df_final which contains only the data you want

print(df_final['category'].value_counts())

# Assuming df_final is your dataframe and it has been loaded correctly.

# Shuffle the dataframe
df_final = df_final.sample(frac=1).reset_index(drop=True)

# Filter out the 'POLITICS' category
politics_df = df_final[df_final['category'] == 'POLITICS']
other_categories_df = df_final[df_final['category'] != 'POLITICS']

# Sample 12,000 entries from the 'POLITICS' category
politics_sampled_df = politics_df.sample(n=17500)

# Concatenate the sampled 'POLITICS' with the rest of the categories
df_final = pd.concat([politics_sampled_df, other_categories_df]).sample(frac=1).reset_index(drop=True)

# Upsampling the data
SEED = 42
df_list = []
#Get news in top 15 categories
for i in top_categories:
    df_list.append(pd.DataFrame(df_final[df_final["category"]==i]))
for i in range(len(df_list)):
    df_list[i] = pd.DataFrame(df_list[i][df_list[i]["short_description"]!=""])
for i in range(len(df_list)):
    df_list[i] = df_list[i].sample(df_list[0].shape[0], replace=True, random_state=SEED)
df_upsample = pd.concat(df_list)
print(df_upsample.shape)

import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Split the dataset
train, test = train_test_split(df_upsample, test_size=0.2, random_state=42)

# Apply preprocessing to the 'short_description' column
train['clean_text'] = train['short_description'].apply(preprocess_data)
test['clean_text'] = test['short_description'].apply(preprocess_data)

# Now 'train' and 'test' DataFrames have a 'clean_text' column with preprocessed text

import joblib

train_reduced = train[['category','clean_text']]
test_reduced = test[['category','clean_text']]

from sklearn.feature_extraction.text import TfidfVectorizer

# Function to transform data using TF-IDF
def transform_data_tfidf(train_data, test_data, text_column, target_column):
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7, min_df=3, ngram_range=(1,2))

    # Fit and transform the training data
    X_train_transformed = tfidf_vectorizer.fit_transform(train_data[text_column])
    y_train = train_data[target_column]

    # Transform the test data
    X_test_transformed = tfidf_vectorizer.transform(test_data[text_column])
    y_test = test_data[target_column]

    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

    return X_train_transformed, y_train, X_test_transformed, y_test

# Applying the transformation to your dataset
X_train_tfidf, y_train, X_test_tfidf, y_test = transform_data_tfidf(train_reduced, test_reduced, 'clean_text', 'category')