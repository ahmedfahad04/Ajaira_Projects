import pickle
import re
import string
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk import ne_chunk, pos_tag
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("punkt_tab")

import warnings

warnings.filterwarnings("ignore")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and effective


# Function to get SentenceTransformer embeddings
def get_embedding(sentence):
    embedding = embedding_model.encode(sentence, convert_to_numpy=True)
    return embedding


def cleaner(data):

    stop_words = set(stopwords.words("english"))

    # Tokens
    tokens = word_tokenize(str(data).replace("'", "").lower())

    # Remove Puncs
    without_punc = [w for w in tokens if w.isalpha()]

    # Stopwords
    without_sw = [t for t in without_punc if t not in stop_words]

    # Lemmatize
    text_len = [WordNetLemmatizer().lemmatize(t) for t in without_sw]

    # Stem
    text_cleaned = [PorterStemmer().stem(w) for w in text_len]

    return " ".join(text_cleaned)


def predict_sentiment(new_sentence):

    loaded_vt = pickle.load(open("model/vectorizer.pickle", "rb"))
    loaded_model = pickle.load(open("model/knn_model.model", "rb"))

    # Load the trained KNN model
    new_sentence_cleaned = cleaner(new_sentence)

    # Transform the new sentence using the CountVectorizer
    new_sentence_vectorized = loaded_vt.transform([new_sentence_cleaned])

    # Predict the sentiment
    predicted_sentiment = loaded_model.predict(new_sentence_vectorized)

    print("Predicted Sentiment:", predicted_sentiment[0])

    sentiment_map = {
        -1: "Negative",
        0: "Neutral",
        1: "Positive",
    }

    return sentiment_map[predicted_sentiment[0]]


def predict_sentiment_using_nltk(new_sentence):
    # Load the trained KNN model (now trained on embeddings)
    loaded_model = pickle.load(open("../model/knn_model_with_embeddings.model", "rb"))

    # Clean the new sentence
    new_sentence_cleaned = cleaner(new_sentence)

    # Get BERT embedding for the new sentence
    new_sentence_embedding = get_embedding(new_sentence_cleaned)

    # Reshape for KNN (expects 2D array)
    new_sentence_embedding = new_sentence_embedding.reshape(1, -1)

    # Predict the sentiment
    predicted_sentiment = loaded_model.predict(new_sentence_embedding)

    print("Predicted Sentiment:", predicted_sentiment[0])

    sentiment_map = {
        -1: "Negative",
        0: "Neutral",
        1: "Positive",
    }

    return sentiment_map[predicted_sentiment[0]]


# Example usage
new_sentence = "Football is my favorite sports!!"
result = predict_sentiment_using_nltk(new_sentence)
print(result)
