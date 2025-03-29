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
