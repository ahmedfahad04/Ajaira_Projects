import pickle
import re
import string
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk import ne_chunk, pos_tag
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from transformers import BertModel, BertTokenizer
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


def main():

    df = pd.read_csv("../dataset/cleaned_twitter_dataset.csv")
    X_text = df["text"].tolist()
    y = (
        df["sentiment"].map(
            {"Negative": -1, "Neutral": 0, "Positive": 1, "Irrelevant": 0}
        )
    ).tolist()

    # Generate embeddings for all sentences
    X_embeddings = np.array([get_embedding(cleaner(text)) for text in X_text])

    # Initialize KNN with a specific distance metric
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",  # You can change this to 'euclidean', 'manhattan', etc.
    )

    # Train the model
    knn.fit(X_embeddings, y)

    print("Model trained successfully!")

    # Save the trained model
    pickle.dump(knn, open("../model/knn_model_with_embeddings.model", "wb"))


if __name__ == "__main__":
    main()
