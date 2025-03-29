import requests
import streamlit as st

# Title of the app
st.title("Sentiment Analyzer")

# Input box for sentence
sentence = st.text_input("Enter your sentence here:")

# Display character count when sentence is entered
if sentence:
    # You can either calculate locally:
    # char_count = len(sentence)

    try:
        response = requests.post(
            "http://localhost:8000/predict-sentiment", json={"sentence": sentence}
        )
        char_count = response.json()["sentiment"]
    except:
        char_count = "Error connecting to API"

    # Display the count in a metric widget
    st.metric(label="Sentence Sentiment", value=char_count)

# Add some basic info
st.write("Enter a sentence above to find out user sentiment.")
