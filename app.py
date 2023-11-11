import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the pickle file from your Google Drive
with open('sentiment_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

classifier = model_data['model']
tfidf_vectorizer = model_data['vectorizer']


def perform_sentiment_analysis(text):
  # Convert the text to a TF-IDF matrix
  tfidf_matrix = tfidf_vectorizer.transform([text])

  # Make a prediction using the trained model
  sentiment_prediction = classifier.predict(tfidf_matrix)

  # Return the sentiment prediction
  return sentiment_prediction

st.title('Sentiment Analysis Web App')

# Get the user input text
text = st.text_input('Enter text to analyze:')

# Perform sentiment analysis on the user input text
sentiment_prediction = perform_sentiment_analysis(text)

# Display the sentiment prediction to the user
if st.button('Analyze Sentiment'):
    if sentiment_prediction[0] == 2:
        st.write('Sentiment Prediction: Positive')
    else:
        st.write('Sentiment Prediction: Negative')
  


