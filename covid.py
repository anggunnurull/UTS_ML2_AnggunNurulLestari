import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load model, vectorizer, and scaler
model = joblib.load('sentiment_analysis_model.sav')
tfidf = joblib.load('tfidf_vectorizer.sav')
scaler = joblib.load('scaler.sav')

# Label mapping
label_map = {
    0: "Extremely Negative",
    1: "Negative",
    2: "Positive",
    3: "Neutral"
}

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Corona_NLP_test.csv")  # Pastikan file CSV ada
    df = df[['Location', 'TweetAt', 'OriginalTweet']].dropna()
    return df

df = load_data()

st.title("COVID-19 Tweet Sentiment Analysis with Multiple Filters")

# 1. Pilih lokasi
location_options = sorted(df['Location'].unique())
selected_location = st.selectbox("Select a location:", location_options)

# Filter berdasarkan lokasi
df_filtered_location = df[df['Location'] == selected_location]

# 2. Pilih tanggal TweetAt
date_options = sorted(df_filtered_location['TweetAt'].unique())
selected_date = st.selectbox("Select a tweet date:", date_options)

# Filter berdasarkan lokasi dan tanggal
df_filtered = df_filtered_location[df_filtered_location['TweetAt'] == selected_date]

# 3. Pilih tweet
tweet_options = df_filtered['OriginalTweet'].tolist()
selected_tweet = st.selectbox("Select a tweet:", tweet_options)

# Tombol untuk analisis
if st.button("Analyze Sentiment"):
    try:
        tweet_tfidf = tfidf.transform([selected_tweet])
        tweet_scaled = scaler.transform(tweet_tfidf)

        prediction = model.predict(tweet_scaled)
        predicted_label = label_map.get(prediction[0], "Unknown")

        st.success(f"Predicted Sentiment: {predicted_label}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
