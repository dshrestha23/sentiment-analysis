import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
with open("models/model_lg.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Preprocessing function to clean the text
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub("[^a-zA-Z]", " ", text)  # Remove non-alphabetical characters
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    clean_words = [word for word in words if word not in stop_words]
    return " ".join(clean_words)


# App title and description
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🌟 Twitter(X) Sentiment Analysis 🌟</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Use AI to classify the sentiment of any tweet as <b>positive</b> or <b>negative</b>.</p>",
    unsafe_allow_html=True,
)
st.markdown("#### 👇 Enter the tweet you want to analyze:")

# User input form
tweet_input = st.text_area("Tweet Input")

# Sentiment analysis button
if st.button("Analyze Sentiment"):
    if tweet_input:
        # Preprocess the input tweet
        clean_tweet = preprocess_text(tweet_input)

        # Convert the cleaned tweet to the format required by the model (e.g., TF-IDF vector)
        tweet_vector = vectorizer.transform([clean_tweet])

        # Predict the sentiment using the loaded model
        prediction = model.predict(tweet_vector)
        prediction_proba = model.predict_proba(tweet_vector)

        # Mapping sentiment labels
        sentiment_map = {0: "Negative 😔", 1: "Positive 😊"}
        sentiment = sentiment_map[prediction[0]]

        # Display the result with confidence level
        st.markdown(f"### The sentiment of the tweet is: **{sentiment}**")
        st.markdown(
            f"### Confidence Score: **{prediction_proba[0][prediction[0]] * 100:.2f}%**"
        )

        # Adding emojis for visual appeal
        if prediction[0] == 1:
            st.success("This tweet has a positive vibe! 😄")
        else:
            st.error("This tweet expresses negative sentiment. 😞")

    else:
        st.warning("Please enter a valid tweet for analysis.")
else:
    st.markdown(
        "<p style='text-align: center; color: #FF6347;'>✨ Try inputting a tweet and click 'Analyze Sentiment' ✨</p>",
        unsafe_allow_html=True,
    )

# Footer
st.markdown("<hr style='border:1px solid #4CAF50;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Created by <b>Dipesh Shrestha</b></p>",
    unsafe_allow_html=True,
)
