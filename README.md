# 🌟 Twitter(X) Sentiment Analysis Web App 🌟

This is a simple web app that uses a **Logistic Regression** model to analyze the sentiment of a tweet. Built using **Streamlit**, it classifies tweets as **Positive** or **Negative** and displays a confidence score for the prediction.

## Features

- **Tweet Sentiment Classification**: Predicts whether a tweet is positive or negative using a Logistic Regression model.
- **Confidence Score**: Displays the confidence level of the sentiment classification.
- **Interactive Interface**: Built with Streamlit, providing an intuitive and attractive user experience.
- **Clean and Simple Design**: Emphasizes usability with a modern, friendly UI.

## Demo

You can try out the app [https://sentiment-analysis-x.streamlit.app/]([https://share.streamlit.io/your-app](https://sentiment-analysis-x.streamlit.app/)).

## How It Works

The app is powered by a **Logistic Regression** model trained on a labeled dataset of tweets. The workflow includes the following steps:

1. **Text Preprocessing**: The tweet is cleaned by removing non-alphabetical characters and stopwords.
2. **TF-IDF Vectorization**: The cleaned text is converted into numerical format using **TF-IDF** vectorization.
3. **Sentiment Prediction**: The Logistic Regression model predicts whether the tweet has a positive or negative sentiment.
4. **Confidence Score**: The probability of the sentiment classification is displayed as a confidence score.

## Installation

To run this app locally, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
    ```

2. **Install the required dependencies**:

    Ensure you have Python installed (preferably version 3.8+). Install the required packages using `pip`


3. **Download NLTK stopwords (optional)**:

    In case you haven't already downloaded NLTK's stopwords, run this command:

    ```bash
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
    ```

4. **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

5. **Open your browser**:

    The app will be available at `http://localhost:8501`.


