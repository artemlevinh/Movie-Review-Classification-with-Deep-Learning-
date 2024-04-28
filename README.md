# Project Title

## Overview

This project focuses on sentiment analysis of movie reviews using deep learning techniques, specifically recurrent neural networks (RNNs) and long short-term memory (LSTM) networks. The goal is to classify movie reviews as positive or negative based on the text content. The data preprocessing includes exploratory data analysis (EDA) and visualization to better understand the dataset and extract meaningful features. We utilize the Keras Tokenizer to preprocess the text data and convert it into a format suitable for training the neural network. Additionally, we use the Keras Tuner to tune the model within a range of hyperparameters for optimal performance.

In the realm of natural language processing (NLP), sentiment analysis plays a crucial role in understanding the public's opinion and gauging the relevance of content. By analyzing movie reviews, we can gain insights into the audience's perception of films, which is valuable for filmmakers, producers, and distributors. Traditional neural networks may not be as effective in capturing the context and nuances of natural language, which is why we opt for RNNs and LSTMs, as they are better suited for sequential data like text.
## Instructions

### 1. Get data and set up train/test splits
- Load the data and split it into training and testing sets: X_train, X_test, y_train.

### 2. Preprocess data using Sklearn TFIDF Vectorizer
- Use the Sklearn TFIDF Vectorizer to preprocess the text data.
- Write and save a preprocessor function for later use.

### 3. Fit model on preprocessed data
- Fit a model on the preprocessed data.
- Save the preprocessor function and the trained model for future use.

### 4. Generate predictions from X_test data
- Use the trained model to generate predictions from the X_test data.

### 5. Submit model to competition
- Submit the generated predictions to the competition and record the results.
- Repeat the submission process to improve the placement on the leaderboard.

## Usage

### Requirements
- Python 3.x
- Sklearn
- [Any other dependencies]

### Installation
```bash
pip install -r requirements.txt
