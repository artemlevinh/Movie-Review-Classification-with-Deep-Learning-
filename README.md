# Project Title

## Overview

This project focuses on sentiment analysis of movie reviews using deep learning techniques, specifically recurrent neural networks (RNNs) and long short-term memory (LSTM) networks. The goal is to classify movie reviews as positive or negative based on the text content. The data preprocessing includes exploratory data analysis (EDA) and visualization to better understand the dataset and extract meaningful features. We utilize the Keras Tokenizer to preprocess the text data and convert it into a format suitable for training the neural network. Additionally, we use the Keras Tuner to tune the model within a range of hyperparameters for optimal performance.

In the realm of natural language processing (NLP), sentiment analysis plays a crucial role in understanding the public's opinion and gauging the relevance of content. By analyzing movie reviews, we can gain insights into the audience's perception of films, which is valuable for filmmakers, producers, and distributors. Traditional neural networks may not be as effective in capturing the context and nuances of natural language, which is why we opt for RNNs and LSTMs, as they are better suited for sequential data like text.

## Project Structure 

## Part 1 Data Analysis & Preprocess

### 1. Get data and set up train/test splits
- Load the data and split it into training and testing sets: X_train, X_test, y_train.


### 2. Data Visulaization and Preprocessing
-In the exploratory data analysis (EDA) phase, we visualizes the class balance, review lengths, and word frequency per class for movie reviews. We CountVectorizer to analyze word frequency, matplotlib and seaborn for plotting, and WordCloud to generate a visual representation of word frequency.
- Defines a custom tokenizer using NLTK's WordNet lemmatizer and a TF-IDF vectorizer. It preprocesses text data by tokenizing, lemmatizing, and removing stopwords, then fits the vectorizer to the training data and transforms both the training and test data into TF-IDF features.
- Write and save a preprocessor function for later use.

### 3. Fit model on preprocessed data
- Fit RNN, LSTM, and  on the preprocessed data.
- Save the preprocessor function and the trained model to local ".onnx" file

### 4. Generate predictions from X_test data
- Use the trained model to generate predictions from the X_test data.

### 5. Submit model to competition
- Submit the generated predictions to the competition and record the results.
- Repeat the submission process to improve the placement on the leaderboard.

## Usage

### Requirements
- Python 3.x
- Sklearn
- 

### Installation
```bash
pip install -r requirements.txt
```

## Running the code

To train the model and generate predictions, run the following command:

```bash
python main.py
```

