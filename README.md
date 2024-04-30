# Movie-Review-Classification-with-Machine Learning

## Overview

This project conducts a binary classificatoin task on movie reviews using tradition machine learning (support vector machines, regression trees, gradient boosting, etc) and deep learning techniques, specifically traditionrecurrent neural networks (RNNs) and long short-term memory (LSTM) networks. The goal is to classify movie reviews as positive or negative based on the text content. The data preprocessing includes exploratory data analysis (EDA) and visualization to better understand the dataset and extract meaningful features. We utilize the Keras Tokenizer to preprocess the text data and convert it into a format suitable for training the neural network. Additionally, we use the Keras Tuner to tune the model within a range of hyperparameters for optimal performance.

In the realm of natural language processing (NLP), sentiment analysis plays a crucial role in understanding the public's opinion and gauging the relevance of content. By analyzing movie reviews, we can gain insights into the audience's perception of films, which is valuable for filmmakers, producers, and distributors. Traditional neural networks may not be as effective in capturing the context and nuances of natural language, which is why we opt for RNNs and LSTMs, as they are better suited for sequential data like text.

## Project WorkFlow

## Part 1 Data Analysis & Preprocess

### 1. Get data and set up train/test splits
- Load the data and split it into training and testing sets: X_train, X_test, y_train.


### 2. Data Visulaization and Preprocessing
-In the exploratory data analysis (EDA) phase, we visualizes the class balance, review lengths, and word frequency per class for movie reviews. We CountVectorizer to analyze word frequency, matplotlib and seaborn for plotting, and WordCloud to generate a visual representation of word frequency.
- Defines a custom tokenizer using NLTK's WordNet lemmatizer and a TF-IDF vectorizer. It preprocesses text data by tokenizing, lemmatizing, and removing stopwords, then fits the vectorizer to the training data and transforms both the training and test data into TF-IDF features.
- Write and save a preprocessor function for later use.

### 3.Designing Model Architectures and Fine_Tuning 
-In this section, we use a variety of traditional machine learning models, including support vector machines (SVMs), decision trees, and gradient boosting classifiers. These models are trained on preprocessed text data to learn patterns that differentiate between positive and negative movie reviews.
-For deep learning, we utilize recurrent neural networks (RNNs) and long short-term memory (LSTM) networks. These models are well-suited for sequence data like text, allowing them to capture sequential dependencies in movie reviews and better understand their sentiment.
-The text data is preprocessed using the Keras Tokenizer, which converts words into numerical tokens and pads sequences to ensure uniform length inputs. This preprocessing step is crucial for training neural networks on text data.
-To optimize the performance of our models, we use the Keras Tuner library for hyperparameter tuning. This involves defining a search space for hyperparameters (e.g., learning rate, number of units in LSTM layer) and using the tuner to find the optimal combination of hyperparameters for each model architecture.

### 4. Fit model on preprocessed data

- Fit Classic ML models, CNN, RNN, LSTM on the preprocessed data.
- Save the preprocessor function and the trained models to local ".onnx" file

### 5. Generate predictions from X_test data
- Use the trained model to generate predictions from the X_test data.



## Usage

### Requirements
- Python 3.x
- Sklearn
- aimodelshare
- pandas
- scikit-learn
- numpy
- nltk
- matplotlib
- seaborn
- wordcloud

### Installation
```bash
pip install -r requirements.txt
```

## Running the code

To train the model and generate predictions, run the following command:

```bash
python main.py
```

