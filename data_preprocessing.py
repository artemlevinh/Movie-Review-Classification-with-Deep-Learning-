

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def to_series(df):
    # Check if the DataFrame has only one column
    if len(df.columns) == 1:
        # Convert to Series and return
        return df.iloc[:, 0]
    else:
        # Return the DataFrame as is
        return df

class Preprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit(self, X_train):
        self.vectorizer.fit(X_train)

    def transform(self, data):
        return self.vectorizer.transform(data)

    def fit_transform(self, X_train):
        return self.vectorizer.fit_transform(X_train)

class LemmatizedTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def __call__(self, doc):
        tokens = word_tokenize(doc.lower())
        return [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words and w.isalpha()]

    def preprocessor(self, data):
        return self.vectorizer.transform(data)

    def fit(self, X_train):
        self.vectorizer = TfidfVectorizer(tokenizer=self)
        self.vectorizer.fit(X_train)

    def transform(self, data):
        return self.vectorizer.transform(data)
