# -*- coding: utf-8 -*-
"""Data_Analysis.ipynb

Use the following code to explore the data.
"""

!pip install -q aimodelshare

from aimodelshare import download_data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
from data_exploration import plot_class_balance, plot_review_lengths, plot_word_frequency, generate_wordcloud
from data_preprocessing import to_series

download_data('public.ecr.aws/y2e2a1d6/sst2_competition_data-repository:latest')

X_train=pd.read_csv("sst2_competition_data/X_train.csv")
X_test=pd.read_csv("sst2_competition_data/X_test.csv")
y_train_labels=pd.read_csv("sst2_competition_data/y_train_labels.csv")

X_train=to_series(X_train)
X_test=to_series(X_test)
y_train_labels=to_series(y_train_labels)

plot_class_balance(y_train_labels)

plot_review_lengths(X_train)

plot_word_frequency(X_train, y_train_labels)

generate_wordcloud(X_train)
