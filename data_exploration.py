import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 1. Class Balance
def plot_class_balance(labels):
    class_counts = labels.value_counts()
    print("Class Counts:\n", class_counts)
    plt.figure(figsize=(8, 4))
    sns.countplot(labels)
    plt.title('Distribution of Classes')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.show()

#plot_class_balance(y_train_labels)

# 2. Review Lengths
def plot_review_lengths(reviews):
    review_lengths = reviews.apply(len)
    print("Review Lengths Stats:\n", review_lengths.describe())
    plt.figure(figsize=(10, 5))
    sns.histplot(review_lengths, bins=30, kde=True)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Length of Reviews')
    plt.ylabel('Number of Reviews')
    plt.show()

#plot_review_lengths(X_train)

# 3. Word Frequency per Class
def plot_word_frequency(texts, labels, num_features=20):
    # Initialize the CountVectorizer with a limited number of features for simplicity
    vect = CountVectorizer(stop_words='english', max_features=num_features)

    # Separate texts by class for frequency analysis
    positive_texts = texts[labels == 'Positive']  # Filter texts where labels are 'Positive'
    negative_texts = texts[labels == 'Negative']  # Filter texts where labels are 'Negative'

    # Fit and transform the texts for positive class
    vect.fit(positive_texts)
    pos_dtm = vect.transform(positive_texts)
    pos_sum_words = pos_dtm.sum(axis=0)
    pos_words_freq = [(word, pos_sum_words[0, idx]) for word, idx in sorted(vect.vocabulary_.items(), key=lambda item: item[1])]

    # Fit and transform the texts for negative class
    vect.fit(negative_texts)
    neg_dtm = vect.transform(negative_texts)
    neg_sum_words = neg_dtm.sum(axis=0)
    neg_words_freq = [(word, neg_sum_words[0, idx]) for word, idx in sorted(vect.vocabulary_.items(), key=lambda item: item[1])]

    # Create DataFrames for each class
    df_pos = pd.DataFrame(pos_words_freq, columns=['word', 'positive_frequency']).nlargest(10, 'positive_frequency')
    df_neg = pd.DataFrame(neg_words_freq, columns=['word', 'negative_frequency']).nlargest(10, 'negative_frequency')

    # Print top 10 words for each class
    print("Top 10 Positive Words:")
    print(df_pos)
    print("\nTop 10 Negative Words:")
    print(df_neg)

    # Merge the two dataframes on the word column
    df_final = pd.merge(df_pos, df_neg, on="word", how="outer").fillna(0)

    # Plotting
    df_final.set_index('word').plot.barh(figsize=(12, 10), width=0.7)
    plt.xlabel('Frequency of Words')
    plt.title('Top Word Frequencies Per Class')
    plt.show()

# Example usage
#plot_word_frequency(X_train, y_train_labels)

# 4. Wordcloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

#generate_wordcloud(X_train)
