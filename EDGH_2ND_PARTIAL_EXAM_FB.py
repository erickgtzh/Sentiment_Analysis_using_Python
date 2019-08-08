"""
USING FACEBOOK DATABASE
Created on Sat Apr 13 14:22:27 2019

@author: Erick
"""
import nltk
from nltk import word_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import csv
import tweepy as tw
from sklearn import preprocessing
from textblob import TextBlob
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import string
import re
from unicodedata import normalize
import unicodedata
from nltk import word_tokenize
from nltk.data import load
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import LinearSVC
stemmer = SnowballStemmer("spanish") 
from string import punctuation
from nltk.stem.porter import PorterStemmer
from nltk import wordpunct_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from stop_words import get_stop_words
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]
        
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

#Definimos el vectorizer de nuevo y creamos un pipeline de vectorizer -&gt; classificador
vectorizer = CountVectorizer(
                analyzer = 'word',
                lowercase = True,
                stop_words = spanish_stopwords)

#LinearSVC() es el clasificador

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('cls', LinearSVC()),
])


#Aqui definimos el espacio de parámetros a explorar
parameters = {
    'vect__max_df': (4,5),
    'vect__min_df': (1,2),
    'vect__max_features': (500, 1000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas or bigramas
    'cls__C': (0.2, 0.5, 0.7),
    'cls__loss': ('hinge', 'squared_hinge'),
    'cls__max_iter': (500, 1000)
}

#grid_search = GridSearchCV(pipeline, parameters, cv=3,n_jobs=1 , scoring='roc_auc')
grid_search = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
grid_search.fit(X_test,y_test)

# summarize results
print("\nBest Model: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
print('\n')
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))
    
# save best model to current working directory
joblib.dump(grid_search, "twitter_sentiment.pkl")
# load from file and predict using the best configs found in the CV step
model_NB = joblib.load("twitter_sentiment.pkl" )
# get predictions from best model above
y_preds = model_NB.predict(X_test)
print('accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')
print('confusion matrix: \n',confusion_matrix(y_test,y_preds))
print('\n')
# run predictions on twitter data
tweet_preds = model_NB.predict(data_aux['Tweets'])
# append predictions to dataframe
data_tweet_preds = data.copy()
data_tweet_preds['predictions'] = tweet_preds
data_tweet_preds.shape
print(classification_report(y_test, y_preds))

positive = data_tweet_preds['predictions'].value_counts()[1]
negative = data_tweet_preds['predictions'].value_counts()[0]
#predicted
plt.ion()
plt.pie([positive,negative], labels = ["positive","negative"])  # Dibuja un gráfico de quesitos
plt.title('Sentiment analysis using Twitter')
plt.show()

#real
plt.ion()
plt.pie([len(pos_tw),len(neg_tw)], labels = ["positive","negative"])
plt.title('Sentiment analysis using Twitter')
plt.show()

# create a word frequency dictionary for negative
words = []
for tweet in neg_tw:
    if len(tweet) > 3:
        words.append(" ".join(tweet))

wordfreq = Counter(words)
# draw a Word Cloud with word frequencies
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Reds',
                      normalize_plurals=True).generate_from_frequencies(wordfreq)
plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# create a word frequency dictionary for positive
words = []
for tweet in pos_tw:
    if len(tweet) > 3:
        words.append(" ".join(tweet))

wordfreq = Counter(words)
# draw a Word Cloud with word frequencies
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Greens',
                      normalize_plurals=True).generate_from_frequencies(wordfreq)
plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()