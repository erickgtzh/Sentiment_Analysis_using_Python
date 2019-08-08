"""
USING TWITTER DATABASE
Created on Sat Apr 13 14:22:27 2019

@author: Erick
"""
import nltk
from nltk.corpus import stopwords
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
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from nltk.stem.snowball import SnowballStemmer
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
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from stop_words import get_stop_words

auth = tw.OAuthHandler("dDVlY4yNzjY5G08rcnzHnACcC","ti1I3gzK1yHLfTamXtkUr0H24xvtEF5m8dQhC9F9i3WHBSBmnv")
auth.set_access_token("852765721046130688-WSjFq74sdENxSBocXjld3mcDBlmND0a", "Xl7jf5sfCkeURWEMZlyPQ9zy9EXr4NcQJwry1trFod5Tp")
api = tw.API(auth)
    
#Be careful using it because the following likes delete all my tweets! :'(
for status in tw.Cursor(api.user_timeline).items():
    try:
        api.destroy_status(status.id)
    except:
        pass
    
api.update_status("Hello, tweepy!")

#------------------------Now you're out of danger-------------------------#

#search_words ="#AMLO"

    
search_words = "#AMLO"
filtered_tweets = []
data_since ="2016-11-16"

tweets = tw.Cursor(api.search,
  q =search_words,
  lang="es",
  since=data_since).items(50)

new_search = search_words + "-filter:retweets"

tweets = tw.Cursor(api.search,
    	q=new_search,
    	lang="es",
    	since=data_since).items(50)
    #polarity.append(TextBlob(tweet.text).sentiment.polarity)
    #final_tweets.append(tweet.text)

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

#data = pd.read_csv("Ejercicio1_ElEconomista.mx.csv",names=['Tweets'],header=None, encoding='latin-1')
data_aux = data

#Pre-processing
def processTweet(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','',tweet)
    # Remove tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    # To lowercase
    tweet = tweet.lower()
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    # Remove words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # Remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # Remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ') 
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
    return tweet
# ______________________________________________________________
# clean dataframe's text column
data['Tweets'] = data['Tweets'].apply(processTweet)
# preview some cleaned tweets
data['Tweets'].head()

# most common words in twitter dataset
all_words = []
for line in list(data['Tweets']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())
# plot word frequency distribution of first few words
plt.figure(figsize=(12,5))
plt.xticks(fontsize=13, rotation=90)
fd = nltk.FreqDist(all_words)
fd.plot(25,cumulative=False)
plt.show()

def text_process(raw_text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in list(raw_text) if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.lower().split() if word.lower() not in nltk.corpus.stopwords.words("spanish")]

def remove_words(word_list):
    remove = ['@','RT','#','!','¡','¿','?','text']
    return [w for w in word_list if w not in remove]

def delete_punctuation(cadena):
    s = []
    for word in cadena:
        s.append(''.join((c for c in unicodedata.normalize('NFD',word) if unicodedata.category(c) != 'Mn')))
    return s
# -------------------------------------------
# tokenize message column and create a column for tokens
data = data.copy()
aux = data
data['tokens'] = data['Tweets'].apply(text_process) # tokenize style 1
data['Tweets'] = data['tokens'].apply(remove_words) #tokenize style 2

# split sentences to get individual words
all_words = []
for line in data['tokens']: # try 'tokens'
    all_words.extend(line)
    
# create a word frequency dictionary
wordfreq = Counter(all_words)
# draw a Word Cloud with word frequencies
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate_from_frequencies(wordfreq)
plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# vectorize
bow_transformer = CountVectorizer(analyzer=text_process).fit(data['Tweets'])
# print total number of vocab words
print(len(bow_transformer.vocabulary_))

# transform the entire DataFrame of messages
messages_bow = bow_transformer.transform(data['Tweets'])
# check out the bag-of-words counts for the entire corpus as a large sparse matrix
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

# from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

import indicoio
indicoio.config.api_key ="116ef74d6866c78baddc8edc57974fa9"

sentiment = []
for tuit in data_aux['Tweets']:
    sentiment.append(indicoio.sentiment(tuit))

test_sent = sentiment

neu_tw = []
neg_tw = []
pos_tw = []

for x in range (len(test_sent)):
    #positive
    if(test_sent[x]>.6):
        pos_tw.append(data['Tweets'][x])
        test_sent[x] = 1
    #negative
    elif(test_sent[x]<.4):
        neg_tw.append(data['Tweets'][x])
        test_sent[x] = -1
    #neutral
    else:
        neu_tw.append(data['Tweets'][x])
        test_sent[x] = 0

y = test_sent
X = data['Tweets']

# Run Train Data Through Pipeline analyzer=text_process
# uncomment below to train on a larger dataset but it's very slow for a slower machine.
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)

# Run Train Data Through Pipeline analyzer=text_process
# uncomment below to train on a larger dataset but it's very slow for a slower machine.
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
# create pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii',
                            stop_words= nltk.corpus.stopwords.words("spanish"),
                            lowercase=True)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
# this is where we define the values for GridSearchCV to iterate over
parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3),
             }
# do 10-fold cross validation for each of the 6 possible combinations of the above params
grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
grid.fit(X_train,y_train)
# summarize results
print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
print('\n')
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))

# save best model to current working directory
joblib.dump(grid, "twitter_sentiment.pkl")
# load from file and predict using the best configs found in the CV step
model_NB = joblib.load("twitter_sentiment.pkl" )
# get predictions from best model above
y_preds = model_NB.predict(X_test)
print('accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')
print('confusion matrix: \n',confusion_matrix(y_test,y_preds))
print('\n')
print(classification_report(y_test, y_preds))

# run predictions on twitter data
tweet_preds = model_NB.predict(data['Tweets'])
# append predictions to dataframe
df_tweet_preds = data.copy()
df_tweet_preds['predictions'] = tweet_preds
df_tweet_preds.shape

neutral = df_tweet_preds.predictions.value_counts()[0]
negative = df_tweet_preds.predictions.value_counts()[-1]
positive = df_tweet_preds.predictions.value_counts()[1]

print('Model predictions: Positives - {}, Negatives - {}, Neutral - {}'.format(positive,negative,neutral))

# save dataframe with appended preditions 
df_tweet_preds.to_pickle('edgh_predicts_df.p')

#real
plt.ion()
plt.pie([len(pos_tw),len(neg_tw),len(neu_tw)], labels = ["positive","negative","neutral"])
plt.title('Sentiment analysis using Twitter')
plt.show()

#predicted
plt.ion()
plt.pie([positive,negative,neutral], labels = ["positive","negative","neutral"])  # Dibuja un gráfico de quesitos
plt.title('Sentiment analysis using Twitter')
plt.show()

# create a word frequency dictionary for negative
wordfreq = Counter(neg_tw)
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
wordfreq = Counter(pos_tw)
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

# create a word frequency dictionary for neutral
wordfreq = Counter(neu_tw)
# draw a Word Cloud with word frequencies
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate_from_frequencies(wordfreq)
plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()