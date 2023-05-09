import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn import svm
from joblib import dump, load
from sklearn.pipeline import Pipeline
from langdetect import detect


stemmer = SnowballStemmer("spanish")

def tokenizer(review):
    review = review.lower()
    tokens = [stemmer.stem(token) for token in word_tokenize(review)]
    return tokens
    
def normalize(stop_words): 
    lower = {word.lower() for word in stop_words}
    return [stemmer.stem(word) for word in lower]

def is_spanish(text: str) -> bool:
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    return lang == 'es'

# Descargando las stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = normalize(set(stopwords.words('spanish')))

reviews_df = pd.read_csv('./data/MovieReviews.csv', sep = ',')
reviews_df.set_index(reviews_df.columns[0],inplace=True)

reviews_df.sentimiento.value_counts(dropna = False, normalize = True)
X_train, X_test, y_train, y_test = train_test_split(reviews_df.review_es, reviews_df.sentimiento, test_size = 0.2, stratify = reviews_df.sentimiento, random_state = 1)

tfidf = TfidfVectorizer(tokenizer = tokenizer, stop_words = stop_words, lowercase = True)
X_tfidf = tfidf.fit_transform(X_train)
clf_tfidf = svm.SVC(kernel='linear')
clf_tfidf.fit(X_tfidf, y_train)


p1 = Pipeline([('vectorizer', tfidf ),('svm', clf_tfidf)])
