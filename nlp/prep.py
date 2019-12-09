import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd 

import acquire

def lower_string(string):
    return string.lower()

def normalize_str(string):
    return unicodedata.normalize('NFKD', string)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')

def remove_special_character(string):
    return re.sub(r"[^a-z0-9'\s]", '', string)

def basic_clean(string):
    string = lower_string(string)
    string = normalize_str(string)
    string = remove_special_character(string)
    return string

def tokenize(string):
    tokeninzer = nltk.tokenize.ToktokTokenizer()
    return tokeninzer.tokenize(string, return_str=True)

def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lemmitize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    article_lemmatized = ' '.join(lemmas)
    return article_lemmatized

def remove_stopwords(string, extra_words = [], exclude_words = []):
    if type(extra_words) != list:
        extra_words = extra_words.split()
    
    if type(exclude_words) != list:
        exclude_words = exclude_words.split()

    stopword_list = stopwords.words('english')

    for word in exclude_words:
        stopword_list.remove(word)
    
    for word in extra_words:
        stopword_list.append(word)
    
    words_list = str.split()

    filtered_words = [word for word in words_list if word not in stopword_list]

    article_without_stopwords = ' '.join(filtered_words)

    return article_without_stopwords

def prep_article(dictionary):
    p_article = {}
    p_article['title'] = dictionary['title']
    p_article['original'] = dictionary['body']
    p_article['stemmed'] = stem(dictionary['body'])
    p_article['lemmatized'] = lemmitize(dictionary['body'])
    p_articl['clean'] = remove_stopwords(dictionary['body'])
    return p_article

def prepare_article_data(list_of_dictionaries):
    transformed_articles = []
    for x in list_of_dictionaries:
        transformed_articles.append(prep_article(x))
        return transformed_articles
