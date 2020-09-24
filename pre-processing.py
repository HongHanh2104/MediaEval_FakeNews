import pandas as pd 
import numpy as np 
from collections import Counter
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.corpus import stopwords
#nltk.download
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import TweetTokenizer

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='Path to the Dataset')
parser.add_argument('--file', help='Data file to pre-process')
args = parser.parse_args()

root_path = args.root
filename = args.file

# Read file that need to pre-process
tweets = pd.read_csv(os.path.join(root_path, filename))

# Create hashtag column
tweets['hashtag'] = tweets['Text'].apply(lambda x: re.findall(r"#(\w+)", x))

def get_url_patern():
    return re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'
        r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'
    )

def get_hashtags_pattern():
    return re.compile(r'#\w*')

def get_emojis_pattern():
    try:
        # UCS-4
        emojis_pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
        # UCS-2
        emojis_pattern = re.compile(
            u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return emojis_pattern

def get_mentions_pattern():
    return re.compile(r'@\w*')

class TwitterPreprocessor:
    def __init__(self, text):
        super().__init__()
        self.text = text
   
    def remove_urls(self):
        self.text = re.sub(pattern=get_url_patern(), repl='', string=self.text)
        return self

    def remove_hashtags(self):
        self.text = re.sub(pattern=get_hashtags_pattern(), repl='', string=self.text)
        return self

    def remove_emojis(self):
        self.text = re.sub(pattern=get_emojis_pattern(), repl='', string=self.text)
        return self

    def remove_mentions(self):
        self.text = re.sub(pattern=get_mentions_pattern(), repl='', string=self.text)
        return self

    def remove_punctuation(self):
        self.text = self.text.translate(str.maketrans('', '', string.punctuation))
        return self

    def lowercase(self):
        self.text = self.text.lower()
        return self

    def remove_stopwords(self, extra_stopwords=None):
        if extra_stopwords is None:
            extra_stopwords = []
        text = nltk.word_tokenize(self.text)
        stop_words = set(stopwords.words('english'))
        new_sentence = []
        for w in text:
            if w not in stop_words and w not in extra_stopwords:
                new_sentence.append(w)
        self.text = ' '.join(new_sentence)
        return self

def preprocess_without_stopword(data):
    texts = [(TwitterPreprocessor(t).lowercase().remove_urls().remove_hashtags().remove_emojis().remove_mentions().remove_punctuation().text) \
         for t in data]
    return pd.DataFrame(texts)

def preprocess_with_stopword(data):
    texts = [(TwitterPreprocessor(t).lowercase().remove_urls().remove_hashtags().remove_emojis().remove_mentions().remove_punctuation().remove_stopwords().text) \
         for t in data]
    return pd.DataFrame(texts)

cleaned_text = preprocess_without_stopword(tweets['Text'])
unstop_text = preprocess_with_stopword(tweets['Text'])

tweets['Cleaned_Text'] = cleaned_text
tweets['Unstop_Text'] = unstop_text
tweets.to_csv(f'{root_path}/cleaned_{filename}', index=False)