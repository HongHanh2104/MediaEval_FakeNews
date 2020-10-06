import pandas as pd 
import numpy as np 
from collections import Counter
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.corpus import stopwords
#nltk.download
#nltk.download('words')
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

MIN_YEAR = 1900
MAX_YEAR = 2100

def get_url_patern():
    return re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'
        r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'
    )

def get_hashtags_pattern():
    return re.compile(r'#\w*')

def get_emojis_pattern():
    emojis_pattern = re.compile(pattern = '['
        u'\U0001F600-\U0001F64F'  
        u'\U0001F300-\U0001F5FF' 
        u'\U0001F680-\U0001F6FF'  
        u'\U0001F1E0-\U0001F1FF'  
        u'\U00002500-\U00002BEF'  
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u2640-\u2642'
        u'\u2600-\u2B55'
        u'\u200d'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\ufe0f'  
        u'\u3030'
                           ']+', flags=re.UNICODE)
    return emojis_pattern

def get_mentions_pattern():
    return re.compile(r'@\w*')

def get_blank_spaces_pattern():
    return re.compile(r'\s{2,}|\t')

def is_year(text):
    if (len(text) == 3 or len(text) == 4) and (MIN_YEAR < len(text) < MAX_YEAR):
        return True
    else:
        return False

class TwitterPreprocessor:
    def __init__(self, text):
        super().__init__()
        self.text = text
   
    def remove_urls(self):
        self.text = re.sub(pattern=get_url_patern(), repl='', string=self.text)
        return self

    def remove_hashtags(self):
        special_text = ['5G', '5g', 'virus', 'coronavirus', 'corona', 'conspiracy', 'COVID19']
        text = re.findall(r'#(\w*)', self.text)
        if any(t in text for t in special_text):
            for t in text:
                self.text = re.sub(pattern=re.compile(rf'#{t}'), repl=f'{t}', string=self.text) if t in special_text \
                else re.sub(pattern=re.compile(rf'#{t}'), repl='', string=self.text)

        else:
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

    def remove_blank_spaces(self):
        self.text = re.sub(pattern=get_blank_spaces_pattern(), repl=' ', string=self.text)
        return self

    def lowercase(self):
        self.text = self.text.lower()
        return self

    def remove_numbers(self, preserve_years=False):
        text_list = self.text.split()
        new_sentence = []
        for text in text_list:
            if text.isnumeric():
                if preserve_years:
                    if is_year(text):
                        new_sentence.append(text)
            else:
                new_sentence.append(text)

        self.text = ' '.join(new_sentence)
        return self
    
    def remove_nonEnglish(self, extra_nonEnglish=None):
        if extra_nonEnglish is None:
            extra_nonEnglish = []
        text = nltk.word_tokenize(self.text)
        english_words = set(nltk.corpus.words.words())
        new_sentence = []
        for w in text:
            if w not in english_words and w not in extra_nonEnglish:
                new_sentence.append(w)
        self.text = ' '.join(new_sentence)
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
    
    def add_white_space(self):
        self.text = re.sub(pattern=r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", repl=r"\1 ", string=self.text)
        return self

# def preprocess_without_stopword(data):
#     texts = [(TwitterPreprocessor(t).lowercase().remove_urls().remove_hashtags().remove_emojis().remove_mentions().remove_punctuation().remove_blank_spaces().add_white_space().text) \
#          for t in data]
#     return pd.DataFrame(texts)

def preprocess_with_stopword(data):
    texts = [(TwitterPreprocessor(t).lowercase().remove_urls().remove_emojis().remove_mentions().remove_blank_spaces().text) \
         for t in data]
    return pd.DataFrame(texts)

#cleaned_text = preprocess_without_stopword(tweets['Text'])
cleaned_text = preprocess_with_stopword(tweets['Text'])

tweets['Cleaned_Text'] = cleaned_text
tweets.to_csv(f'{root_path}/cleaned_{filename}', index=False)
print('Complete pre-precessing data!'.upper())
