import pandas as pd 
import numpy as np 
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
#nltk.download
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
import preprocessor as p

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

# Using preprocessor to remove URLs, Mentions, ... in text
for i, t in enumerate(tweets['Text']):
    tweets.loc[i, 'Cleaned_Text'] = p.clean(t)

# Create tokenization on cleaned text
def preprocess_data(data):
    # Remove numbers
    data = data.astype(str).str.replace('\d+', '')
    lower_text = data.str.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer = TweetTokenizer()

    def lemmatize_text(text):
        return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((text))]

    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', (word))
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    words = lower_text.apply(lemmatize_text)
    words = words.apply(remove_punctuation)
    return pd.DataFrame(words)

pre_tweets = preprocess_data(tweets['Text'])
tweets['Cleaned_Text'] = pre_tweets

# Remove stopwords
stop_words = set(stopwords.words('english'))
tweets['Unstop_Text'] = tweets['Cleaned_Text'].apply(lambda x: [item for item in \
                                    x if item not in stop_words])


#print(tweets['Cleaned_Text'][0])
tweets.to_csv(f'{root_path}/cleaned_{filename}.csv', index=False)