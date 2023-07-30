import datetime
from typing import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


def check_time(time_path):
    r'''
    time_path : txt file path for check the update time
    
    -: PURPOSE OF FUNCTION :-
    This function check the time ,If it cross 1 Day alert you'''
    time = open(time_path ,'r+')
    ct = list(map(int ,time.read().split('\n')[-1].split(', ')))

    if (datetime.datetime(ct[0] ,ct[1] ,ct[2] ,ct[3] ,ct[4] ,ct[5] ,ct[6]) - datetime.datetime.now()).days == 1:
        return 'Re_Train'
    if (datetime.datetime(ct[0] ,ct[1] ,ct[2] ,ct[3] ,ct[4] ,ct[5] ,ct[6]) - datetime.datetime.now()).days == 0 :
        return 'not_necessay'
    
def update_time(time_path):
        r'''
         -: PURPOSE OF FUNCTION
         Update the current time in the txt file '''
        update = open(time_path ,'a+')
        update.write('\n')
        update.write(datetime.datetime.now())

def preprocess_text(text):
    
    # Download the 'punkt' resource if not already downloaded

    text = re.sub(r'[^a-zA-Z\s]' , '  ', text)
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '  ', text)
    text = text.replace('\n', '  ')
    text = text.replace('\r', '  ')
    text = text.replace('\t', '  ')

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '  ', text)

    # Remove unwanted spaces
    text = re.sub(r"\s+" ," ", text)

    # Tokenize text into words
    words = word_tokenize(text)

    # Remove stopwords and Lemmatize words
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)     