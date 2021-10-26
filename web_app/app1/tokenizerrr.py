import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



def tokenize(text):
    """
    Replaces the URL's in the text with a custom word 'urlplaceholder'.Tokenize the text,then Normalize,Lemmatize and Stem the tokens.
    Args:
    text: messages from the message data set
    Return: 
    clean_tokens:clean tokens after stemming
    """
    #Replacing URL's with urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    
    # Excluding everything except letters and numbers
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    text=text.lower()
    # tokenize the text
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    stemmer=PorterStemmer()
    tokens=[lemmatizer.lemmatize(token).strip() for token in tokens]
    
    clean_tokens = []
    for token in tokens:
        clean_token = stemmer.stem(token)
        clean_tokens.append(clean_token)
        
    return clean_tokens
class Keywords(BaseEstimator, TransformerMixin):

    def key_words(self, text):
        """
        Args:
         text -  text data
        Returns:
        count of common words during a disaster
        """
        # list of common words
        words = ['food','hunger','hungry','starving','water','drink','eat','thirsty','need']

        lemmatizer = WordNetLemmatizer()
        stemmer=PorterStemmer()
        # lemmatize the words
        lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
        # Get the stem words of each word in lemmatized_words
        words = [stemmer.stem(w) for w in lemmatized_words]
        count=0

        # tokenize the input text
        clean_tokens = tokenize(text)
        for token in clean_tokens:
            if token in words:
                count=count+1
        return count

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X_key_words = pd.Series(X).apply(self.key_words)
        return pd.DataFrame(X_key_words)