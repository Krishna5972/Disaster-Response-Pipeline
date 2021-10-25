import json
import plotly
import pandas as pd
from os import environ

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import re

app = Flask(__name__) 

class Keywords(BaseEstimator, TransformerMixin):

    def key_words(self, text):
        """
        INPUT: text - string, raw text data
        OUTPUT: count of words present in the text
        """
        # list of words 
        words = ['food','hunger','hungry','starving','water','drink','eat','thirsty',
                 'need','shortage']
        lemmatizer = WordNetLemmatizer()
        stemmer=PorterStemmer()
        # lemmatize the words
        lemmatized_words = [lemmatizer.lemmatize(w).strip() for w in words]
        # stem the words
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #request vs offer
    request_offer_df=df[['request','offer']]
    request_offer_df=request_offer_df.apply(pd.to_numeric, errors='ignore')
    request_count=request_offer_df['request'].value_counts()[1]
    offer_count=request_offer_df['offer'].value_counts()[1]
    print(request_count)
    print(offer_count)

    
    request_offer_counts=[request_count,offer_count]
    request_offer_names=['Request','Offer']

    #relateed v unrelated
    related_df=df[['related']]
    related_df=related_df.apply(pd.to_numeric, errors='ignore')
    related_counts=related_df['related'].value_counts()[1]
    unrelated_counts=related_df['related'].value_counts()[0]
    related_counts_list=[related_counts,unrelated_counts]
    related_names=['related','not related']







    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            #graph 1
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'color':'red',
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                
            }
        },
        #graph 2
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts_list,

                )
            ],

            'layout': {
                'title': 'Distribution of Message Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Type"
                }
            }
        },
        #graph 3
        {
            'data': [
                Bar(
                    x=request_offer_names,
                    y=request_offer_counts,

                )
            ],

            'layout': {
                'title': 'Distribution of Related and Unrelated messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Type"
                }
            }


        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)


if __name__ == '__main__':
    main()