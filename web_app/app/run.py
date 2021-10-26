import json
import plotly
import pandas as pd
from os import environ
from app import app
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
from app.tokenizerrr import tokenize
from app.tokenizerrr import Keywords

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')

df = pd.read_sql_table('DisasterResponse', engine)

model = joblib.load("classifier.pkl")


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

