import sys
import time

from sqlalchemy import create_engine
import pandas as pd 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
from app1.tokenizerrr import tokenize,Keywords


def load_data(database_filepath):
    """
    Loads the data from sql table and splits the data into X and y datasets
    Args:
    database_filepath: path of the database
    Return: 
    X:features dataset with message column
    y:dependent dataset 
    category_names: classifiable categories
    """
    engine = create_engine('sqlite:///'+database_filepath)
    database_filepath=str(database_filepath)
    tablename=database_filepath.split('.')[0]
    df = pd.read_sql_table(tablename,engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X,y,category_names




def build_model():
    """
    Builds the model and performs GridSearchCV to find the best parameters.
    Args:none
    Return: 
    model: model
    
    """
    kneighbor_model=Pipeline([
    ('union',FeatureUnion([
        ('text_piprline',Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfid',TfidfTransformer())
            
        ])),
        ('countkeywords',Keywords())
        
    ])),
    ('kn',MultiOutputClassifier(KNeighborsClassifier(n_neighbors=10)))


                         ])

    #parameters = {'kn__estimator__n_neighbors':[7,10,12]}

    
    #cv= GridSearchCV(kneighbor_model,param_grid=parameters, cv=3, verbose=3)

    #return cv  #remove below return statement to peform GridSearchCV
    return kneighbor_model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
   
    print(classification_report(Y_test.values, y_pred, target_names=category_names))
    accuracy=(y_pred==Y_test.values).mean()
    print('The model accuracy score is {:.4f}'.format(accuracy))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        start_time = time.time()
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print("--- time taken %s seconds ---" % ((time.time() - start_time)))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()