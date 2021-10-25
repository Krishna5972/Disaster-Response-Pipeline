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
import pickle






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
        INPUT: text - string, raw text data
        OUTPUT: bool -bool object, True or False
        """
        # list of words that are commonly used during a disaster event
        words = ['food','hunger','hungry','starving','water','drink','eat','thirsty',
                 'need','shortage']

        # lemmatize the buzzwords
        lemmatized_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
        # Get the stem words of each word in lemmatized_words
        words = [PorterStemmer().stem(w) for w in lemmatized_words]
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

    # parameters = {'kn__estimator__n_neighbors':[7,10,12]}

    # cv= GridSearchCV(kneighbor_model,param_grid=parameters, cv=3, verbose=3)

    # return cv  #remove below return statement to peform GridSearchCV
    return kneighbor_model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    y_singletest=model.predict(["We're asking for water, medical supply, food"])[0]
    print(y_singletest)
    #print(classification_report(Y_test.values, y_pred, target_names=category_names))
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
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        start_time = time.time()
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