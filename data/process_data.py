import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the two data sets and to merge them into a Dataframe

    Args:
    messages_filepath: path of the messages dataset
    categories_filepath : path of the categories dataset

    Returns:merged df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='outer',on='id')
    return df
    

def clean_data(df):
    """
    Cleans the Dataframe by spliting the catefories columns into column labels.0 or 1 is assigned 
    depending on the category and duplicate values are dropped.

    Args:
    df: merged Dataframe containing messages and categories
    
    Returns:df with categorial column labels 
    """
    categories = df['categories'].str.split(';',expand=True)
    col_name=[]
    for i in range(0,len(categories.columns)):
        col_name.append(categories[i][0])
    category_colnames =list(map(lambda x: x.split('-')[0], col_name))  
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
    
        categories[column] = categories[column].astype(str).str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        #converting values other than 0 or 1 to 0
    for columns in categories:
        categories[(categories[columns]!=0) & (categories[columns]!=1)]=0

    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    df.drop_duplicates(inplace=True)
    return df



def save_data(df, database_filename):
    """
    Saves the cleaned dataframe into a table.

    Args:
    df: merged Dataframe containing messages and categories
    database_filename: name of the database

    Return: none
    """
    engine = create_engine('sqlite:///'+database_filename)
    database_filename=str(database_filename)
    tablename=database_filename.split('/')[1]
    tablename=tablename.split('.')[0]
    print("The name of the table:"+" "+tablename)
    df.to_sql(tablename, engine, index=False , if_exists='replace')
    df.to_csv('Disasterresponse.csv')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()