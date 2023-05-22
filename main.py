

from flask import Flask, jsonify, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import unicodedata
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocesses the given text by removing stopwords, lemmatizing, and normalizing the characters.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        list: A list of preprocessed words.
    """
    stop = stopwords.words('english')
    lem = WordNetLemmatizer()
    
    # Normalize characters to remove diacritical marks
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    
    # Remove punctuation and split into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    # Lemmatize and remove stopwords
    return [lem.lemmatize(w) for w in words if w not in stop]

def concatenate_text_columns(df, column_names, new_column_name):
    """
    Concatenates multiple text columns in a DataFrame into a single column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text columns.
        column_names (list): A list of column names to be concatenated.
        new_column_name (str): The name of the new column.

    Returns:
        pandas.DataFrame: The DataFrame with the new concatenated column.
    """
    df[new_column_name] = df[column_names].apply(lambda x: ' '.join(x), axis=1)
    return df

def apply_preprocessing(df, text_column):
    """
    Applies the text preprocessing function to a specific column in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text column.
        text_column (str): The name of the text column.

    Returns:
        pandas.DataFrame: The DataFrame with the preprocessed text column.
    """
    df['preprocessed_text'] = df[text_column].apply(preprocess_text)
    df['preprocessed_text'] = df['preprocessed_text'].apply(lambda x: ' '.join(x))
    return df


app = Flask(__name__)

# Load the data
df = pd.read_csv('shoppers_stop (1).csv')


# Load the pre-calculated embeddings
embeddings = np.load('ss_embeddings.npy')


@app.route('/', methods=['POST'])
def get_recommendations():
    """
    Flask route for handling POST requests to get clothing recommendations.

    Returns:
        flask.Response: The JSON response containing the top clothing recommendations.
    """
    data = request.get_json()
    print(data)
    search_keyword = data['search_keyword']
    top_n = 5  # Number of results to return
    print(search_keyword, top_n)
    
    # Perform clothing similarity search
    results = cloth_similarity(df, search_keyword)

    # Get the top N results
    top_results = results.head(top_n)

    df_ranked_list = top_results[['name', 'link', 'bert_cosine_similarity']]

    # Convert the top results to JSON
    top_results_json = df_ranked_list.to_json(orient='records')

    return jsonify(top_results_json)


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def cloth_similarity(df, search_keyword, text_column_names=["name", "desc", "class"]):
    """
    Calculates the similarity between a search keyword and textual columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the clothing data.
        search_keyword (str): The keyword to search for similarity.
        text_column_names (list): List of column names containing textual information. Default is ["name", "desc", "class"].

    Returns:
        pandas.DataFrame: The DataFrame sorted by cosine similarity in descending order.
    """
    # Read the DataFrame
    df = pd.read_csv('shoppers_stop (1).csv')

    # Drop the 'Unnamed: 0' column if present
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # Create new columns for preprocessing
    df["Product_Description"] = df["desc"].apply(str)
    df["Product_Name"] = df["name"].apply(str)
    df["Product_Class"] = df["class"].apply(str)

    # Concatenate specified text columns into a new column 'Full_Des'
    df = concatenate_text_columns(df, text_column_names, "Full_Des")

    # Apply preprocessing to 'Full_Des'
    df = apply_preprocessing(df, "Full_Des")

    print('1')

    # Calculate BERT embeddings
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    # embeddings = model.encode(df['preprocessed_text'])
    
    print('2')

    # Calculate cosine similarities
    search_keyword_embedding = model.encode([search_keyword])[0].reshape(1, -1)
    cosine_similarities = cosine_similarity(embeddings, search_keyword_embedding)

    # Add cosine similarities as a new column in the DataFrame
    df['bert_cosine_similarity'] = cosine_similarities.flatten()

    # Sort the DataFrame by cosine similarity in descending order
    df_sorted = df.sort_values('bert_cosine_similarity', ascending=False)

    return df_sorted


if __name__ == '__main__':
    # Start the application on port 5500
    app.run(port=5500)
