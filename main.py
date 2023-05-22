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

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    stop = stopwords.words('english')
    lem = WordNetLemmatizer()
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [lem.lemmatize(w) for w in words if w not in stop]

def concatenate_text_columns(df, column_names, new_column_name):
    df[new_column_name] = df[column_names].apply(lambda x: ' '.join(x), axis=1)
    return df

def apply_preprocessing(df, text_column):
    df['preprocessed_text'] = df[text_column].apply(preprocess_text)
    df['preprocessed_text'] = df['preprocessed_text'].apply(lambda x: ' '.join(x))
    return df


app = Flask(__name__)

# Load the data
df = pd.read_csv('myntra_products_catalog.csv')


# Load the pre-calculated embeddings
embeddings = np.load('embeddings.npy')

@app.route('/hi')
def hello_world():
    return 'Hello, World!'

@app.route('/api', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    print(data)
    search_keyword = data['search_keyword']
    top_n = data['top_n']  # Number of results to return
    print(search_keyword, top_n)
    # Perform clothing similarity search
    results = cloth_similarity(df, search_keyword)

    # Get the top N results
    top_results = results.head(top_n)

    # Convert the top results to JSON
    top_results_json = top_results.to_json(orient='records')

    return jsonify(top_results_json)

def cloth_similarity(df, search_keyword, text_column_names=["Product_Description", "Product_Name"]):
    df = pd.read_csv('myntra_products_catalog.csv')  # Load the data
    df["Product_Description"] = df["Description"].apply(str)
    df["Product_Name"] = df["ProductName"].apply(str)
    df = concatenate_text_columns(df, text_column_names, "Full_Des")
    df = apply_preprocessing(df, "Full_Des")
    print('1')
    # Calculate BERT embeddings
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    # embeddings = model.encode(df['preprocessed_text'])
    print('2')
    # Calculate cosine similarities
    search_keyword_embedding = model.encode([search_keyword])[0].reshape(1, -1)
    cosine_similarities = cosine_similarity(embeddings, search_keyword_embedding)

    df['bert_cosine_similarity'] = cosine_similarities.flatten()
    df_sorted = df.sort_values('bert_cosine_similarity', ascending=False)
    return df_sorted

if __name__ == '__main__':
    app.run(port=5500, debug=True)
