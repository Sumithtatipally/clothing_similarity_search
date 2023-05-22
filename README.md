# clothing_similarity_search
The goal of this project is to create a machine learning model capable of receiving text describing a clothing item and returning a ranked list of links to similar items from different websites. Your solution must be a function deployed on Google Cloud that accepts a text string and returns JSON responses with ranked suggestions.

This repository contains a Flask application that provides clothing recommendations based on search keywords using cosine similarity with BERT embeddings. The application is deployed on Google Cloud Functions.

## Prerequisites

- Python 3.10
- Google Cloud SDK
- Google Cloud project ID

## Installation

1. Clone the repository:
 ```shell
   git clone https://github.com/sumithtatipally/clothing_similarity_search.git
 ```
2. Navigate to the project directory:
 ```shell
  cd clothing_similarity_search
  ```
3. Set up a virtual environment and activate it:
  ```shell
  python3 -m venv venv
` source venv/bin/activate
  ```
4. Install required packages
  ```shell
  pip install -r requirements.txt
  ```
5. To run it locally
  ```shell
  python main.py
  ```
## Deployment to Google Cloud Functions
1. Create a new project in the Google Cloud Console. Note down the project ID.

2. Enable the Cloud Functions API for your project. In the Cloud Console, navigate to APIs & Services > Library and search for "Cloud Functions".

3. Set the Google Cloud project ID
  gcloud config set project your-project-id
  
4. Deploy the code to Google Cloud Functions:
  gcloud functions deploy myntra-recommendations --runtime python310 --trigger-http --allow-unauthenticated --entry-point get_recommendations


## Usage
To get clothing recommendations, send a POST request to the Cloud Function's URL with the following JSON payload:

json
Copy code
{
  "search_keyword": "your_search_keyword",
  "top_n": 5
}
Replace your_search_keyword with the desired keyword for product search and 5 with the desired number of results to return.


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
