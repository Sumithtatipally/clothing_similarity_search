from bs4 import BeautifulSoup
import requests
import re
import time
import pandas as pd
from requests_html import HTMLSession, AsyncHTMLSession

# Set the base URL and the URL of the page to scrape
baseurl = "https://www.shoppersstop.com"
url = "https://www.shoppersstop.com/women-westernwear/c-A2060"

# Set the headers for the HTTP request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'
}

# Send a GET request to the URL and parse the HTML content using BeautifulSoup
source = requests.get(url, headers=headers)
soup = BeautifulSoup(source.content, 'lxml')

# Find all product links in the HTML content
links = soup.find_all('li', attrs={'class': 'pro-box'})

print(links)

# Initialize an empty list to store the product links
product_links = []

# Iterate over multiple pages to collect product links
for page in range(1, 25):
    source = requests.get(f'https://www.shoppersstop.com/women-westernwear/c-A2060?page={page}', headers=headers)
    soup = BeautifulSoup(source.content, 'lxml')
    productlist = soup.find_all('li', {'class': 'pro-box'})

    for item in productlist:
        a_tag = item.find('a', {'href': 'javascript:void(0)'})
        if a_tag:
            input_element = a_tag.find('input', {'id': 'pdpurl'})
            if input_element:
                product_links.append(input_element['value'])

print(len(product_links))

# Initialize an empty list to store product information
products = []

# Iterate over each product link and scrape product details
for index, link in enumerate(product_links, 1):
    full_url = baseurl + link
    r = requests.get(full_url, headers=headers)
    soup = BeautifulSoup(r.content, 'lxml')

    try:
        # Extract the brand, description, and classification information from the HTML content
        brand = soup.find("div", class_='pdp-pname block-ellipsis').text
        desc = soup.find("li", class_='product_inner_content').text.strip().replace('\n', '')
        classification = soup.find("li", class_='classification_details').text.strip()
        classification = classification.replace('\t', '').replace('\n', ' ').strip()
        classification = ' '.join(classification.split())
    except:
        None

    # Create a dictionary of product information
    items = {"name": brand, "desc": desc, "class": classification, "link": full_url}
    products.append(items)

    if index % 50 == 0:
        print(f"Processed {index} links")

print("Finished scraping all links.")

# Create a DataFrame from the collected product information
df = pd.DataFrame(products)

# Define the list of words to remove from the description and classification
words_to_remove = ['Product Description',"Country of Origin","Sleeves","Fit","Pattern","Fabric","Pack Of","Length",
                  "Occasion","Product Type","Knit/Woven","Gender"]
df['desc'] = df['desc'].str.replace('|'.join(words_to_remove), '', regex=True)
df['class'] = df['class'].str.replace('|'.join(words_to_remove), '', regex=True)

df.to_csv("datawomen_western.csv")

