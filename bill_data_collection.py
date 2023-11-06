# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 08:13:22 2023

@author: andrewallen42
"""

import requests
from bs4 import BeautifulSoup
import re 
import pandas as pd
from transformers import AutoTokenizer
import os
import openai
import time
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import seaborn as sns
import matplotlib.pyplot as plt

def remove_html_tags(text):
    """
    Remove HTML tags from the input text.
    
    Args:
        text (str): The text containing HTML tags.
        
    Returns:
        str: Cleaned text with HTML tags removed.
    """
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text

def preprocess_bill_htm(bill_text):
    """
    Preprocess bill text from its native HTML scraped format to a more GPT-readable format.
    
    Args:
        bill_text (str): The bill text in its original HTML format.
        
    Returns:
        str: Preprocessed bill text for GPT readability.
    """
    try:
        bill_text = re.sub(r'\n+', ' ', bill_text)
        
        # Extract everything after the last "A BILL To"
        parts = bill_text.rsplit('A BILL', 1)
        parts = bill_text.rsplit('_____________', 1)
        bill_text = parts[1]
        bill_text = re.sub(r'\s{2,}', ' ', bill_text)
        bill_text = re.sub(r'<[^>]+>', '', bill_text)
    except:
        bill_text = None
        
    return bill_text

def convert_congress_url(input_url):
    """
    Convert a congress.gov API URL to a more readable bill URL.
    
    Args:
        input_url (str): The original congress.gov API URL.
        
    Returns:
        str: A more readable bill URL.
    """
    # Use regular expressions to extract the relevant parts of the input URL
    match = re.search(r'/bill/(\d+)/(\w+)/(\d+)\?format=json', input_url)
    if match:
        congress_number, bill_type, bill_number = match.groups()
        
        # Convert the Congress number to the "Nth Congress" format
        congress_label = f"{congress_number}th-congress"
        
        # Map bill types to their corresponding values
        bill_type_mapping = {
            "hr": "house-bill",
            "s": "senate-bill",
            # Add more mappings as needed
        }
        
        # Construct the new URL
        new_url = f'https://www.congress.gov/bill/{congress_label}/{bill_type_mapping.get(bill_type, bill_type)}/{bill_number}/text'
        return new_url
    else:
        return None

# Define your API keys
data_gov_key = #[INSERT_KEY]  # Replace [INSERT KEY] with your actual congress.gov API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API Key

# Data Collection

# Set the endpoint for getting a list of bills
list_of_bills_endpoint = f'https://api.congress.gov/v3/bill/117/hr?api_key={data_gov_key}'

# Define query parameters
query_params = {
    'fromDateTime': '2023-02-01T04:03:00Z',
    'toDateTime': '2023-02-27T04:03:00Z',
    'sort': 'updateDate+desc',
    'limit': 200
}

# Get a list of bills in response
list_of_bills_response = requests.get(list_of_bills_endpoint, params=query_params)
list_of_bills = list_of_bills_response.json()['bills']
bill_urls = [convert_congress_url(x['url']) for x in list_of_bills]
bill_update_dates = [x['updateDate'] for x in list_of_bills]
bill_nums = [x['number'] for x in list_of_bills]
bill_titles = [x['title'] for x in list_of_bills]

# Create dictionaries to store bill information
num_title_dict = dict(zip(bill_nums, bill_titles))
date_dict = dict(zip(bill_nums, bill_update_dates))
url_dict = dict(zip(bill_nums, bill_urls))

# Get the FIRST summary for each bill
summaries = {}
for bill in bill_nums:
    try:
        summaries_endpoint = f'https://api.congress.gov/v3/bill/117/hr/{bill}/summaries?api_key={data_gov_key}'
        summary_obj = requests.get(summaries_endpoint)
        summaries[bill] = summary_obj.json()['summaries'][0]['text']
    except:
        summaries[bill] = None

# Get the HTML link of each bill's text (if possible)
text = {}
for bill in bill_nums:
    try:
        text_endpoint = f'https://api.congress.gov/v3/bill/117/hr/{bill}/text?api_key={data_gov_key}'
        text_obj = requests.get(text_endpoint)
        textversions = text_obj.json()['textVersions']
        for item in textversions:
            for format_item in item['formats']:
                url = format_item['url']
                if url.endswith(".htm"):
                    text[bill] = url
    except:
        text[bill] = None

# Parse the bill body from the links
bill_text_dict = {}
for bill_num, url in text.items():
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            bill_text = soup.get_text()
            bill_text = preprocess_bill_htm(bill_text)
            bill_text_dict[bill_num] = bill_text
    except:
        bill_text_dict[bill_num] = None

# Create DataFrames for summaries and bill text
filtered_dict_summaries = {key: value for key, value in summaries.items() if key in bill_text_dict}
filtered_dict_summaries = {key: remove_html_tags(value) for key, value in filtered_dict_summaries.items()}
df1 = pd.DataFrame(list(filtered_dict_summaries.items()), columns=['bill_num', 'bill_summary'])
df2 = pd.DataFrame(list(bill_text_dict.items()), columns=['bill_num', 'bill_text'])

# Merge the two DataFrames using a common key
bill_df = pd.merge(df1, df2, on='bill_num')
bill_df = bill_df[bill_df.notna().all(axis=1)]

# Tokenization and Filtering
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

bill_df['TokenCount'] = bill_df['bill_text'].apply(lambda x: len(x.split()))
bill_df['TokenCountSummary'] = bill_df['bill_summary'].apply(lambda x: len(x.split()))

bill_df = bill_df[bill_df['TokenCount'] <= 7500]

bill_df['title'] = [num_title_dict[str(x)] for x in bill_df['bill_num']]
bill_df['update_date'] = [date_dict[str(x)] for x in bill_df['bill_num']]

# Generate GPT-4 summaries for bills
created = []
gpt_summaries = []

for i, bill_text_to_summarize in enumerate(bill_df['bill_text']):
    if i <= 35:
        continue
    print(f"{i} out of {len(bill_df['bill_text'])}")
    summary_length = bill_df['TokenCountSummary'].iloc[i]
    system_desc = '''You are a government relations professional, tasked with summarizing bills from the U.S. House of Representatives. You specialize in creating summaries that are easy to understand for those who are college-educated but do not know much about specific legislative procedures or artifacts.'''
    user_desc = f''' According to your expertise, create a summary of the following bill that is roughly {summary_length} words long.
    
        As you do, do the following:
        1) Incorporate main ideas and essential information, eliminating extraneous language.
        
        2) Rely strictly on the provided text, without including external information.
        
        Bill to summarize:
        {bill_text_to_summarize}
        '''
    completion = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {"role": "system", "content": system_desc},
        {"role": "user", "content": user_desc}
      ]
    )
    output = completion.choices[0].message['content']
    gpt_summaries.append(output)
    time.sleep(20)

# Insert a placeholder at a specific index
index_to_insert = 37  # 0-based index
value_to_insert = None
gpt_summaries.insert(index_to_insert, value_to_insert)

# Combine the GPT summaries with the DataFrame and filter
bill_df_filtered = bill_df.iloc[0:60]
bill_df_filtered['bill_summary_gpt'] = gpt_summaries
bill_df_filtered = bill_df_filtered.dropna()
bill_df_filtered['title'] = [num_title_dict[str(x)] for x in bill_df_filtered['bill_num']]
bill_df_filtered['url'] = [url_dict[str(x)] for x in bill_df_filtered['bill_num']]

# Calculate ROUGE scores
score_list = []

for i, reference_summary in enumerate(bill_df_filtered['bill_summary']):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    candidate_summary = bill_df_filtered['bill_summary_gpt'].iloc[i]
    scores = scorer.score(reference_summary, candidate_summary)
    score_list.append(scores)

bill_df_filtered['rouge1_recall'] = [x['rouge1'][1] for x in score_list]



