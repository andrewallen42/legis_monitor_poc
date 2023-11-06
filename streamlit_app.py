# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:40:14 2023

@author: andrewallen42
"""

import streamlit as st
import pandas as pd
import time
import hugchat
from hugchat.login import Login
from hugchat import hugchat
import datetime as dt

# Read in data
bills_filepath = 'bill_df_w_gpt_summaries.csv'
df = pd.read_csv(bills_filepath)

# Sidebar - User's interests
st.sidebar.title("Client's Interests")
client_input = st.sidebar.text_area("What are you looking for?", "I am interested in legislation related to...")

# Create a ChatBot
email = st.secrets["db_username"]
passwd = st.secrets["db_password"]

# Convert the "update_date" column to datetime
df['update_date'] = pd.to_datetime(df['update_date'])

# Initialize variables
responses = []

# Streamlit app
st.title("Legislative Monitoring App")
st.write("Welcome to the legislative monitoring app. We help you find bills related to the client's interests.")

# Open a dialog box to provide initial recommendations
with st.expander("Get Started"):
    st.markdown("### Welcome to the Legislative Monitoring App")
    st.markdown("**Overview:**")
    st.write("This web app is a proof-of-concept tool designed to help clients track legislation relevant to their personal or commercial interests. It provides access to bills updated in the House of Representatives in Feb 2023, retrieved using the Congress.gov API and summarized with GPT-4. This app can be easily adapted for other types of regular government publications.")
    
    st.markdown("**Data Source:**")
    st.write("The legislation data used in this app is sourced from the House of Representatives updates in February 2023.")

    st.markdown("**How It Works:**")
    st.write("The app leverages HuggingFace's HuggingChat, powered by LLaMa, to help you analyze and identify relevant bills.")
    
    st.markdown("### Getting Started:")
    st.write("To start, try searching for 'legislation related to airports and air commerce' from the 22nd to the 24th using the date range selection on the sidebar.")
    st.write("Click the 'Run' button to process the bills and review the results.")
    st.write("The current proof-of-concept is limited to 10 bills - if the tool is running into rate limit issues, try smaller date ranges.")


# Sidebar - Date Range Selection
st.sidebar.write("Select the date range")

# Get the min and max dates from the DataFrame
min_date = dt.date(2023, 2, 1)
max_date = dt.date(2023, 2, 28)

start_date = dt.date(2023, 2, 4)
end_date = dt.date(2023, 2, 4)

start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date, key="start_date")
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date, key="end_date")

if start_date > end_date:
    st.error("Please select a valid date range.")
    st.stop()

if st.sidebar.button("Run"):
    filtered_df = df[(df['update_date'].dt.date <= end_date) & (df['update_date'].dt.date >= start_date)][0:10].reset_index()
    total_bills = len(filtered_df)

    if filtered_df.empty:
        st.write("No bills within the selected date range.")
    else:
        # Log in to Hugging Face and grant authorization to Hugging Chat
        sign = Login(email, passwd)
        cookies = sign.login()

        # Create a ChatBot
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

        st.subheader("Processing bills...")
        progress_bar = st.progress(0)  # Initialize progress bar
        progress_text = st.empty()  # Placeholder for progress text

        for index, row in filtered_df.iterrows():
            progress = index
            progress_bar.progress(progress / total_bills)
            progress_text.write(f"Processing Bill {progress} of {total_bills} Bills")
            bill_title = row['title']
            bill_summary = row['bill_summary_gpt']

            id = chatbot.new_conversation()
            chatbot.change_conversation(id)
            prompt = f'''You are a government relations professional tasked with monitoring bills in 
            the House of Representatives. Your client is interested in {client_input}.
             Your job is to review bill titles and summaries to identify bills 
             that align with the client's interests. 
             
             Please review the following bill and determine its relevance to the client's interest statement:
            
             Bill Title: {bill_title}
             Bill Summary: {bill_summary}
            
            
             Respond ONLY with the one-word answer to this question: Is this bill relevant to the 
             client's interest statement? (Yes/No).
             
             Do not include any other extraneous verbiage or information.
             '''
            time.sleep(3)
            response = chatbot.query(prompt)
            responses.append(response['text'].strip())

        filtered_df['response'] = responses
        relevant_bills = filtered_df[filtered_df['response'] == 'Yes']

        if not relevant_bills.empty:
            st.subheader("Relevant Bills:")
            for idx, row in relevant_bills.iterrows():
                st.write(f"**Bill Title:** {row['title']}")
                st.write(f"**Bill Summary:** {row['bill_summary_gpt']}")
                st.write(f"**URL:** [{row['url']}]({row['url']})")
                st.write('---')
        else:
            st.write("No relevant bills found.")

        st.write("Processing is complete.")
