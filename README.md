# legis_monitor_poc
As an intern at a government relations firm, I often monitored official gov't publications for updates of interest to particular clients. In today's Generative AI-driven world, this web app is a proof-of-concept tool designed to help accomplish just that: allow clients to track legislation relevant to their personal or commercial interests.

# About the Data
For this app, I used the Congress.Gov API to extract information related to bills that were *updated* in the month of February 2023. Then, I used GPT-4 to summarize each of those bills (see https://github.com/andrewallen42/legis_monitor_poc/edit/main/bill_data_collection.py). The app takes as input a client statement of interest (ex: "I'm interested in legislation related to environmental issues and climate change") and includes it in a prompt that classifies the bills into Relevant or Not-Relevant.

# Limitations
The app currently uses HuggingFace's HuggingChat (powered by LLaMa), and often runs into rate limits. In addition, the dataset contains a relatively narrow set of bills; therefore, many of the topics of the client's interest might not be included.

# Next Steps
  -Fine-tune GPT-4 or another model on the bill summaries that already exist on Congress.gov.
  -Address the rate limit challenge.
