import os
import pandas as pd

# OpenAI API configuration
os.environ["OPENAI_BASE_URL"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-Ef8IbvRhChFvZbQG8WZQUR967z765rzDXSAbskZxtO3k7bfn" #迭代100 1
model="gpt-4o-mini"  # General conversation model

temperature=1
max_tokens=16000
gamma=0.0001
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns