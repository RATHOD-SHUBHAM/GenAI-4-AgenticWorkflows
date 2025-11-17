"""
1. init_chat_model

https://api.python.langchain.com/en/latest/chat_models/langchain.chat_models.base.init_chat_model.html

https://reference.langchain.com/python/langchain/models/
"""

# ---------------------- Groq Model Using init_chat ----------------------
from langchain.chat_models import init_chat_model
import getpass
import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file and sets them in os.environ
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

o3_mini = init_chat_model("groq:openai/gpt-oss-20b", temperature=0)

print(o3_mini.invoke("what's your name").content)

# ---------------------- Azure Chat OpenAI Model Using init_chat ----------------------
from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file and sets them in os.environ

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')

o3_mini = init_chat_model("azure_openai:gpt-4o-mini", temperature=0, api_version='2025-03-01-preview')

print(o3_mini.invoke("what's your name").content)