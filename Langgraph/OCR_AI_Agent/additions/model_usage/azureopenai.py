# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file and sets them in os.environ

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')

def main():
    print("Hello from ocr-ai-agent!")

    llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version='2025-03-01-preview'
    )

    ai_msg = llm.invoke('Tell me a joke and include some emojis')
    print(ai_msg.content)


if __name__ == "__main__":
    main()
