from langchain_groq import ChatGroq

import getpass
import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file and sets them in os.environ
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

def main():
    print("Hello from ocr-ai-agent!")

    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    )

    ai_msg = llm.invoke('Tell me a joke and include some emojis')
    print(ai_msg.content)


if __name__ == "__main__":
    main()