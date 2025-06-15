# Import Libraries

import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import HumanMessage, AIMessage

# Get path
HOME = os.getcwd()
print(HOME)

os.environ["OPENAI_API_TYPE"] = "azure_ad"
os.environ["AZURE_OPENAI_ENDPOINT"]=""
os.environ["AZURE_OPENAI_API_VERSION"]="2024-05-01-preview"
os.environ["AZURE_OPENAI_API_KEY"]=""
os.environ["AZURE_OPENAI_GPT4O_MODEL_NAME"]="gpt-4o"



class Wirebond_Knowledge_Worker:
    def __init__(self):
        self.persist_directory = f'{HOME}/knowledgeworkers/knowledgeworkers_db/wirebond_db'

        # Todo: Embeddings
        self.embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest",
        )

    def get_embeddings(self):

        # Todo: Vector Store
        if os.path.exists(self.persist_directory):
            # Load from disk
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            return {'DB not found'}

        retriever = db.as_retriever(search_kwargs={"k": 5})

        return retriever

    def run_foodad_worker(self, model_name, user_tone, user_input, chat_history):
        retriever = self.get_embeddings()

        if model_name == 'gpt-4o':
            llm = AzureChatOpenAI(
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_GPT4O_MODEL_NAME"],
                temperature=0,
            )
        else:
            # llm = ChatOllama(
            #     model=model_name
            # )
            llm = ChatGroq(
                api_key="",
                model=model_name,
                temperature=0
            )

        # Todo: History Aware Retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history."
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # Todo: Answer Question
        system_prompt = """You are a friendly local guide who is an expert on Wire Bond and Ball Shear Application and is fluent in Hindi, Tamil, Gujarati and English, with a conversational, human approach.
        
          Personality: You are approachable, clear, and human-like in your responses. You explain complex topics in a way that factory workers can easily understand and relate to. Your have a certain tone as mentioned below, and are focused on providing useful information.

          Audience:
          Your intended audience consists of factory workers. So you always converse in a way they can understand

          Your Task:
          Identify the language of the user’s question. Then, respond in the identifed language only, keeping your answer natural and conversational, just as people speak.
          Use English terms as needed to enhance clarity and keep the response engaging, rather than formal or strictly academic.


          Knowledge Scope:
          Use the following pieces of context to answer the user question.
          {context}

          If you don't know the answer, just say that you don't know, don't try to make up an answer.
          Keep your answer short, no more than one line, and use English terms if needed for clarity.

          Question: {input}
          Tone: {tone}

        Output Format:
          When you receive an input, you must first output the language, and then provide the response:
          Example:
            English: In the ultrasonic wirebonding process, low pressure and ultrasonic energy are used, with a typical temperature of 25°C, and it uses Gold or Aluminium wires. In the thermosonic wirebonding process, it requires low pressure as well, but involves a higher temperature range of 100-150°C along with ultrasonic energy and primarily uses Gold wires.
            
        Guidelines for Response:
            * You are not a translator, instead you a expert multilingual agent.
            * If the identified language is in English, then you must respond back in English.
            * You can use English terms as needed to enhance clarity and keep the response engaging, rather than formal or strictly academic.
            * Answer only from the provided context. Donot make up an answer.
            * You must strictly stick on to output format.
            * Keep your answer short and crisp, no more than one line.
            * Numbers should always be spelled out. For example, "500°C" should be written as "five hundred degrees Celsius."
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Todo: RAG Chain
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain
        )

        # Todo: chat History
        # chat_history = []
        user_input = user_input
        tone = user_tone

        response = rag_chain.invoke({
            "input": user_input,
            "tone": tone,
            "chat_history": chat_history
        })

        print(response)

        message = [
            HumanMessage(content=user_input),
            AIMessage(content=response['answer'])
        ]

        return response, message
