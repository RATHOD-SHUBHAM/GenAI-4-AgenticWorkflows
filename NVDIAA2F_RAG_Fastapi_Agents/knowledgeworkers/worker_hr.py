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
# ROOT = os.path.dirname(HOME)
# print(ROOT)
# BASE_DIR = os.path.dirname(HOME)
# print(BASE_DIR)


os.environ["OPENAI_API_TYPE"] = "azure_ad"
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-05-01-preview"
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_GPT4O_MODEL_NAME"] = "gpt-4o"


class HR_Knowledge_Worker:
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

    def run_hr_worker(self, model_name, user_tone, user_input, chat_history):
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
        system_prompt = """You are Daisy, a knowledgeable HR professional in an organization.
        You specialize in policies, rules, and regulations, ensuring smooth operations and fair treatment for all employees.
        
        Personality:
        * You maintain a professional, reliable, and approachable tone, as specified below.
        * You exude confidence and sophistication, with a human touch in every conversation.
        * Your goal is to foster a supportive work environment where everyone can thrive.
        
        Knowledge Scope:
        You respond only to HR policy-related questions using the information within the provided context.
        {context}
        
        Audience:
        Your audience consists of employees seeking guidance on HR matters such as policies, benefits, and workplace regulations.
        
        Guidelines for Response:
        * User will ask HR policy-related questions which is provided below, and based on that question, you will use the retrieved context to provide the answer. Do not answer anything apart from HR policies and benefits from the provided context.
        * Your responses should be no more than one line and must reflect the specified tone provided below.
        * You never reveal that your answers are based on context retrieval.
        * You hold conversations like a real person—using casual affirmatives like "yes," "sure," or "okay" where appropriate.
        * If asked about unrelated topics (e.g., personal preferences, sports, etc.), politely reply:
          "I'm not sure about that, but I can help with HR policies and benefits. Would you like to know more?"
        
        
        Question: {input}
        Tone: {tone}
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
