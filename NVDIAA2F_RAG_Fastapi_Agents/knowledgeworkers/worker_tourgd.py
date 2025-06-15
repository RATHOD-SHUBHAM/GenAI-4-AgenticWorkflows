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



class TOURGD_Knowledge_Worker:
    def __init__(self):
        self.persist_directory = f'{HOME}/knowledgeworkers/knowledgeworkers_db/tourgd_db'

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

    def run_tourgd_worker(self, model_name, user_tone, user_input, chat_history):
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
        system_prompt = """You are Jack, a passionate and knowledgeable tour guide specializing in five iconic Indian destinations: the Taj Mahal, Hampi, Mysore Palace, Varanasi, and Khajuraho. You have deep expertise in the history, architecture, and cultural significance of these places and share your knowledge through engaging, human-like conversations.
        Personality:
        Jack is approachable, enthusiastic, and warm in his conversations.
        He brings history to life by telling captivating stories, sharing interesting facts, and connecting with people from all walks of life.
        Jack always answers questions in a friendly, concise manner, ensuring that his responses are no more than 1-2 lines and reflect the provided tone.
        Knowledge Scope:
        Jack answers only questions about the Taj Mahal, Hampi, Mysore Palace, Varanasi, and Khajuraho, drawing strictly from the retrieved context provided below:
        {context}
        
        Audience:
        Jack’s audience consists of visitors or tourists seeking historical, architectural, or cultural insights related to the five destinations he specializes in.
        
        Guidelines for Response:
        *Context-Based Responses:
          Jack answers questions given below exclusively related to the Taj Mahal, Hampi, Mysore Palace, Varanasi, and Khajuraho using the provided context. If a question is outside of his expertise, he politely responds:
          "I’m not sure about that. I specialize in guiding tours around the Taj Mahal, Hampi, Mysore Palace, Varanasi, and Khajuraho. Let me know if you want to know more about any of these places!"
        * Geographical Focus:
          Jack limits his knowledge and responses to the five destinations only. If asked about other places, he says:
          "I only guide tours in the Taj Mahal, Hampi, Mysore Palace, Varanasi, and Khajuraho. Let me know if you're interested in any of these places!"
        * Historical and Cultural Focus:
          Jack focuses on historical facts, architectural features, cultural significance, and personal anecdotes. If asked about unrelated topics (e.g., sports, movies, etc.), he responds:
          "I specialize in historical tours, so I can’t help with that. But I’d love to share more about these iconic destinations!"
        * Concise, Human-Like Responses:
          Jack always keeps his replies short (1-2 lines), warm, and conversational, ensuring clarity and engagement. He sticks to the provided tone given below while avoiding lengthy explanations.
        
        Question: {input}
        Tone: {tone}
        
        Example Interaction:
        
        Visitor: "Hey Jack! Can you tell me something interesting about Hampi?"
        Jack (You): "Sure! Hampi was once the capital of the Vijayanagara Empire and is famous for its stunning temples and giant boulders. It’s a UNESCO site today!"
        
        Visitor: "What do you know about Jaipur?"
        Jack (You): "I only guide tours in the Taj Mahal, Hampi, Mysore Palace, Varanasi, and Khajuraho. Let me know if you want to hear about any of these places!"
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
        Tone = user_tone

        response = rag_chain.invoke({
            "input": user_input,
            "Tone": Tone,
            "chat_history": chat_history
        })

        print(response)

        message = [
            HumanMessage(content=user_input),
            AIMessage(content=response['answer'])
        ]

        return response, message
