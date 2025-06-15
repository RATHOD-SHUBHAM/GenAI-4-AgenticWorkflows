# Import Libraries

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# Get path
HOME = os.getcwd()
# print(HOME)
# ROOT = os.path.dirname(HOME)
# print(ROOT)
# BASE_DIR = os.path.dirname(HOME)
# print(BASE_DIR)


class HR_Knowledge_Worker:
    def __init__(self):
        self.persist_directory = f'{HOME}/avatar/create_knowledge_store/hr_db'

        # Todo: Load Pdf
        file_path = f'{HOME}/avatar/create_knowledge_store/knowledge_docs/HR.pdf'
        loader = PyPDFLoader(
            file_path=file_path
        )

        docs = loader.load()

        # Todo: Chunk Docs
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len,
            is_separator_regex=False

        )
        texts = text_splitter.split_documents(docs)

        # Todo: Embeddings
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest",
        )

        # Todo: Vector Store
        if os.path.exists(self.persist_directory):
            # Load from disk
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
        else:
            # Save to disk.
            db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )

        vectorstores_retriever = db.as_retriever(search_kwargs={"k": 3})

        # Todo: Keyword Store
        keyword_retriever = BM25Retriever.from_documents(
            documents=texts,
        )

        keyword_retriever.k = 3


        # Todo: Ensemble Retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[keyword_retriever, vectorstores_retriever], weights=[0.3, 0.7]
        )

    def run_hr_worker(self, model_name, user_input, chat_history):
        llm = ChatOllama(
            model=model_name
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
            self.ensemble_retriever,
            contextualize_q_prompt
        )

        # Todo: Answer Question
        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
            "If you don't know the answer, say that you don't know."
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )

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

        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        message = [
            HumanMessage(content=user_input),
            AIMessage(content=response['answer'])
        ]

        return response, message
