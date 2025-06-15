# Import Libraries

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Get path
HOME = os.getcwd()
print(HOME)
ROOT = os.path.dirname(HOME)
print(ROOT)
# BASE_DIR = os.path.dirname(HOME)
# print(BASE_DIR)


class HR_Knowledge_Worker:
    def __init__(self):
        self.persist_directory = f'{HOME}/hr_db'

        # Todo: Load Pdf
        self.file_path = f'{ROOT}/knowledge_docs/HR.pdf'

    def create_DB(self):
        loader = PyPDFLoader(
            file_path=self.file_path
        )

        docs = loader.load()

        # Todo: Chunk Docs
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=800,
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
            return {'DB already exists'}
        else:
            # Save to disk.
            db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )


if __name__ == '__main__':
    obj = HR_Knowledge_Worker()
    obj.create_DB()