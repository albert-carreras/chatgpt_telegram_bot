import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader


def ingest_docs(embeddings):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(script_dir, 'data', 'Basic.txt')

    documents = TextLoader(file).load()

    texts = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    ).split_documents(documents)

    FAISS.from_documents(texts, embeddings).save_local("faiss_index")
