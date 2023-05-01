import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader


def ingest_docs(embeddings):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Basic data
    basic_file = os.path.join(script_dir, "data", "Basic.txt")
    documents = TextLoader(basic_file).load()

    # Whatsapp conversations
    # whatsapp_file = os.path.join(script_dir, 'data', 'whatsapp.txt')
    # documents.extend(TextLoader(whatsapp_file).load())

    # Telegram conversations
    # telegram_file = os.path.join(script_dir, 'data', 'telegram.json')
    # documents.extend(TelegramDocumentLoader(telegram_file).load())

    texts = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    ).split_documents(documents)

    FAISS.from_documents(texts, embeddings).save_local("faiss_index")
