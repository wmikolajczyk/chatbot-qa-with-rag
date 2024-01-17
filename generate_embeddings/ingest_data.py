import glob
import json
import logging

import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

DATA_DIR = "./data"
SPLIT_INTO_CHUNKS = False
MAX_CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TEXT_SPLITTER_SEPARATOR = ""
VECTOR_DB_PATH = "./vector_db"


def get_device():
    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logger.info(f"device: {device}")
    return device


def get_embedding_model():
    # feel free to change embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={"device": get_device()})
    return embedding_model


def store_data_in_vector_db(docs_embeddings_pairs, embedding_model, metadata, vector_db_path):
    # feel free to change vector db
    vector_db = FAISS.from_embeddings(docs_embeddings_pairs, embedding_model, metadata)
    vector_db.save_local(vector_db_path)


def load_vector_db(vector_db_path, embedding_model):
    vector_db = FAISS.load_local(vector_db_path, embedding_model)
    return vector_db


def main():
    # Read docs
    logger.info("Reading docs...")

    docs = []
    metadata = []
    for filepath in glob.glob(f"{DATA_DIR}/*.json"):
        with open(filepath, "r") as f:
            data_dict = json.load(f)
            docs.append(data_dict["text"])
            metadata.append({"id": data_dict["id"]})
    logger.info(f"Number of documents: {len(docs)}")

    # split into chunks (if needed)
    if SPLIT_INTO_CHUNKS:
        text_splitter = CharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator=TEXT_SPLITTER_SEPARATOR
        )
        docs_chunks = text_splitter.create_documents(docs, metadata)
        docs = [doc.page_content for doc in docs_chunks]
        metadata = [doc.metadata for doc in docs_chunks]

    # calculate embeddings
    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_documents(docs)
    docs_embeddings_pairs = list(zip(docs, embeddings))

    # store data in vector db
    store_data_in_vector_db(docs_embeddings_pairs, embedding_model, metadata, VECTOR_DB_PATH)


if __name__ == "__main__":
    main()
