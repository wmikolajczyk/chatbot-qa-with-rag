import abc
import glob
import json
import logging
from functools import lru_cache

import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

DATA_DIR = "data"
SPLIT_INTO_CHUNKS = False
MAX_CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TEXT_SPLITTER_SEPARATOR = ""
VECTOR_DB_PATH = "vector_db"

MODEL_ID = "hf-e5"
VECTOR_DB_ID = "faiss"


def get_device():
    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logger.info(f"device: {device}")
    return device


class AbstractEmbeddings:
    @abc.abstractmethod
    def embed_documents(self, docs):
        pass


class HFE5Embeddings(AbstractEmbeddings):
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2", model_kwargs={"device": get_device()}
        )

    def embed_documents(self, docs):
        embeddings = self.embedding_model.embed_documents(docs)
        return embeddings


@lru_cache(maxsize=1)
def get_embedding_model(model_id) -> AbstractEmbeddings:
    models = {"hf-e5": HFE5Embeddings}
    model_cls = models[model_id]
    return model_cls()


class AbstractVectorDb:
    @abc.abstractmethod
    def load_vector_db(self):
        pass

    @abc.abstractmethod
    def store_embeddings(self, docs_embeddings_pairs, metadata):
        pass

    @abc.abstractmethod
    def similarity_search(self, query, n_docs=3):
        pass


class FAISSVectorDb(AbstractVectorDb):
    def __init__(self, embedding_model, vector_db_path):
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path

    @lru_cache(maxsize=1)
    def load_vector_db(self):
        vector_db = FAISS.load_local(self.vector_db_path, self.embedding_model)
        return vector_db

    def store_embeddings(self, docs_embeddings_pairs, metadata):
        # feel free to change vector db
        vector_db = FAISS.from_embeddings(docs_embeddings_pairs, self.embedding_model, metadata)
        vector_db.save_local(self.vector_db_path)

    def similarity_search(self, query, n_docs=3):
        vector_db = self.load_vector_db()

        query_embedding = self.embedding_model.embed_documents([query])[0]
        relevant_docs = vector_db.similarity_search_by_vector(query_embedding, k=n_docs)

        return relevant_docs


@lru_cache(maxsize=1)
def get_vector_db(vector_db_id, embedding_model, vector_db_path) -> AbstractVectorDb:
    vector_dbs = {"faiss": FAISSVectorDb}
    vector_db_cls = vector_dbs[vector_db_id]
    return vector_db_cls(embedding_model, vector_db_path)


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
    embedding_model = get_embedding_model(MODEL_ID)
    embeddings = embedding_model.embed_documents(docs)
    docs_embeddings_pairs = list(zip(docs, embeddings))

    # store data in vector db
    vector_db = get_vector_db(VECTOR_DB_ID, embedding_model, VECTOR_DB_PATH)
    vector_db.store_embeddings(docs_embeddings_pairs, metadata)


if __name__ == "__main__":
    main()
