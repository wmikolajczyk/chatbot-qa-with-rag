from ingest_data import get_embedding_model, get_vector_db

VECTOR_DB_PATH = "./vector_db"
MODEL_ID = "hf-e5"
VECTOR_DB_ID = "faiss"


def similarity_search(query):
    embedding_model = get_embedding_model(MODEL_ID)
    vector_db = get_vector_db(VECTOR_DB_ID, embedding_model, VECTOR_DB_PATH)

    relevant_docs = vector_db.similarity_search(query)
    return relevant_docs


def main():
    query = "Top Gun"
    relevant_docs = similarity_search(query)
    print(relevant_docs)


if __name__ == "__main__":
    main()
