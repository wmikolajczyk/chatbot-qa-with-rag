from functools import lru_cache

from langchain.chains import LLMChain
from langchain_community.llms.openai import OpenAI
from langchain_core.prompts import PromptTemplate

from ingest_data import get_embedding_model, get_vector_db

VECTOR_DB_PATH = "./vector_db"
MODEL_ID = "hf-e5"
VECTOR_DB_ID = "faiss"
LLM_MODEL_ID = "openai"


@lru_cache(maxsize=1)
def get_llm_model(llm_model_id):
    models = {"openai": OpenAI}
    model_cls = models[llm_model_id]
    return model_cls(temperature=0)


def similarity_search(query, n_docs=3):
    embedding_model = get_embedding_model(MODEL_ID)
    vector_db = get_vector_db(VECTOR_DB_ID, embedding_model, VECTOR_DB_PATH)

    relevant_docs = vector_db.similarity_search(query, n_docs=n_docs)
    result = [{"id": doc.metadata["id"], "text": doc.page_content} for doc in relevant_docs]
    return result


def question_answering(question, relevant_docs):
    context_str = "\n\n".join(doc["text"] for doc in relevant_docs)

    llm_model = get_llm_model(LLM_MODEL_ID)

    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
    qa_chain_prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    llm_chain = LLMChain(prompt=qa_chain_prompt, llm=llm_model, verbose=True)

    answer = llm_chain.run(question=question, context=context_str)
    return answer


def simple_question_answering(question):
    llm_model = get_llm_model(LLM_MODEL_ID)

    prompt_template = """{question}"""
    qa_chain_prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    llm_chain = LLMChain(prompt=qa_chain_prompt, llm=llm_model, verbose=True)

    answer = llm_chain.run(question=question)
    return answer


def main():
    query = "Who is James Bond?"
    relevant_docs = similarity_search(query)
    print(relevant_docs)

    answer = question_answering(query, relevant_docs)
    print(answer)


if __name__ == "__main__":
    main()
