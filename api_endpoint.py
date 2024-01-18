from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot_qa import similarity_search, question_answering

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    text: str


@app.post("/get_context")
async def get_context(question: Question, n_docs: int = 3) -> dict:
    relevant_docs = similarity_search(question.text, n_docs=n_docs)
    return {"text_context": [el["text"] for el in relevant_docs], "id": [el["id"] for el in relevant_docs]}


@app.post("/generate_answer")
async def generate_answer(question: Question, n_docs: int = 3) -> dict:
    relevant_docs = similarity_search(question.text, n_docs=n_docs)
    answer = question_answering(question.text, relevant_docs)
    return {"answer": answer}
