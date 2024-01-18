import ast
from functools import partial

import gradio as gr

from chatbot_qa import similarity_search, question_answering


def question_answering_demo(question_text, *relevant_docs_input):
    relevant_docs = [ast.literal_eval(el) for el in relevant_docs_input]  # literal_eval - string -> dict
    answer = question_answering(question_text, relevant_docs)
    return answer


similarity_search_2_docs = partial(similarity_search, n_docs=2)

with gr.Blocks() as demo:
    gr.Markdown(
        f"""
    # Chatbot-QA-with-RAG
    ## Interactive demo
    """
    )
    with gr.Row():
        with gr.Column():
            question_text = gr.Textbox(label="Question")
        with gr.Column():
            btn_generate = gr.Button("Find relevant context")
    with gr.Row():
        with gr.Column():
            c1 = gr.Textbox(label="Context1")
        with gr.Column():
            c2 = gr.Textbox(label="Context2")
    with gr.Row():
        with gr.Column():
            answer_text = gr.Textbox(label="Answer")
        with gr.Column():
            btn_answer = gr.Button("Generate answer")
    btn_generate.click(similarity_search_2_docs, inputs=question_text, outputs=[c1, c2])
    btn_answer.click(question_answering_demo, inputs=[question_text, c1, c2], outputs=[answer_text])

demo.launch()
