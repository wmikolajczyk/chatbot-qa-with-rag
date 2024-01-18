# Chatbot-QA-with-RAG
### Prepare and ingest data
1. Run `python prepare_data.py` to download dataset
2. Run `python ingest_data.py` to calculate embeddings for text documents and store them in vector db
### Run Chatbot QA as Gradio demo or API service (FastAPI)
- To run Chatbot QA in Gradio demo run `python gradio_demo.py`
- To run API service with Chatbot QA run `uvicorn api_endpoint:app --host 0.0.0.0 --port 80`
