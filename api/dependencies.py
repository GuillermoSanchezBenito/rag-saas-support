from fastapi import Request
from src.retrieval.vectorstore import VectorDB
from src.rag.pipeline import SupportRAGPipeline

# Dependency Injection logic is handled at app startup in main.py to keep the
# DB connection and models loaded in memory once.

def get_rag_pipeline(request: Request) -> SupportRAGPipeline:
    """FastAPI dependency to get the instantiated RAG pipeline from application state."""
    return request.app.state.rag_pipeline
