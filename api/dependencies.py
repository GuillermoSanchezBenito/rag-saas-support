from fastapi import Request
from src.rag.pipeline import SupportRAGPipeline

def get_pipeline(req: Request) -> SupportRAGPipeline:
    # get active rag pipeline from app state
    return req.app.state.rag
