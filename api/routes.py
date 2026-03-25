from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from api.dependencies import get_pipeline
from src.rag.pipeline import SupportRAGPipeline
from src.utils.logger import logger


router = APIRouter()

class Query(BaseModel):
    query: str = Field(..., description="User query")

class Source(BaseModel):
    source: str
    page: int | None
    snippet: str

class Response(BaseModel):
    answer: str
    sources: List[Source]
    metadata: Dict[str, Any]

@router.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}

@router.post("/query", response_model=Response, tags=["RAG"])
async def handle_query(
    req: Query,
    rag: SupportRAGPipeline = Depends(get_pipeline)
):
    """Process RAG query."""
    try:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Empty query")
            
        return await rag.aquery(req.query)
    
    except Exception as e:
        logger.error(f"Query error: {req.query}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {e}"
        )
