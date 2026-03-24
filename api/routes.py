from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from api.dependencies import get_rag_pipeline
from src.rag.pipeline import SupportRAGPipeline
from src.utils.logger import logger

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., example="How do I reset my password?", description="The user's question about the SaaS software.")

class SourceInfo(BaseModel):
    source: str
    page: int | None
    content_snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any]

@router.get("/health", tags=["System"])
async def health_check():
    """Returns the health status of the API."""
    logger.info("Health check endpoint pinged.")
    return {"status": "ok", "message": "Service is healthy"}

@router.post("/query", response_model=QueryResponse, tags=["RAG Support"])
async def query_support(
    request: QueryRequest,
    pipeline: SupportRAGPipeline = Depends(get_rag_pipeline)
):
    """
    Submits a query to the Support RAG pipeline.
    Retrieves context from the Vector DB and generates a factual response.
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
            
        result = await pipeline.aquery(request.query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {request.query}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the response: {str(e)}"
        )
