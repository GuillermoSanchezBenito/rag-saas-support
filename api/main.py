from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from src.retrieval.vectorstore import VectorDB
from src.rag.pipeline import SupportRAGPipeline
from src.config import settings
from src.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Initializes the Vector DB and RAG pipeline centrally.
    """
    logger.info(f"Starting {settings.app_name} Application...")
    
    # Initialize heavy components only once
    try:
        vector_db = VectorDB()
        app.state.rag_pipeline = SupportRAGPipeline(vector_db)
        logger.info("RAG Pipeline and VectorDB initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize AI components: {e}", exc_info=True)
        # We don't crash the server here so health checks can still report failure if needed.

    yield
    
    logger.info("Shutting down Application...")
    # Clean up resources if necessary
    app.state.rag_pipeline = None

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Retrieval-Augmented Generation API for SaaS Technical Support",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware to allow requests from hypothetical frontend dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # In production, lock this down
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register the router
    app.include_router(router)

    # Add Top-Level Exception Handler for JSON logging of unhandled errors
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled Exception Caught", exc_info=exc, extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"}
        )

    return app

app = create_app()
