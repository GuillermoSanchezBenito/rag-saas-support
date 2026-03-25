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
    # init app DB and pipeline centrally
    logger.info(f"Starting {settings.app_name}...")
    
    try:
        db = VectorDB()
        app.state.rag = SupportRAGPipeline(db)
        logger.info("RAG and VectorDB ready.")
    except Exception as e:
        logger.error(f"AI init failed: {e}", exc_info=True)

    yield
    
    logger.info("Shutting down...")
    app.state.rag = None


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="SaaS RAG Support API",
        version="1.0.0",
        lifespan=lifespan
    )

    # setup cors
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # global error boundary
    @app.exception_handler(Exception)
    async def global_handler(req: Request, exc: Exception):
        logger.error("Unhandled error", exc_info=exc, extra={"path": req.url.path})
        return JSONResponse(status_code=500, content={"detail": "Internal error"})

    return app

app = create_app()
