import asyncio
from src.retrieval.vectorstore import VectorDB
from src.rag.pipeline import SupportRAGPipeline
from src.utils.logger import logger

# A script to perform basic offline evaluation of the RAG system
# Evaluates purely Retrieval Metrics initially (e.g. how many correct documents are retrieved)
# For a full evaluation suite, consider integrating Ragas (https://docs.ragas.io)

EVAL_DATASET = [
    {
        "question": "How do I reset my API key?",
        "expected_keywords": ["settings", "dashboard", "developer", "regenerate"]
    },
    {
        "question": "What happens if I exceed my usage limit?",
        "expected_keywords": ["429", "rate limit", "upgrade", "block"]
    }
]

async def run_evaluation():
    logger.info("Starting Offline RAG Evaluation...")
    db = VectorDB()
    pipeline = SupportRAGPipeline(db)
    
    success_count = 0
    total = len(EVAL_DATASET)
    
    for item in EVAL_DATASET:
        question = item["question"]
        expected = item["expected_keywords"]
        
        logger.info(f"Evaluating: {question}")
        res = await pipeline.aquery(question)
        answer = res["answer"].lower()
        
        # Simple heuristic check
        matches = [kw for kw in expected if kw.lower() in answer]
        if len(matches) > 0:
            success_count += 1
            logger.info(f"SUCCESS: Found expected context -> {matches}")
        else:
            logger.warning(f"FAILED: Did not find expected keywords in answer.")

    logger.info(f"Evaluation Complete! Score: {success_count}/{total} ({(success_count/total)*100}%)")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
