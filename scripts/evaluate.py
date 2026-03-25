import asyncio
from src.retrieval.vectorstore import VectorDB
from src.rag.pipeline import SupportRAGPipeline
from src.utils.logger import logger

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
    logger.info("Starting offline eval...")
    db = VectorDB()
    pipeline = SupportRAGPipeline(db)
    
    success_count = 0
    total = len(EVAL_DATASET)
    
    for item in EVAL_DATASET:
        question = item["question"]
        expected = item["expected_keywords"]
        
        logger.info(f"Eval: {question}")
        res = await pipeline.aquery(question)
        answer = res["answer"].lower()
        
        matches = [kw for kw in expected if kw.lower() in answer]
        if matches:
            success_count += 1
            logger.info(f"SUCCESS: matched {matches}")
        else:
            logger.warning("FAILED: no keywords matched")

    score = (success_count / total) * 100
    logger.info(f"Done. Score: {success_count}/{total} ({score:.1f}%)")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
