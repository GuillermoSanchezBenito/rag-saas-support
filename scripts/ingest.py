import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.retrieval.vectorstore import VectorDB
from src.utils.logger import logger

def main():
    logger.info("Starting data ingestion...")
    
    # load docs
    loader = DocumentLoader("data/raw")
    docs = loader.load_documents()
    
    if not docs:
        logger.warning("No docs found in data/raw.")
        return
        
    # chunk
    chunker = TextChunker()
    chunks = chunker.split_documents(docs)
    
    # store
    db = VectorDB()
    db.add_documents(chunks)
    
    logger.info("Ingestion complete. DB ready.")

if __name__ == "__main__":
    main()
