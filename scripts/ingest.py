import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.retrieval.vectorstore import VectorDB
from src.utils.logger import logger

def main():
    logger.info("Starting Data Ingestion Process...")
    
    # 1. Load documents from the data directory
    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()
    
    if not documents:
        logger.warning("No documents found in data/raw. Add PDF or Markdown files and try again.")
        return
        
    # 2. Split documents into semantic chunks
    chunker = TextChunker()
    chunks = chunker.split_documents(documents)
    
    # 3. Embed and store chunks in Vector DB
    db = VectorDB()
    db.add_documents(chunks)
    
    logger.info("Data Ingestion complete! The database is ready for queries.")

if __name__ == "__main__":
    main()
