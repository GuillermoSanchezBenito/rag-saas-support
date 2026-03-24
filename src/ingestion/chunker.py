from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.utils.logger import logger

class TextChunker:
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents into smaller chunks."""
        if not documents:
            logger.warning("No documents provided for text splitting.")
            return []
            
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} text chunks.")
        
        return chunks
