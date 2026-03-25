from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.utils.logger import logger

class TextChunker:
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        # main text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        # split docs into chunks
        if not docs:
            logger.warning("No docs provided for splitting.")
            return []
            
        logger.info(f"Splitting {len(docs)} docs...")
        chunks = self.splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks.")
        
        return chunks
