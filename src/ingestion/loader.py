import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from src.utils.logger import logger

class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    def load_documents(self) -> List[Document]:
        """Loads all supported documents from the data directory."""
        logger.info(f"Loading documents from {self.data_dir}")
        documents = []

        # Load PDFs
        try:
            pdf_loader = DirectoryLoader(
                self.data_dir, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF documents.")
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}")

        # Load Markdown
        try:
            md_loader = DirectoryLoader(
                self.data_dir, 
                glob="**/*.md", 
                loader_cls=UnstructuredMarkdownLoader
            )
            md_docs = md_loader.load()
            documents.extend(md_docs)
            logger.info(f"Loaded {len(md_docs)} Markdown documents.")
        except Exception as e:
            logger.error(f"Error loading Markdown files: {e}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
