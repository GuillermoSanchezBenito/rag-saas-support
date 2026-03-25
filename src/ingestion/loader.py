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
        # load all docs from dir
        logger.info(f"Loading docs from {self.data_dir}")
        docs = []

        # load pdfs
        try:
            pdf_loader = DirectoryLoader(
                self.data_dir, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
            pdfs = pdf_loader.load()
            docs.extend(pdfs)
            logger.info(f"Loaded {len(pdfs)} PDFs.")
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}")

        # load markdown
        try:
            md_loader = DirectoryLoader(
                self.data_dir, 
                glob="**/*.md", 
                loader_cls=UnstructuredMarkdownLoader
            )
            mds = md_loader.load()
            docs.extend(mds)
            logger.info(f"Loaded {len(mds)} Markdowns.")
        except Exception as e:
            logger.error(f"Error loading Markdowns: {e}")

        logger.info(f"Total docs loaded: {len(docs)}")
        return docs
