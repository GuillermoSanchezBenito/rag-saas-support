import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.config import settings
from src.utils.logger import logger

class VectorDB:
    def __init__(self):
        self.persist_directory = settings.vector_db_path
        self.embedding_function = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key
        )
        self.vectorstore = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Loads existing FAISS index or prepares for a new one."""
        if os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
            logger.info(f"Loading existing FAISS index from {self.persist_directory}")
            self.vectorstore = FAISS.load_local(
                self.persist_directory, 
                self.embedding_function,
                allow_dangerous_deserialization=True  # Required for local FAISS loading in newer Langchain versions
            )
        else:
            logger.info("No existing FAISS index found. Initialization will happen on first add_documents.")

    def add_documents(self, documents: List[Document]):
        """Adds text chunks to the vector database and saves to disk."""
        if not documents:
            logger.warning("No documents to add to the vector store.")
            return

        logger.info(f"Adding {len(documents)} chunks to the vector store...")
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embedding_function)
        else:
            self.vectorstore.add_documents(documents)
            
        self.save()
        logger.info("Successfully added documents to vector store.")

    def save(self):
        """Persists the FAISS index to disk."""
        if self.vectorstore is not None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vectorstore.save_local(self.persist_directory)
            logger.info(f"Vector store saved to {self.persist_directory}")

    def get_retriever(self, search_kwargs: dict = {"k": 4}):
        """Returns the retrieval object for the RAG pipeline."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized. Please add documents first.")
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
