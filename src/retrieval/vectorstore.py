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
        # load existing faiss index or wait for first doc
        if os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
            logger.info(f"Loading FAISS index from {self.persist_directory}")
            self.vectorstore = FAISS.load_local(
                self.persist_directory, 
                self.embedding_function,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("No FAISS index found. Will init on first add_documents.")

    def add_documents(self, docs: List[Document]):
        # add chunks to db and persist
        if not docs:
            logger.warning("No docs to add.")
            return

        logger.info(f"Adding {len(docs)} chunks to vector store...")
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embedding_function)
        else:
            self.vectorstore.add_documents(docs)
            
        self.save()
        logger.info("Docs added to vector store.")

    def save(self):
        # save faiss index to disk
        if self.vectorstore is not None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vectorstore.save_local(self.persist_directory)
            logger.info(f"Vector store saved to {self.persist_directory}")

    def get_retriever(self, search_kwargs: dict = {"k": 4}):
        # return retriever interface for RAG
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Add docs first.")
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
