from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from src.retrieval.vectorstore import VectorDB
from src.rag.prompts import get_rag_prompt
from src.config import settings
from src.utils.logger import logger
import tiktoken

class SupportRAGPipeline:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        try:
            self.retriever = self.vector_db.get_retriever(search_kwargs={"k": 5})
        except ValueError as e:
            logger.warning("Vectorstore not initialized. Starting with empty retriever context. Add documents before querying.")
            self.retriever = None

        self.llm = ChatOpenAI(
            model_name=settings.llm_model, 
            temperature=0.1,  # Low temperature for factual consistency
            api_key=settings.openai_api_key
        )
        self.prompt = get_rag_prompt()
        self.output_parser = StrOutputParser()

    def _format_docs(self, docs: List[Document]) -> str:
        """Formats the retrieved documents into a single contextual string with metadata."""
        formatted_str = "\n\n".join(
            f"Source [{doc.metadata.get('source', 'Unknown')}]:\n{doc.page_content}" 
            for doc in docs
        )
        return formatted_str

    def _count_tokens(self, text: str) -> int:
        """Estimates the number of tokens using tiktoken (Bonus Step)."""
        try:
            encoding = tiktoken.encoding_for_model(settings.llm_model)
            return len(encoding.encode(text))
        except Exception:
            return 0

    async def aquery(self, question: str) -> Dict[str, Any]:
        """Asynchronously executes the RAG pipeline."""
        logger.info(f"Processing query: {question}")
        
        if self.retriever is None:
            logger.error("Attempted to query an empty vector database.")
            return {
                "answer": "The knowledge base is currently empty. Please contact human support.",
                "sources": [],
                "metadata": {"error": "Vector database not initialized."}
            }

        # 1. Retrieve Context
        logger.debug("Retrieving context from vector store.")
        retrieved_docs = await self.retriever.ainvoke(question)
        context_str = self._format_docs(retrieved_docs)
        
        # 2. Build Pipeline
        rag_chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.output_parser
        )

        # 3. Generate Answer
        logger.debug("Generating response via LLM.")
        answer = await rag_chain.ainvoke(question)

        # 4. Compile Metadata for Observability (Bonus Step)
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"), 
                "page": doc.metadata.get("page", None),
                "content_snippet": doc.page_content[:150] + "..."
            } 
            for doc in retrieved_docs
        ]
        
        input_tokens = self._count_tokens(context_str + question + self.prompt.messages[0].prompt.template)
        output_tokens = self._count_tokens(answer)

        response_payload = {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "model": settings.llm_model,
                "input_tokens_estimate": input_tokens,
                "output_tokens_estimate": output_tokens,
                "total_tokens_estimate": input_tokens + output_tokens
            }
        }
        
        logger.info("Query successfully processed", extra={
            "query": question, 
            "total_tokens": input_tokens + output_tokens
        })
        return response_payload
