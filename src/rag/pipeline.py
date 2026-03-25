import tiktoken
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.retrieval.vectorstore import VectorDB
from src.rag.prompts import get_rag_prompt
from src.config import settings
from src.utils.logger import logger


class SupportRAGPipeline:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        
        # init retriever, fallback to None if db is empty
        try:
            self.retriever = self.vector_db.get_retriever(search_kwargs={"k": 5})
        except ValueError:
            logger.warning("Empty vectorstore. Add documents before querying.")
            self.retriever = None

        # setup llm with low temp for factual answers
        self.llm = ChatOpenAI(
            model_name=settings.llm_model, 
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.prompt = get_rag_prompt()
        self.parser = StrOutputParser()

    def _format_docs(self, docs: List[Document]) -> str:
        # merge retrieved docs into a single string with source info
        return "\n\n".join(
            f"Source [{d.metadata.get('source', 'Unknown')}]:\n{d.page_content}" 
            for d in docs
        )

    def _count_tokens(self, text: str) -> int:
        # quick token estimate
        try:
            enc = tiktoken.encoding_for_model(settings.llm_model)
            return len(enc.encode(text))
        except Exception:
            return 0

    async def aquery(self, question: str) -> Dict[str, Any]:
        # main async entrypoint for queries
        logger.info(f"Query: {question}")
        
        if not self.retriever:
            logger.error("Vector DB not initialized.")
            return {
                "answer": "Knowledge base empty. Please contact human support.",
                "sources": [],
                "metadata": {"error": "vector DB offline"}
            }

        # get context
        logger.debug("Fetching docs.")
        docs = await self.retriever.ainvoke(question)
        context = self._format_docs(docs)
        
        # chain execution
        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.parser
        )

        logger.debug("Running LLM chain.")
        answer = await chain.ainvoke(question)

        # build trackable metadata
        sources = [
            {
                "source": d.metadata.get("source", "Unknown"), 
                "page": d.metadata.get("page", None),
                "snippet": d.page_content[:150] + "..."
            } 
            for d in docs
        ]
        
        # estimate token usage
        in_tokens = self._count_tokens(context + question + self.prompt.messages[0].prompt.template)
        out_tokens = self._count_tokens(answer)

        logger.info("Query finished", extra={
            "query": question, 
            "total_tokens": in_tokens + out_tokens
        })
        
        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "model": settings.llm_model,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens
            }
        }
