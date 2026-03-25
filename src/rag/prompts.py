from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

SYSTEM_TEMPLATE = """You are a technical support engineer for a B2B SaaS company.
Provide accurate, helpful answers based ONLY on the provided context.

Rules:
1. Only use the info in the context.
2. If the answer isn't in the context, say you don't know and suggest human support.
3. Be concise. Provide step-by-step instructions if needed.
4. Keep a professional tone.
5. Format with Markdown.

Context:
---
{context}
---
"""

USER_TEMPLATE = """Question:
{question}
"""

def get_rag_prompt() -> ChatPromptTemplate:
    # compile prompt
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(USER_TEMPLATE)
    ])
