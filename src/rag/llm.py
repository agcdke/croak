"""
Shared LLM factory — breaks the circular import between chain.py and vector_store.py.

Both chain.py and vector_store.py need a ChatOllama instance.
Previously chain.py exposed _get_llm() and vector_store.py imported it,
creating a circular dependency:
    vector_store.py → chain.py → vector_store.py

Solution: move the LLM factory here. Both modules import from llm.py,
which imports from neither of them.
"""
from langchain_community.chat_models import ChatOllama
from src.config import settings


def get_llm() -> ChatOllama:
    """Return a ChatOllama instance using current settings."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
        num_gpu=settings.num_gpu,
    )