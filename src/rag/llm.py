"""
Shared LLM factories — breaks circular imports between chain.py and sparql_engine.py.

Two models are used:
  get_llm()         → gemma3       — general text / RAG answers
  get_sparql_llm()  → mobr/cap     — NL → SPARQL generation (code-tuned model)

Both are configured via settings (OLLAMA_MODEL / SPARQL_MODEL env vars).
"""
from langchain_community.chat_models import ChatOllama
from src.config import settings


def get_llm() -> ChatOllama:
    """Return a ChatOllama instance for general RAG text generation (gemma3)."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
        num_gpu=settings.num_gpu,
    )


def get_sparql_llm() -> ChatOllama:
    """Return a ChatOllama instance for SPARQL generation (mobr/cap)."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.sparql_model,
        temperature=0.0,   # zero temperature — we want deterministic, exact code
        num_gpu=settings.num_gpu,
    )