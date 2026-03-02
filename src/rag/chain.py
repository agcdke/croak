"""
RAG chain built with pure LangChain Expression Language (LCEL).

Libraries used:
  - ChatOllama          — chat-oriented LLM (local, free via Ollama)
  - ChatPromptTemplate  — structured system/human message prompt
  - RunnablePassthrough — passes values through the pipeline unchanged
  - StrOutputParser     — extracts plain string from the LLM AIMessage response

LCEL pipeline for query() and query_with_context():

    {"context": retriever | _format_docs,
     "question": RunnablePassthrough()}
                │
                ▼
        ChatPromptTemplate   ← formats system + human messages
                │
                ▼
           ChatOllama        ← generates AIMessage
                │
                ▼
         StrOutputParser     ← extracts str from AIMessage.content

LCEL pipeline for query_table_only():
    Same structure but uses a table-filtered retriever.

LCEL pipeline for stream_query():
    Same chain but calls .stream() instead of .invoke().
"""
from typing import Any, Dict, Generator, List

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from loguru import logger

from src.config import settings
from src.rag.llm import get_llm
from src.rag.vector_store import vector_store_manager
from src.guardrails import validate_input, validate_output


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
# {context}  → filled by the retriever branch via _format_docs()
# {question} → passed through unchanged by RunnablePassthrough()

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions strictly based on "
    "the provided context documents.\n\n"
    "Guidelines:\n"
    "- Answer ONLY from the context below. Do NOT use outside knowledge.\n"
    "- If the context does not contain enough information, say so explicitly.\n"
    "- For RDF/Turtle data, interpret Subject–Predicate–Object triples meaningfully.\n"
    "- For tabular data (JSON chunks), read field names and values carefully.\n"
    "- Be concise, accurate, and well-structured.\n\n"
    "Context:\n{context}"
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),   # instructions + {context}
        ("human", "{question}"),     # user question via RunnablePassthrough
    ]
)

# Separate prompt for table-only queries (stricter framing)
TABLE_SYSTEM_PROMPT = (
    "You are a data analyst assistant. Answer questions based ONLY on the "
    "structured table data provided in the context below.\n\n"
    "Guidelines:\n"
    "- Read JSON field names and values precisely.\n"
    "- Perform aggregations or comparisons only if the data supports them.\n"
    "- If the data is insufficient, say so clearly.\n\n"
    "Table Context:\n{context}"
)

table_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TABLE_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _format_docs(docs: List[Document]) -> str:
    """
    Format a list of retrieved Documents into a single context string.

    This function is used as a Runnable step inside the LCEL chain:

        retriever | _format_docs   →  plain string injected into {context}

    Each chunk is annotated with its source file, page number, and chunk type
    so the LLM can attribute its answer correctly.
    """
    if not docs:
        return "No relevant context found."
    parts = []
    for doc in docs:
        fname  = doc.metadata.get("file_name", "unknown")
        page   = doc.metadata.get("page", "?")
        ctype  = doc.metadata.get("chunk_type", "text")
        header = f"[Source: {fname} | page: {page} | type: {ctype}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Build the sources list returned alongside every answer.
    Compatible with FastAPI response model, MCP tool output, and Streamlit UI.
    """
    return [
        {
            "content_preview": doc.page_content[:300] + (
                "..." if len(doc.page_content) > 300 else ""
            ),
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


# ══════════════════════════════════════════════════════════════════════════════
# RAGChain
# ══════════════════════════════════════════════════════════════════════════════

class RAGChain:
    """
    Retrieval-Augmented Generation chain built with pure LCEL.

    Core pipeline (query / query_with_context):
    ─────────────────────────────────────────────
                          question (str)
                               │
             ┌─────────────────┴──────────────────┐
             │                                    │
    "context": retriever               "question": RunnablePassthrough()
             │ _format_docs()                     │
             └─────────────────┬──────────────────┘
                               │  dict: {context: str, question: str}
                               ▼
                     ChatPromptTemplate
                    (system msg + human msg)
                               │
                               ▼
                          ChatOllama
                        (generates AIMessage)
                               │
                               ▼
                        StrOutputParser
                        (extracts str)
                               │
                               ▼
                          answer (str)

    Extra sources are retrieved separately (same retriever, no LLM call)
    and returned alongside the answer for display in the UI.
    """

    def __init__(self):
        self._llm: ChatOllama | None = None

    # ── LLM (lazy-loaded) ──────────────────────────────────────────────────────

    @property
    def llm(self) -> ChatOllama:
        """
        Lazy-load ChatOllama on first use via shared get_llm() factory.
        ChatOllama uses the chat-message API (vs raw completions),
        which is required for ChatPromptTemplate and StrOutputParser.
        """
        if self._llm is None:
            logger.info(
                f"Initialising ChatOllama — model={settings.ollama_model}  "
                f"base_url={settings.ollama_base_url}"
            )
            self._llm = get_llm()
        return self._llm

    # ── LCEL chain builders ────────────────────────────────────────────────────

    def _build_rag_chain(self, k: int = 5):
        """
        Build the core LCEL RAG chain for text (and mixed) retrieval.

        Pipeline breakdown:
          1. Input dict {"question": str} enters the chain.

          2. RunnablePassthrough passes the question string through unchanged
             into the "question" key of the prompt.

          3. The retriever branch:
               retriever.invoke(question)  →  List[Document]
               _format_docs(docs)          →  str
             produces the context string for the "context" key.

          4. ChatPromptTemplate formats system + human messages.

          5. ChatOllama generates an AIMessage.

          6. StrOutputParser extracts AIMessage.content → plain str.
        """
        retriever = vector_store_manager.as_retriever(k=k)

        chain = (
            {
                # Branch 1: retrieve docs → format as context string
                "context":  retriever | _format_docs,
                # Branch 2: pass the question through unchanged
                "question": RunnablePassthrough(),
            }
            | rag_prompt          # ChatPromptTemplate → ChatPromptValue
            | self.llm            # ChatOllama         → AIMessage
            | StrOutputParser()   # StrOutputParser    → str
        )
        return chain

    def _build_table_chain(self, k: int = 5):
        """
        Build a table-specific LCEL chain using the table_prompt.
        Retrieves only chunk_type == 'table' documents.
        """
        # Use a filtered similarity search wrapped as a simple callable Runnable
        def table_retriever(question: str) -> List[Document]:
            return vector_store_manager.similarity_search_by_type(
                query=question, chunk_type="table", k=k
            )

        chain = (
            {
                "context":  table_retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | table_prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    # ── Public query methods ───────────────────────────────────────────────────

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Run end-to-end RAG with input + output guardrails.

        Pipeline:
          1. INPUT GUARD   — validate_input() blocks injection / harmful / off-topic
          2. RAG CHAIN     — LCEL retriever | prompt | LLM | StrOutputParser
          3. OUTPUT GUARD  — validate_output() blocks empty / hallucinated / harmful answers
          4. Return        — {question, answer, sources, guardrail_blocked, guardrail_code}

        Args:
            question: Natural language question from the user.
            k:        Number of chunks to retrieve (default 5).

        Returns:
            dict with keys:
              question          : str
              answer            : str
              sources           : list[dict]
              guardrail_blocked : bool  — True if a guard fired
              guardrail_code    : str   — guard code ("ok" when safe)
        """
        logger.info(f"RAG query [{k} chunks]: '{question[:80]}...'")

        # ── 1. INPUT GUARDRAIL ─────────────────────────────────────────────
        input_guard = validate_input(question)
        if not input_guard.passed:
            logger.warning(
                f"[GUARDRAIL] Input blocked — code={input_guard.code} "
                f"detail={input_guard.detail}"
            )
            return {
                "question":          question,
                "answer":            input_guard.reason,
                "sources":           [],
                "guardrail_blocked": True,
                "guardrail_code":    input_guard.code,
            }

        # ── 2. RAG CHAIN ───────────────────────────────────────────────────
        chain  = self._build_rag_chain(k=k)
        answer = chain.invoke(question)

        docs    = vector_store_manager.similarity_search(question, k=k)
        sources = _format_sources(docs)

        # ── 3. OUTPUT GUARDRAIL ────────────────────────────────────────────
        output_guard = validate_output(answer)
        if not output_guard.passed:
            logger.warning(
                f"[GUARDRAIL] Output blocked — code={output_guard.code} "
                f"detail={output_guard.detail}"
            )
            return {
                "question":          question,
                "answer":            output_guard.reason,
                "sources":           sources,
                "guardrail_blocked": True,
                "guardrail_code":    output_guard.code,
            }

        logger.info(f"Answer generated. Sources used: {len(docs)}")
        return {
            "question":          question,
            "answer":            answer,
            "sources":           sources,
            "guardrail_blocked": False,
            "guardrail_code":    "ok",
        }

    def query_with_context(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Backward-compatible alias for query().
        Called by: FastAPI endpoints, MCP tools, Streamlit UI.
        """
        return self.query(question, k=k)

    def query_table_only(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        RAG query restricted to table chunks only.

        Uses the table_prompt (stricter data-analysis framing) and retrieves
        only document chunks with chunk_type == 'table'.

        Useful for questions like:
          - "What are the values in column X?"
          - "Which row has the highest Y?"

        Args:
            question: The user's question about tabular data.
            k:        Number of table chunks to retrieve.
        """
        logger.info(f"Table-only RAG query [{k} chunks]: '{question[:80]}...'")

        # Pre-check: are there any table chunks at all?
        table_docs = vector_store_manager.similarity_search_by_type(
            query=question, chunk_type="table", k=k
        )
        if not table_docs:
            return {
                "question": question,
                "answer":   "No relevant table data found in the knowledge base.",
                "sources":  [],
            }

        chain  = self._build_table_chain(k=k)
        answer = chain.invoke(question)

        logger.info(f"Table answer generated. Sources used: {len(table_docs)}")
        return {
            "question": question,
            "answer":   answer,
            "sources":  _format_sources(table_docs),
        }

    def stream_query(self, question: str, k: int = 5) -> Generator[str, None, None]:
        """
        Stream the answer token-by-token via the same LCEL chain.

        Calls chain.stream() instead of chain.invoke(); StrOutputParser
        ensures each yielded chunk is a plain string (not an AIMessageChunk).

        Usage in Streamlit:
            st.write_stream(rag_chain.stream_query("What is..."))

        Usage in CLI:
            for token in rag_chain.stream_query("What is..."):
                print(token, end="", flush=True)

        Args:
            question: Natural language question.
            k:        Number of chunks to retrieve.

        Yields:
            str — successive tokens of the generated answer.
        """
        logger.info(f"Streaming RAG query [{k} chunks]: '{question[:60]}...'")

        # Bail early if nothing is indexed
        docs = vector_store_manager.similarity_search(question, k=k)
        if not docs:
            yield "No relevant documents found in the knowledge base."
            return

        chain = self._build_rag_chain(k=k)
        for token in chain.stream(question):
            yield token


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════
rag_chain = RAGChain()