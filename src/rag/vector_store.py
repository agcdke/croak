"""
ChromaDB vector store manager.

Adapted from reference implementation (agcdke/toadlet-llm):

  split_text()
    RecursiveCharacterTextSplitter with separators=["\n\n", "\n", " ", ""]
    operates on the raw joined text string (not pre-split Documents).

  split_table_to_json()
    RecursiveJsonSplitter.split_json(json_data=table_docs, convert_lists=True)
    receives the full list-of-lists-of-dicts from extract_pdf_text_tables().

  Flatten pattern (mirrors reference get_text_table_from_pdf_files()):
    table_json_chunks = [v for element in table_chunks for _, v in element.items()]

  UUID doc_id per chunk — assigned after splitting (reference id_key pattern).
  OllamaEmbeddings — free, local, no API key.
  gc.collect() after ingestion — matches reference memory management.
"""
import gc
import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from loguru import logger

from src.config import settings


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ID_KEY = "doc_id"   # metadata key for UUID — mirrors reference id_key


def remove_unicode(text: str) -> str:
    """
    Remove unicode characters.
    Mirrors reference remove_unicode() exactly.
    """
    return re.sub(r'[^\x00-\x7F]+', '', text)


# ══════════════════════════════════════════════════════════════════════════════
# SPLITTING FUNCTIONS  (direct ports from reference)
# ══════════════════════════════════════════════════════════════════════════════

def split_text(docs: str) -> List[str]:
    """
    Split document text into chunks.
    Mirrors reference split_text() exactly.

    Args:
        docs: raw joined text string (from extract_pdf_text_tables)

    Returns:
        list of text chunk strings
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    text_chunks = text_splitter.split_text(docs)
    logger.info(f"Split texts from RecursiveCharacterTextSplitter: {len(text_chunks)} chunks.")
    return text_chunks


def split_table_to_json(table_docs: List) -> List[dict]:
    """
    Split document tables and convert into JSON format.
    Mirrors reference split_table_to_json() exactly.

    Args:
        table_docs: list[list[dict]] — the table_list returned by
                    extract_pdf_text_tables() (one inner list per page)

    Returns:
        list of dicts from RecursiveJsonSplitter (before flattening)
    """
    table_splitter = RecursiveJsonSplitter(max_chunk_size=settings.chunk_size)
    table_chunks = table_splitter.split_json(
        json_data=table_docs,
        convert_lists=True,
    )
    logger.info(f"Split tables from RecursiveJsonSplitter: {len(table_chunks)} chunks.")
    return table_chunks


def split_text_documents(docs: List[Document]) -> List[Document]:
    """
    Convenience wrapper: split a list of text Documents using split_text().

    Joins all page_contents, runs split_text(), then assigns UUID doc_ids.
    Used by ingest_documents() for the text pipeline.
    """
    # Join all document page contents (TTL files arrive pre-chunked, PDFs as one doc)
    joined = " ".join(d.page_content for d in docs if d.page_content)
    if not joined.strip():
        return []

    # Carry metadata from the first document (file_name, source_type, etc.)
    base_meta = docs[0].metadata if docs else {}

    text_chunks = split_text(joined)
    text_doc_ids = [str(uuid.uuid4()) for _ in text_chunks]

    doc_text_chunks = [
        Document(
            page_content=remove_unicode(string_val),
            metadata={
                **base_meta,
                ID_KEY:       text_doc_ids[i],
                "chunk_type": "text",
            },
        )
        for i, string_val in enumerate(text_chunks)
    ]

    logger.info(f"Created {len(doc_text_chunks)} LangChain text Documents.")
    return doc_text_chunks


def split_table_documents(table_docs: List[Document]) -> List[Document]:
    """
    Convert table Documents into ChromaDB-ready LangChain Documents.

    Each incoming Document holds a JSON string of a list-of-dicts
    (one inner list per PDF page, each dict being one table row).

    Pipeline (mirrors reference get_text_table_from_pdf_files() table branch):
      1. Decode JSON string → list[list[dict]] (collect all pages)
      2. split_table_to_json(tables)
             → RecursiveJsonSplitter produces list[dict] where each dict
               has integer-string keys: {"0": row_dict, "1": row_dict, ...}
      3. Flatten:
             table_json_chunks = [v for element in table_chunks
                                   for _, v in element.items()]
             → list of individual row dicts
      4. Assign UUID doc_id per row-dict, wrap in Document.
    """
    if not table_docs:
        return []

    # ── Step 1: collect all page table lists into one structure ────────────
    # Each Document.page_content = JSON string of list[dict] (one page's rows)
    # Rebuild the full table_list: list[list[dict]]
    all_tables: List[List[dict]] = []
    base_meta = table_docs[0].metadata if table_docs else {}

    for doc in table_docs:
        try:
            page_rows = json.loads(doc.page_content)
            if isinstance(page_rows, list) and page_rows:
                all_tables.append(page_rows)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                f"Skipping non-JSON table doc "
                f"(file={doc.metadata.get('file_name','?')}): {e}"
            )

    if not all_tables:
        logger.warning("No valid table data found to split.")
        return []

    # ── Step 2: split with RecursiveJsonSplitter ───────────────────────────
    table_chunks = split_table_to_json(all_tables)

    # ── Step 3: flatten (mirrors reference exactly) ────────────────────────
    # table_chunks is list[dict], each dict looks like:
    #   {"0": {"col1": "val", ...}, "1": {"col1": "val", ...}, ...}
    # We want each individual row dict as a separate chunk.
    table_json_chunks = [
        v
        for element in table_chunks
        for _, v in element.items()
    ]
    logger.info(f"Created {len(table_json_chunks)} table JSON chunks (after flatten).")

    # ── Step 4: wrap in LangChain Documents with UUID doc_ids ─────────────
    table_doc_ids = [str(uuid.uuid4()) for _ in table_json_chunks]

    doc_table_chunks = [
        Document(
            page_content=remove_unicode(str(json_val)),
            metadata={
                **base_meta,
                ID_KEY:       table_doc_ids[i],
                "chunk_type": "table",
            },
        )
        for i, json_val in enumerate(table_json_chunks)
    ]

    logger.info(f"Created {len(doc_table_chunks)} LangChain table Documents.")
    return doc_table_chunks


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """
    Manages ChromaDB vector store: ingestion, retrieval, and lifecycle.

    Embedding model : OllamaEmbeddings (free, local, no API key)
    Text chunking   : RecursiveCharacterTextSplitter via split_text()
    Table chunking  : RecursiveJsonSplitter via split_table_to_json() + flatten
    """

    def __init__(self):
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._store: Optional[Chroma] = None

    # ── Embeddings (lazy) ──────────────────────────────────────────────────────

    @property
    def embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            logger.info(
                f"Initialising OllamaEmbeddings: "
                f"model={settings.embedding_model}  "
                f"base_url={settings.ollama_base_url}"
            )
            self._embeddings = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
            )
        return self._embeddings

    # ── ChromaDB store (lazy) ──────────────────────────────────────────────────

    def _persist_dir(self) -> Path:
        """Return the persist directory (always absolute after config.ensure_dirs)."""
        p = Path(settings.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def store(self) -> Chroma:
        """Lazy-load ChromaDB. Always uses the absolute persist path."""
        if self._store is None:
            persist_dir = self._persist_dir()
            self._store = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir),
            )
            logger.info(
                f"ChromaDB ready — collection='{settings.chroma_collection_name}' "
                f"path={persist_dir}"
            )
        return self._store

    def _add_to_store(self, chunks: List[Document]) -> None:
        """Add chunks to ChromaDB using the same absolute persist path."""
        persist_dir = self._persist_dir()
        self.store.add_documents(documents=chunks)
        logger.info(
            f"Added {len(chunks)} chunks to '{settings.chroma_collection_name}' "
            f"at {persist_dir}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest_documents(
        self,
        documents: List[Document],
        table_documents: Optional[List[Document]] = None,
    ) -> Tuple[int, int]:
        """
        Ingest text and table documents into ChromaDB.

        Text pipeline:
            split_text_documents() → split_text() →
            RecursiveCharacterTextSplitter → UUID per chunk → ChromaDB

        Table pipeline:
            split_table_documents() → split_table_to_json() →
            RecursiveJsonSplitter → flatten → UUID per row → ChromaDB

        Args:
            documents:       Text LangChain Documents (PDF text, TTL chunks)
            table_documents: Table LangChain Documents (JSON-encoded row lists)

        Returns:
            (text_chunk_count, table_chunk_count)
        """
        text_count  = 0
        table_count = 0

        # ── Text pipeline ──────────────────────────────────────────────────
        if documents:
            text_chunks = split_text_documents(documents)
            if text_chunks:
                self._add_to_store(text_chunks)
                text_count = len(text_chunks)
                logger.info(f"Stored {text_count} text chunks.")

        # ── Table pipeline ─────────────────────────────────────────────────
        if table_documents:
            table_chunks = split_table_documents(table_documents)
            if table_chunks:
                self._add_to_store(table_chunks)
                table_count = len(table_chunks)
                logger.info(f"Stored {table_count} table chunks.")

        # Free memory (mirrors reference gc.collect() pattern)
        gc.collect()

        logger.info(
            f"Ingestion complete — "
            f"text_chunks={text_count}, table_chunks={table_count}"
        )
        return text_count, table_count

    def _multi_query_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Multi-query retrieval — implemented without MultiQueryRetriever
        to avoid brittle third-party import paths across LangChain versions.

        Pipeline:
          1. Use the LLM to generate 3 alternative phrasings of the query.
          2. Run each variant + the original through ChromaDB similarity_search.
          3. Union and deduplicate all results by page_content.
          4. Return up to k unique Documents.

        This produces richer context than a single embedding search because
        different phrasings surface different relevant chunks.
        """
        from src.rag.llm import get_llm

        llm = get_llm()

        # ── Step 1: generate alternative query phrasings ──────────────────────
        prompt = (
            "Generate 3 different phrasings of the following question to improve "
            "document retrieval. Output only the 3 questions, one per line, "
            "no numbering, no explanation."
            "\n\nQuestion: " + remove_unicode(query)
        )
        try:
            response  = llm.invoke(prompt)
            variants  = [
                line.strip()
                for line in response.content.strip().splitlines()
                if line.strip()
            ][:3]
        except Exception as e:
            logger.warning(f"Multi-query LLM call failed ({e}); falling back to single query.")
            variants = []

        all_queries = [remove_unicode(query)] + [remove_unicode(v) for v in variants]
        logger.info(f"MultiQuery variants: {all_queries}")

        # ── Step 2: search each variant ───────────────────────────────────────
        seen:   set             = set()
        unique: List[Document]  = []
        search_kwargs = {"k": k, **({"filter": filter} if filter else {})}

        for q in all_queries:
            try:
                results = self.store.similarity_search(q, **search_kwargs)
            except Exception as e:
                logger.warning(f"similarity_search failed for variant '{q[:40]}': {e}")
                continue
            for doc in results:
                key = doc.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    unique.append(doc)
                if len(unique) >= k:
                    return unique

        return unique

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k relevant chunks using multi-query search.
        Generates alternative phrasings via LLM then unions the results.
        """
        return self._multi_query_search(query, k=k)

    def similarity_search_by_type(
        self,
        query: str,
        chunk_type: str = "text",
        k: int = 5,
    ) -> List[Document]:
        """
        Retrieve chunks filtered by chunk_type ('text' or 'table')
        using multi-query search.
        """
        return self._multi_query_search(
            query, k=k, filter={"chunk_type": chunk_type}
        )

    def as_retriever(self, k: int = 5):
        """
        Return a LangChain-compatible retriever wrapping _multi_query_search.
        Used by chain._build_rag_chain() as the retriever branch.
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun

        store = self

        class _MultiQueryRetriever(BaseRetriever):
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> List[Document]:
                return store._multi_query_search(query, k=k)

        return _MultiQueryRetriever()

    def list_sources(self) -> List[dict]:
        """List all unique document sources with chunk-type breakdown."""
        try:
            results = self.store._collection.get(include=["metadatas"])
            seen: dict = {}
            for meta in results["metadatas"]:
                fname = meta.get("file_name", "unknown")
                if fname not in seen:
                    seen[fname] = {
                        "file_name":   fname,
                        "source_type": meta.get("source_type", "unknown"),
                        "source":      meta.get("source", ""),
                        "text_chunks": 0,
                        "table_chunks": 0,
                    }
                if meta.get("chunk_type") == "table":
                    seen[fname]["table_chunks"] += 1
                else:
                    seen[fname]["text_chunks"] += 1
            return list(seen.values())
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []

    def collection_stats(self) -> dict:
        """Return collection-level statistics."""
        try:
            results = self.store._collection.get(include=["metadatas"])
            metas   = results["metadatas"]
            total   = len(metas)
            text_n  = sum(1 for m in metas if m.get("chunk_type") == "text")
            table_n = sum(1 for m in metas if m.get("chunk_type") == "table")
            return {
                "collection":    settings.chroma_collection_name,
                "total_chunks":  total,
                "text_chunks":   text_n,
                "table_chunks":  table_n,
                "unique_sources": len({m.get("file_name") for m in metas}),
            }
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            return {}

    def delete_collection(self) -> None:
        """Delete the ChromaDB collection and reset the in-memory store."""
        self.store.delete_collection()
        self._store = None
        gc.collect()
        logger.warning(
            f"Deleted ChromaDB collection '{settings.chroma_collection_name}'."
        )


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════
vector_store_manager = VectorStoreManager()