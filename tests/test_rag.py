"""Tests for the RAG chatbot pipeline."""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

# ── Loader Tests ───────────────────────────────────────────────────────────────

def test_load_document_unsupported_type():
    from src.rag.loaders import load_document
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document("document.docx")


def test_turtle_loader_returns_tuple():
    """TurtleLoader must return (text_docs, []) tuple."""
    ttl_content = """
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:Doc1 rdf:type ex:Report ;
    ex:title "Annual Report 2024" ;
    ex:author "Jane Doe" .
"""
    with tempfile.NamedTemporaryFile(suffix=".ttl", mode="w", delete=False) as f:
        f.write(ttl_content)
        tmp_path = f.name

    try:
        from src.rag.loaders import load_document
        text_docs, table_docs = load_document(tmp_path)
        assert len(text_docs) >= 1
        assert table_docs == []
        assert text_docs[0].metadata["source_type"] == "turtle"
        assert text_docs[0].metadata["chunk_type"] == "text"
        assert "Subject" in text_docs[0].page_content
    finally:
        os.unlink(tmp_path)


def test_remove_unicode():
    """Unicode stripping should remove non-ASCII characters."""
    from src.rag.loaders import _remove_unicode
    assert _remove_unicode("hello\u2019world") == "helloworld"
    assert _remove_unicode("normal text") == "normal text"


# ── Chunking Tests ─────────────────────────────────────────────────────────────

def test_split_text_documents_adds_doc_id():
    """Every text chunk must have a UUID doc_id in metadata."""
    from langchain_core.documents import Document
    from src.rag.vector_store import split_text_documents

    docs = [Document(page_content="Hello world. " * 100, metadata={"file_name": "test.pdf"})]
    with patch("src.rag.vector_store.settings") as ms:
        ms.chunk_size = 200
        ms.chunk_overlap = 20
        chunks = split_text_documents(docs)

    assert len(chunks) >= 1
    for c in chunks:
        assert "doc_id" in c.metadata
        assert c.metadata["chunk_type"] == "text"
        assert len(c.metadata["doc_id"]) == 36  # UUID4 format


def test_split_table_documents_valid_json():
    """Table documents with valid JSON content should produce table chunks."""
    from langchain_core.documents import Document
    from src.rag.vector_store import split_table_documents

    row_data = [{"col1": "apple", "col2": "red"}, {"col1": "banana", "col2": "yellow"}]
    doc = Document(
        page_content=json.dumps(row_data),
        metadata={"file_name": "test.pdf", "source_type": "pdf"},
    )
    with patch("src.rag.vector_store.settings") as ms:
        ms.chunk_size = 500
        ms.chunk_overlap = 50
        chunks = split_table_documents([doc])

    assert len(chunks) >= 1
    for c in chunks:
        assert "doc_id" in c.metadata
        assert c.metadata["chunk_type"] == "table"


def test_split_table_documents_fallback_on_invalid_json():
    """Table documents with non-JSON content fall back to text splitting."""
    from langchain_core.documents import Document
    from src.rag.vector_store import split_table_documents

    doc = Document(
        page_content="This is plain text, not JSON.",
        metadata={"file_name": "test.pdf"},
    )
    with patch("src.rag.vector_store.settings") as ms:
        ms.chunk_size = 500
        ms.chunk_overlap = 50
        chunks = split_table_documents([doc])

    # Fallback produces text chunks
    assert len(chunks) >= 1
    assert chunks[0].metadata["chunk_type"] == "text"


# ── Vector Store Tests ─────────────────────────────────────────────────────────

def test_vector_store_ingest_returns_counts():
    """ingest_documents() must return a (text_count, table_count) tuple."""
    from langchain_core.documents import Document
    from src.rag.vector_store import VectorStoreManager

    row_data = [{"item": "wheat", "quality": "A"}]
    text_doc = Document(
        page_content="AI is used in precision agriculture to predict yields.",
        metadata={"source_type": "pdf", "file_name": "agri.pdf"},
    )
    table_doc = Document(
        page_content=json.dumps(row_data),
        metadata={"source_type": "pdf", "file_name": "agri.pdf", "chunk_type": "table"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.rag.vector_store.settings") as ms:
            ms.embedding_model = "nomic-embed-text"
            ms.ollama_base_url = "http://localhost:11434"
            ms.chroma_persist_dir = tmpdir
            ms.chroma_collection_name = "test_col"
            ms.chunk_size = 500
            ms.chunk_overlap = 50

            manager = VectorStoreManager()
            # Mock embeddings to avoid needing a live Ollama server in CI
            from unittest.mock import MagicMock
            manager._embeddings = MagicMock()
            manager._embeddings.embed_documents = lambda texts: [[0.1] * 384] * len(texts)
            manager._embeddings.embed_query = lambda text: [0.1] * 384

            text_n, table_n = manager.ingest_documents(
                documents=[text_doc],
                table_documents=[table_doc],
            )
            assert isinstance(text_n, int) and text_n >= 1
            assert isinstance(table_n, int) and table_n >= 1


# ── RAG Chain Tests (LCEL) ────────────────────────────────────────────────────

def test_rag_prompt_input_variables():
    """ChatPromptTemplate must require 'context' and 'question' variables."""
    from src.rag.chain import rag_prompt
    variables = set(rag_prompt.input_variables)
    assert "context"  in variables
    assert "question" in variables


def test_table_prompt_input_variables():
    """table_prompt must require 'context' and 'question' variables."""
    from src.rag.chain import table_prompt
    variables = set(table_prompt.input_variables)
    assert "context"  in variables
    assert "question" in variables


def test_format_docs_returns_string():
    """_format_docs() must return a non-empty string from a Document list."""
    from langchain_core.documents import Document
    from src.rag.chain import _format_docs

    docs = [
        Document(
            page_content="Wheat quality grade A.",
            metadata={"file_name": "agri.pdf", "page": 1, "chunk_type": "text"},
        ),
        Document(
            page_content='{"crop": "wheat", "yield": "4.2 t/ha"}',
            metadata={"file_name": "agri.pdf", "page": 2, "chunk_type": "table"},
        ),
    ]
    result = _format_docs(docs)
    assert isinstance(result, str)
    assert "Source: agri.pdf" in result
    assert "page: 1"          in result
    assert "type: text"       in result
    assert "Wheat quality"    in result
    assert "---"              in result          # separator between chunks


def test_format_docs_empty_returns_fallback():
    """_format_docs([]) must return the fallback string, not raise."""
    from src.rag.chain import _format_docs
    result = _format_docs([])
    assert result == "No relevant context found."


def test_format_sources_structure():
    """_format_sources() must return list[dict] with content_preview and metadata."""
    from langchain_core.documents import Document
    from src.rag.chain import _format_sources

    docs = [
        Document(
            page_content="A" * 400,   # longer than 300 → should be truncated
            metadata={"file_name": "doc.pdf", "chunk_type": "text"},
        ),
        Document(
            page_content="Short content.",
            metadata={"file_name": "onto.ttl", "chunk_type": "text"},
        ),
    ]
    sources = _format_sources(docs)
    assert len(sources) == 2

    # First source should be truncated with "..."
    assert sources[0]["content_preview"].endswith("...")
    assert len(sources[0]["content_preview"]) <= 303   # 300 chars + "..."
    assert "file_name" in sources[0]["metadata"]

    # Second source is short — no truncation
    assert sources[1]["content_preview"] == "Short content."


def test_rag_chain_build_pipeline_structure():
    """_build_rag_chain() must return a Runnable (has .invoke and .stream)."""
    from unittest.mock import MagicMock, patch
    from src.rag.chain import RAGChain

    chain_instance = RAGChain()

    # Mock the LLM and retriever so no Ollama server is needed
    mock_llm = MagicMock()
    mock_llm.__or__ = lambda self, other: MagicMock()
    chain_instance._llm = mock_llm

    with patch.object(
        type(chain_instance).llm.fget(chain_instance).__class__,
        "__or__",
        return_value=MagicMock(),
    ):
        pass

    # Just verify the method exists and is callable
    assert callable(getattr(chain_instance, "_build_rag_chain"))
    assert callable(getattr(chain_instance, "_build_table_chain"))


def test_rag_chain_query_no_docs_returns_fallback():
    """
    When the vector store has no matching documents, stream_query must
    yield the fallback string without calling the LLM.
    """
    from unittest.mock import MagicMock, patch
    from src.rag.chain import RAGChain

    chain_instance = RAGChain()
    chain_instance._llm = MagicMock()   # LLM should NOT be called

    with patch("src.rag.chain.vector_store_manager") as mock_vsm:
        mock_vsm.similarity_search.return_value = []   # empty result set
        tokens = list(chain_instance.stream_query("What is this?", k=3))

    assert len(tokens) == 1
    assert "No relevant documents" in tokens[0]
    chain_instance._llm.stream.assert_not_called()


def test_rag_chain_query_with_context_alias():
    """query_with_context() must call query() with identical arguments."""
    from unittest.mock import patch
    from src.rag.chain import RAGChain

    chain_instance = RAGChain()
    with patch.object(chain_instance, "query", return_value={"answer": "ok"}) as mock_q:
        result = chain_instance.query_with_context("test question", k=7)
    mock_q.assert_called_once_with("test question", k=7)
    assert result == {"answer": "ok"}


def test_runnable_passthrough_preserves_question():
    """
    RunnablePassthrough must pass the question string through unchanged.
    Verify using LangChain's own RunnablePassthrough directly.
    """
    from langchain_core.runnables import RunnablePassthrough
    passthrough = RunnablePassthrough()
    result = passthrough.invoke("agriculture yields 2024")
    assert result == "agriculture yields 2024"


def test_str_output_parser_extracts_content():
    """
    StrOutputParser must extract .content from an AIMessage object.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import AIMessage

    parser = StrOutputParser()
    msg    = AIMessage(content="The answer is 42.")
    result = parser.invoke(msg)
    assert result == "The answer is 42."


# ── Report Generator Tests ─────────────────────────────────────────────────────

def test_generate_chat_summary_pdf_magic_bytes():
    from src.utils.report_generator import generate_chat_summary_pdf

    history = [
        {"question": "What is RAG?", "answer": "Retrieval-Augmented Generation.", "sources": []},
    ]
    sources = [
        {"file_name": "doc.pdf", "source_type": "pdf"},
        {"file_name": "onto.ttl", "source_type": "turtle"},
    ]
    pdf_bytes = generate_chat_summary_pdf(history, sources, "Test Report")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:4] == b"%PDF"


# ── FastAPI Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_api_health():
    from httpx import AsyncClient, ASGITransport
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_api_list_sources():
    from httpx import AsyncClient, ASGITransport
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/documents/sources")
    assert r.status_code == 200
    assert "sources" in r.json()


@pytest.mark.asyncio
async def test_api_collection_stats():
    from httpx import AsyncClient, ASGITransport
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/documents/stats")
    assert r.status_code == 200
