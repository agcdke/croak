"""FastMCP server exposing RAG tools for the chatbot."""
from pathlib import Path
from typing import List, Optional
from loguru import logger

from fastmcp import FastMCP

from src.rag.loaders import load_document
from src.rag.vector_store import vector_store_manager
from src.rag.chain import rag_chain
from src.utils.report_generator import generate_chat_summary_pdf
from src.config import settings


ALLOWED_EXTENSIONS = {".pdf", ".ttl", ".turtle"}

mcp = FastMCP(
    name="pdf-turtle-chatbot-mcp",
    instructions=(
        "MCP server for a RAG chatbot that can ingest PDF and Turtle RDF documents "
        "(single or batch), answer questions via semantic search, and generate PDF reports."
    ),
)


# ── Shared helper ──────────────────────────────────────────────────────────────

def _ingest_one_file(file_path: str) -> dict:
    """Load and ingest a single file. Returns a result dict."""
    path   = Path(file_path)
    suffix = path.suffix.lower()
    try:
        text_docs, table_docs = load_document(file_path)
        text_chunks, table_chunks = vector_store_manager.ingest_documents(
            documents=text_docs,
            table_documents=table_docs if table_docs else None,
        )
        return {
            "status": "success",
            "file_path": file_path,
            "file_name": path.name,
            "source_type": suffix.strip("."),
            "text_docs_loaded": len(text_docs),
            "table_docs_loaded": len(table_docs),
            "text_chunks_indexed": text_chunks,
            "table_chunks_indexed": table_chunks,
            "total_chunks_indexed": text_chunks + table_chunks,
        }
    except Exception as exc:
        logger.error(f"MCP ingest error for '{file_path}': {exc}")
        return {"status": "error", "file_path": file_path, "error": str(exc)}


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def ingest_document(file_path: str) -> dict:
    """
    Ingest a single PDF or Turtle (.ttl) file into ChromaDB.

    Args:
        file_path: Absolute or relative path to the .pdf or .ttl file.

    Returns:
        Ingestion result with text/table chunk counts and status.
    """
    logger.info(f"MCP ingest_document: {file_path}")
    return _ingest_one_file(file_path)


@mcp.tool()
def ingest_multiple_documents(file_paths: List[str]) -> dict:
    """
    Ingest multiple PDF and/or Turtle files in one call.

    Each file is processed independently — a failure in one file does NOT
    stop the others. Every file gets its own result entry.

    Args:
        file_paths: List of absolute or relative paths to .pdf / .ttl files.
                    Example: ["/data/report.pdf", "/data/ontology.ttl"]

    Returns:
        Aggregated dict with per-file results and summary counts.

    Example:
        ingest_multiple_documents([
            "/docs/standards.pdf",
            "/docs/crop_ontology.ttl",
            "/docs/yield_data.pdf",
        ])
    """
    logger.info(f"MCP ingest_multiple_documents: {len(file_paths)} files")

    if not file_paths:
        return {"status": "error", "error": "No file paths provided.", "results": []}

    results      = []
    succeeded    = 0
    failed       = 0
    skipped      = 0
    total_chunks = 0

    for fp in file_paths:
        suffix = Path(fp).suffix.lower()

        # Skip unsupported types gracefully
        if suffix not in ALLOWED_EXTENSIONS:
            logger.warning(f"Skipping unsupported file: {fp}")
            results.append({
                "status": "skipped",
                "file_path": fp,
                "error": f"Unsupported extension '{suffix}'",
            })
            skipped += 1
            continue

        result = _ingest_one_file(fp)
        results.append(result)

        if result["status"] == "success":
            succeeded    += 1
            total_chunks += result["total_chunks_indexed"]
        else:
            failed += 1

    logger.info(
        f"Batch ingestion done — succeeded={succeeded}, "
        f"failed={failed}, skipped={skipped}, "
        f"total_chunks={total_chunks}"
    )
    return {
        "status": "ok",
        "total_files": len(file_paths),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "total_chunks_indexed": total_chunks,
        "results": results,
    }


@mcp.tool()
def ingest_directory(directory_path: str, recursive: bool = False) -> dict:
    """
    Scan a directory and ingest all PDF and Turtle files found.

    Args:
        directory_path: Path to the directory to scan.
        recursive:      If True, scan all subdirectories as well.
                        If False (default), scan only the top level.

    Returns:
        Aggregated ingestion results for all discovered files.

    Example:
        ingest_directory("/data/documents", recursive=True)
    """
    logger.info(f"MCP ingest_directory: '{directory_path}' recursive={recursive}")

    scan_dir = Path(directory_path)
    if not scan_dir.exists():
        return {"status": "error", "error": f"Directory not found: {directory_path}"}
    if not scan_dir.is_dir():
        return {"status": "error", "error": f"Path is not a directory: {directory_path}"}

    # Collect files
    glob_pattern = "**/*" if recursive else "*"
    found_files  = sorted(
        f for f in scan_dir.glob(glob_pattern)
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )

    if not found_files:
        return {
            "status": "ok",
            "message": f"No supported files found in '{directory_path}'.",
            "directory": directory_path,
            "total_files": 0,
            "results": [],
        }

    # Reuse ingest_multiple_documents logic
    file_paths = [str(f) for f in found_files]
    result = ingest_multiple_documents(file_paths)
    result["directory"] = directory_path
    result["recursive"] = recursive
    return result


@mcp.tool()
def ask_question(question: str, k: int = 5) -> dict:
    """
    Ask a question against all ingested documents using RAG.

    Args:
        question: The natural language question to answer.
        k: Number of document chunks to retrieve for context (default 5).

    Returns:
        Dictionary with answer and source document previews.
    """
    try:
        logger.info(f"MCP: query '{question[:60]}...'")
        return rag_chain.query_with_context(question, k=k)
    except Exception as exc:
        logger.error(f"MCP query error: {exc}")
        return {"status": "error", "error": str(exc), "answer": ""}


@mcp.tool()
def list_indexed_documents() -> dict:
    """List all documents currently indexed in the vector store."""
    sources = vector_store_manager.list_sources()
    return {"sources": sources, "total": len(sources)}


@mcp.tool()
def search_documents(query: str, k: int = 5) -> dict:
    """
    Perform a semantic similarity search without generating an answer.

    Args:
        query: Search query string.
        k: Number of results to return.
    """
    try:
        docs    = vector_store_manager.similarity_search(query, k=k)
        results = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        return {"query": query, "results": results, "count": len(results)}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "results": []}


@mcp.tool()
def generate_session_report(chat_history: list, output_path: str) -> dict:
    """
    Generate a PDF summary report of a chat session and save it to disk.

    Args:
        chat_history: List of {"question": str, "answer": str} dicts.
        output_path: Path where the PDF should be saved.
    """
    try:
        sources   = vector_store_manager.list_sources()
        pdf_bytes = generate_chat_summary_pdf(
            chat_history=chat_history,
            document_sources=sources,
            title="MCP Chat Session Report",
        )
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        return {"status": "success", "output_path": output_path, "size_bytes": len(pdf_bytes)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def clear_vector_store() -> dict:
    """Clear all documents from the ChromaDB vector store."""
    try:
        vector_store_manager.delete_collection()
        return {"status": "success", "message": "Vector store cleared"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.resource("config://settings")
def get_settings() -> str:
    """Expose current application settings as an MCP resource."""
    return (
        f"Ollama Model: {settings.ollama_model}\n"
        f"Embedding Model: {settings.embedding_model}\n"
        f"ChromaDB Path: {settings.chroma_persist_dir}\n"
        f"Collection: {settings.chroma_collection_name}\n"
        f"Chunk Size: {settings.chunk_size}\n"
        f"Chunk Overlap: {settings.chunk_overlap}\n"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)