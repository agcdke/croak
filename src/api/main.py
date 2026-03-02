"""FastAPI backend for the PDF & Turtle RAG Chatbot."""
import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

from src.config import settings
from src.rag.loaders import load_document
from src.rag.vector_store import vector_store_manager
from src.rag.chain import rag_chain
from src.utils.report_generator import generate_chat_summary_pdf, generate_document_summary_pdf
from loguru import logger


ALLOWED_EXTENSIONS = {".pdf", ".ttl", ".turtle"}

app = FastAPI(
    title="PDF & Turtle RAG Chatbot API",
    description="Chat with PDF and Turtle RDF documents using LangChain + ChromaDB + Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session chat history (keyed by session_id)
_chat_sessions: dict[str, list] = {}


# ── Pydantic Models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    k: int = 5


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list
    session_id: str
    guardrail_blocked: bool = False
    guardrail_code: str = "ok"


class HealthResponse(BaseModel):
    status: str
    message: str


class FileResult(BaseModel):
    """Result for a single file in a batch ingestion."""
    file_name: str
    status: str                     # "success" | "error" | "skipped"
    source_type: str = ""
    text_pages_loaded: int = 0
    table_pages_loaded: int = 0
    text_chunks_indexed: int = 0
    table_chunks_indexed: int = 0
    total_chunks_indexed: int = 0
    error: str = ""


class BatchUploadResponse(BaseModel):
    """Aggregated response for /documents/upload-batch."""
    total_files: int
    succeeded: int
    failed: int
    skipped: int
    results: List[FileResult]
    total_chunks_indexed: int


# ── Shared ingestion helper ────────────────────────────────────────────────────

def _ingest_one(file_path: Path, original_name: str) -> FileResult:
    """
    Load and ingest a single file already saved to disk.
    Returns a FileResult regardless of success or failure.
    """
    suffix = file_path.suffix.lower()
    try:
        text_docs, table_docs = load_document(str(file_path))
        # Preserve the original upload filename in metadata
        for doc in text_docs + table_docs:
            doc.metadata["file_name"] = original_name

        text_chunks, table_chunks = vector_store_manager.ingest_documents(
            documents=text_docs,
            table_documents=table_docs if table_docs else None,
        )
        return FileResult(
            file_name=original_name,
            status="success",
            source_type=suffix.strip("."),
            text_pages_loaded=len(text_docs),
            table_pages_loaded=len(table_docs),
            text_chunks_indexed=text_chunks,
            table_chunks_indexed=table_chunks,
            total_chunks_indexed=text_chunks + table_chunks,
        )
    except Exception as exc:
        logger.error(f"Ingestion error for '{original_name}': {exc}")
        return FileResult(
            file_name=original_name,
            status="error",
            source_type=suffix.strip("."),
            error=str(exc),
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    return {"status": "ok", "message": "PDF & Turtle RAG Chatbot API is running"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "message": "Service healthy"}


# ── Single file upload ─────────────────────────────────────────────────────────

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a single PDF or Turtle (.ttl) file into ChromaDB."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    save_path = Path(settings.upload_dir) / file.filename
    with open(save_path, "wb") as f:
        #shutil.copyfileobj(file.file, f)
        f.write(uploaded_file.getbuffer())
    logger.info(f"Saved: {save_path}")

    result = _ingest_one(save_path, file.filename)
    if result.status == "error":
        raise HTTPException(status_code=500, detail=result.error)

    return {
        "status": "success",
        "file_name": result.file_name,
        "source_type": result.source_type,
        "text_pages_loaded": result.text_pages_loaded,
        "table_pages_loaded": result.table_pages_loaded,
        "text_chunks_indexed": result.text_chunks_indexed,
        "table_chunks_indexed": result.table_chunks_indexed,
        "total_chunks_indexed": result.total_chunks_indexed,
        "message": f"Successfully ingested '{file.filename}' into ChromaDB",
    }


# ── Batch file upload ──────────────────────────────────────────────────────────

@app.post("/documents/upload-batch", response_model=BatchUploadResponse)
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Upload and ingest multiple PDF / Turtle files in a single request.

    All files are processed sequentially. Each file returns an individual
    result (success, error, or skipped) regardless of the others.

    Example curl:
        curl -X POST http://localhost:8000/documents/upload-batch \\
             -F "files=@report.pdf" \\
             -F "files=@ontology.ttl" \\
             -F "files=@standards.pdf"
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results: List[FileResult] = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()

        # Skip unsupported types gracefully
        if suffix not in ALLOWED_EXTENSIONS:
            logger.warning(f"Skipping unsupported file type: {upload.filename}")
            results.append(FileResult(
                file_name=upload.filename,
                status="skipped",
                error=f"Unsupported extension '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
            ))
            continue

        # Save to disk
        save_path = Path(settings.upload_dir) / upload.filename
        with open(save_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        logger.info(f"Saved: {save_path}")

        # Ingest
        result = _ingest_one(save_path, upload.filename)
        results.append(result)

    succeeded = sum(1 for r in results if r.status == "success")
    failed    = sum(1 for r in results if r.status == "error")
    skipped   = sum(1 for r in results if r.status == "skipped")
    total_chunks = sum(r.total_chunks_indexed for r in results)

    logger.info(
        f"Batch complete — {succeeded} succeeded, {failed} failed, "
        f"{skipped} skipped, {total_chunks} total chunks indexed."
    )

    return BatchUploadResponse(
        total_files=len(files),
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        results=results,
        total_chunks_indexed=total_chunks,
    )


# ── Ingest from server-side directory ─────────────────────────────────────────

@app.post("/documents/ingest-directory")
async def ingest_directory(
    directory: str = Query(
        default=None,
        description="Server-side directory path to scan. "
                    "Defaults to the configured UPLOAD_DIR if not specified.",
    ),
):
    """
    Scan a server-side directory and ingest all PDF / Turtle files found.

    Useful when files are already on the server (e.g. mounted volume, CI pipeline).
    Scans one level deep (non-recursive) by default.

    Example curl:
        curl -X POST "http://localhost:8000/documents/ingest-directory"
        curl -X POST "http://localhost:8000/documents/ingest-directory?directory=/data/docs"
    """
    scan_dir = Path(directory) if directory else Path(settings.upload_dir)

    if not scan_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {scan_dir}",
        )
    if not scan_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a directory: {scan_dir}",
        )

    # Collect all supported files
    found_files = [
        f for f in sorted(scan_dir.iterdir())
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not found_files:
        return {
            "status": "ok",
            "message": f"No supported files found in '{scan_dir}'.",
            "directory": str(scan_dir),
            "total_files": 0,
            "results": [],
        }

    results: List[FileResult] = []
    for file_path in found_files:
        logger.info(f"Ingesting from directory: {file_path.name}")
        result = _ingest_one(file_path, file_path.name)
        results.append(result)

    succeeded    = sum(1 for r in results if r.status == "success")
    failed       = sum(1 for r in results if r.status == "error")
    total_chunks = sum(r.total_chunks_indexed for r in results)

    return {
        "status": "ok",
        "directory": str(scan_dir),
        "total_files": len(found_files),
        "succeeded": succeeded,
        "failed": failed,
        "total_chunks_indexed": total_chunks,
        "results": [r.model_dump() for r in results],
    }


# ── Document management ────────────────────────────────────────────────────────

@app.get("/documents/sources")
async def list_sources():
    """List all documents currently in the vector store."""
    sources = vector_store_manager.list_sources()
    return {"sources": sources, "total": len(sources)}


@app.get("/documents/stats")
async def collection_stats():
    """Return chunk-level statistics for the current ChromaDB collection."""
    return vector_store_manager.collection_stats()


@app.delete("/documents/clear")
async def clear_documents():
    """Delete all documents from the vector store."""
    vector_store_manager.delete_collection()
    return {"status": "success", "message": "All documents cleared from vector store"}


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question against ingested documents."""
    try:
        result = rag_chain.query_with_context(request.question, k=request.k)
        if request.session_id not in _chat_sessions:
            _chat_sessions[request.session_id] = []
        _chat_sessions[request.session_id].append(result)
        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id,
            guardrail_blocked=result.get("guardrail_blocked", False),
            guardrail_code=result.get("guardrail_code", "ok"),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    history = _chat_sessions.get(session_id, [])
    return {"session_id": session_id, "history": history, "total": len(history)}


@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    _chat_sessions.pop(session_id, None)
    return {"status": "success", "message": f"Cleared history for session '{session_id}'"}


# ── Reports ────────────────────────────────────────────────────────────────────

@app.get("/reports/chat/{session_id}")
async def download_chat_report(session_id: str):
    """Generate and download a PDF summary of a chat session."""
    history = _chat_sessions.get(session_id, [])
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No chat history found for session '{session_id}'",
        )
    sources  = vector_store_manager.list_sources()
    pdf_bytes = generate_chat_summary_pdf(
        chat_history=history,
        document_sources=sources,
        title=f"RAG Chat Summary — Session: {session_id}",
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="chat_summary_{session_id}.pdf"'},
    )


@app.get("/reports/sources")
async def download_sources_report():
    """Generate and download a PDF listing all indexed sources."""
    sources  = vector_store_manager.list_sources()
    pdf_bytes = generate_chat_summary_pdf(
        chat_history=[],
        document_sources=sources,
        title="Indexed Documents Report",
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="indexed_sources.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
