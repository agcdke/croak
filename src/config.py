"""Application configuration using pydantic-settings."""
import os
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

# Project root is always the directory containing src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # LLM (Ollama - free & local)
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", env="OLLAMA_MODEL")

    # Separate model for SPARQL generation (code-oriented model gives cleaner queries)
    sparql_model: str = Field(default="llama3", env="SPARQL_MODEL")

    # Embeddings
    embedding_model: str = Field(default="nomic-embed-text", env="EMBEDDING_MODEL")

    # ChromaDB — default is absolute path anchored to project root
    chroma_persist_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "chroma_db"),
        env="CHROMA_PERSIST_DIR",
    )
    chroma_collection_name: str = Field(default="documents", env="CHROMA_COLLECTION_NAME")

    # File Uploads — default is absolute path anchored to project root
    upload_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "uploads"),
        env="UPLOAD_DIR",
    )

    # FastAPI
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # MCP
    mcp_host: str = Field(default="0.0.0.0", env="MCP_HOST")
    mcp_port: int = Field(default=8001, env="MCP_PORT")

    # Chunking
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    num_gpu: int = Field(default=1, env="NUM_GPU")

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        """
        Resolve any relative paths (from .env overrides) to absolute,
        create directories, and set write permissions.
        """
        for attr in ("upload_dir", "chroma_persist_dir"):
            p = Path(getattr(self, attr))
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            p.mkdir(parents=True, exist_ok=True)
            os.chmod(str(p), 0o777)
            setattr(self, attr, str(p))


settings = Settings()
settings.ensure_dirs()