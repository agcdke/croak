# 📄🐢 PDF & Turtle RDF Chatbot

Croak - if a FROG (FROm Ground) communicates with a fully local, free RAG (Retrieval-Augmented Generation) chatbot that lets FROG chat with **PDF documents** and **Turtle RDF (`.ttl`) files** using:

| Component | Library | Cost |
|-----------|---------|------|
| RAG Framework | LangChain | Free |
| LLM | Ollama (llama3.2) | Free/Local |
| Embeddings | sentence-transformers | Free/Local |
| Vector Store | ChromaDB | Free |
| Backend API | FastAPI | Free |
| MCP Server | FastMCP | Free |
| PDF Reports | ReportLab | Free |

---

## 🏗️ Project Structure

```
pdf-turtle-chatbot/
├── src/
│   ├── config.py               # Settings via pydantic-settings
│   ├── api/
│   │   └── main.py             # FastAPI REST backend
│   ├── mcp/
│   │   └── server.py           # FastMCP server (tools exposed via MCP)
│   ├── rag/
│   │   ├── loaders.py          # PDF & Turtle file loaders
│   │   ├── vector_store.py     # ChromaDB manager
│   │   └── chain.py            # RAG chain (LangChain + Ollama)
│   └── utils/
│       └── report_generator.py # ReportLab PDF report generation
├── data/
│   ├── uploads/                # Uploaded documents
│   └── chroma_db/              # ChromaDB persistence
├── tests/
│   └── test_rag.py             # pytest tests
├── scripts/
│   ├── demo.py                 # CLI demo
│   ├── start_api.sh            # Start FastAPI
│   └── start_mcp.sh            # Start MCP server
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

Install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull llama3.2
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box with Ollama)
```

### 4. Start the FastAPI Backend

```bash
bash scripts/start_api.sh
# or
uvicorn src.api.main:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

### 5. Start the MCP Server (optional)

```bash
bash scripts/start_mcp.sh
# or
python -m src.mcp.server
```

---

## 📡 API Endpoints

### Document Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload & ingest a PDF or `.ttl` file |
| `GET` | `/documents/sources` | List all indexed documents |
| `DELETE` | `/documents/clear` | Remove all documents from vector store |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Ask a question (RAG) |
| `GET` | `/chat/history/{session_id}` | Get session history |
| `DELETE` | `/chat/history/{session_id}` | Clear session history |

### Reports
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/reports/chat/{session_id}` | Download PDF summary of chat session |
| `GET` | `/reports/sources` | Download PDF of all indexed sources |

---

## 🔌 MCP Tools

The FastMCP server exposes these tools:

| Tool | Description |
|------|-------------|
| `ingest_document(file_path)` | Ingest a PDF or TTL file |
| `ask_question(question, k)` | RAG query |
| `search_documents(query, k)` | Semantic search (no LLM generation) |
| `list_indexed_documents()` | List all sources in vector store |
| `generate_session_report(history, output_path)` | Generate PDF report |
| `clear_vector_store()` | Clear all indexed data |

---

## 💻 CLI Demo

```bash
# Ingest a file and enter interactive chat
python scripts/demo.py --file path/to/document.pdf

# Single query
python scripts/demo.py --file path/to/ontology.ttl --query "What entities are defined?"
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🔧 How It Works

```
User uploads PDF/TTL
        ↓
Loader (PyPDF / rdflib)
        ↓
RecursiveCharacterTextSplitter (chunks)
        ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
        ↓
ChromaDB (persist to disk)

User asks question
        ↓
Embed question → ChromaDB similarity search
        ↓
Top-k chunks retrieved
        ↓
Ollama LLM (llama3.2) generates answer
        ↓
Response + sources returned to user

(Optional) Generate PDF report via ReportLab
```

---

## 🌱 Inspired By

Project structure inspired by [agcdke/toadlet-llm](https://github.com/agcdke/toadlet-llm) — CI/CD for LLM services of agricultural applications.
