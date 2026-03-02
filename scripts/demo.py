"""
Streamlit chatbot UI for the PDF & Turtle RAG Chatbot.

Run:
    streamlit run scripts/demo.py
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# ── Resolve project root and fix relative paths to absolute ───────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.rag.loaders import load_document
from src.rag.vector_store import vector_store_manager
from src.rag.chain import rag_chain
from src.utils.report_generator import generate_chat_summary_pdf
from src.config import settings

# Patch relative paths → absolute so they work regardless of cwd
def _abs(p: str) -> str:
    return str(PROJECT_ROOT / p) if not Path(p).is_absolute() else p

settings.upload_dir        = _abs(settings.upload_dir)
settings.chroma_persist_dir = _abs(settings.chroma_persist_dir)
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PDF & Turtle RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/agcdke/toadlet-llm",
        "Report a bug": "https://github.com/agcdke/toadlet-llm/issues",
        "About": "PDF & Turtle RDF RAG Chatbot — LangChain + ChromaDB + Ollama",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background:#0d1117; }
[data-testid="stSidebar"]          { background:#161b22; border-right:1px solid #21262d; }
[data-testid="stSidebar"] *        { color:#c9d1d9 !important; }
.block-container                   { padding-top:1.4rem; padding-bottom:2rem; }

/* ── Top banner ──────────────────────────────────────────────── */
.banner {
    background:linear-gradient(135deg,#0d2137 0%,#1a1035 60%,#0d2137 100%);
    border:1px solid #30363d; border-radius:14px;
    padding:20px 30px; margin-bottom:16px;
    display:flex; align-items:center; gap:16px;
}
.banner-icon  { font-size:2.4rem; line-height:1; }
.banner-title { color:#e6edf3; font-size:1.65rem; font-weight:800; margin:0; }
.banner-sub   { color:#8b949e; font-size:0.87rem; margin:4px 0 0; }

/* ── Stat cards ──────────────────────────────────────────────── */
.stat-card {
    background:#161b22; border:1px solid #21262d; border-radius:12px;
    padding:14px 8px; text-align:center; transition:border-color .2s;
}
.stat-card:hover { border-color:#388bfd; }
.stat-num   { font-size:1.75rem; font-weight:700; color:#388bfd; line-height:1; }
.stat-label { font-size:0.68rem; color:#6e7681; margin-top:4px;
              text-transform:uppercase; letter-spacing:.06em; }

/* ── Chat bubbles ────────────────────────────────────────────── */
.msg-wrap-user { display:flex; justify-content:flex-end; margin:5px 0; }
.msg-wrap-bot  { display:flex; justify-content:flex-start; margin:5px 0; }
.bubble-user {
    background:linear-gradient(135deg,#1f6feb,#1158c7);
    color:#e6edf3; border-radius:18px 18px 4px 18px;
    padding:12px 18px; max-width:76%;
    font-size:0.93rem; line-height:1.65;
    box-shadow:0 2px 10px rgba(31,111,235,.35);
}
.bubble-bot {
    background:#161b22; color:#c9d1d9;
    border-radius:18px 18px 18px 4px;
    padding:12px 18px; max-width:82%;
    border-left:3px solid #58a6ff;
    font-size:0.93rem; line-height:1.70;
    box-shadow:0 2px 10px rgba(0,0,0,.4);
}
.sender-tag {
    font-size:0.67rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.08em; opacity:.50; margin-bottom:5px;
}
.ts-tag { font-size:0.65rem; color:#6e7681; text-align:right; margin-top:5px; }

/* ── Source chips ─────────────────────────────────────────────── */
.chips { display:flex; flex-wrap:wrap; gap:5px; margin-top:9px; }
.chip  {
    border-radius:20px; padding:2px 10px; font-size:0.71rem;
    border:1px solid #30363d; background:#21262d; color:#8b949e;
    display:inline-flex; align-items:center; gap:4px;
}
.chip-pdf   { border-color:#da3633; color:#ff7b72; background:#3d1a1a; }
.chip-ttl   { border-color:#2ea043; color:#56d364; background:#122118; }
.chip-table { border-color:#bb8009; color:#e3b341; background:#2d1f00; }
.chip-page  { border-color:#388bfd; color:#79c0ff; background:#0d1f3c; }

/* ── Empty state ──────────────────────────────────────────────── */
.empty-state {
    text-align:center; padding:64px 24px; color:#484f58;
}
.empty-state .icon  { font-size:3.5rem; margin-bottom:14px; }
.empty-state .title { font-size:1.1rem; font-weight:600; color:#6e7681; margin-bottom:6px; }
.empty-state .sub   { font-size:0.88rem; line-height:1.7; }

/* ── Thinking dots ────────────────────────────────────────────── */
@keyframes pulse { 0%,100%{opacity:.35} 50%{opacity:1} }
.dot {
    display:inline-block; width:7px; height:7px; border-radius:50%;
    background:#388bfd; margin:0 2px;
    animation:pulse 1.2s infinite;
}
.dot:nth-child(2) { animation-delay:.2s; }
.dot:nth-child(3) { animation-delay:.4s; }
.thinking-wrap {
    padding:12px 0; display:flex; align-items:center; gap:8px;
}
.thinking-label { font-size:0.81rem; color:#388bfd; }

/* ── Source detail card ───────────────────────────────────────── */
.src-card {
    background:#0d1117; border:1px solid #21262d; border-radius:8px;
    padding:10px 14px; margin-bottom:8px; font-size:0.80rem; color:#8b949e;
    font-family:monospace; white-space:pre-wrap; overflow-x:auto;
}

/* ── Input ────────────────────────────────────────────────────── */
[data-testid="stChatInputTextArea"] {
    background:#161b22 !important; border:1px solid #30363d !important;
    border-radius:12px !important; color:#e6edf3 !important;
    font-size:0.92rem !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color:#388bfd !important;
    box-shadow:0 0 0 2px rgba(56,139,253,.2) !important;
}

/* ── File uploader ────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background:#161b22; border:2px dashed #30363d; border-radius:10px; padding:6px;
}
[data-testid="stFileUploader"]:hover { border-color:#388bfd; }

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    background:#21262d; border:1px solid #30363d; border-radius:8px;
    color:#c9d1d9; width:100%; transition:all .15s;
}
.stButton > button:hover { border-color:#388bfd; color:#58a6ff; background:#161b22; }

/* ── Download button ──────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background:linear-gradient(135deg,#0d2137,#1a1035);
    border:1px solid #388bfd; color:#79c0ff;
    border-radius:8px; width:100%;
}
[data-testid="stDownloadButton"] > button:hover {
    background:#0d1f3c; box-shadow:0 0 8px rgba(56,139,253,.4);
}

/* ── Expander ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
}
summary { color:#8b949e !important; }

/* ── Misc ─────────────────────────────────────────────────────── */
hr   { border-color:#21262d !important; }
code { background:#21262d !important; border-radius:4px !important;
       color:#79c0ff !important; }
.stSuccess { background:#122118 !important; border-radius:8px !important; }
.stWarning { background:#2d1f00 !important; border-radius:8px !important; }
.stError   { background:#3d1a1a !important; border-radius:8px !important; }
.stInfo    { background:#0d1f3c !important; border-radius:8px !important; }
[data-testid="stSlider"] { accent-color:#388bfd; }

/* ── Sidebar section label ────────────────────────────────────── */
.sec {
    font-size:0.70rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.10em; color:#6e7681; margin:14px 0 7px;
}

/* ── Indexed-source row ───────────────────────────────────────── */
.src-row {
    background:#0d1117; border:1px solid #21262d; border-radius:8px;
    padding:8px 12px; margin-bottom:5px;
}
.src-row-name { font-size:0.84rem; color:#c9d1d9; }
.src-row-meta { font-size:0.69rem; color:#6e7681; margin-top:2px; }

/* ── Stack table ──────────────────────────────────────────────── */
.stack-row {
    display:flex; align-items:center; gap:8px; padding:4px 0;
    border-bottom:1px solid #21262d; font-size:0.80rem;
}
.stack-name { color:#c9d1d9; font-weight:600; }
.stack-desc { color:#6e7681; margin-left:auto; font-size:0.74rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "messages":        [],    # list[dict]: role, content, sources, ts
    "top_k":           5,
    "stats":           {},
    "show_sources":    True,
    "show_timestamps": True,
    "typing":          False,
    "uploader_key":    0,     # incremented to reset the file uploader widget
    "llm_model":       settings.ollama_model,
    "embed_model":     settings.embedding_model,
    "sparql_mode":     False,  # when True, route TTL queries through SPARQL engine
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP RESTORE — runs once per Streamlit server process
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _restore_on_startup():
    """
    Called once when the Streamlit server process starts.
    st.cache_resource persists across user sessions and page reloads,
    so this runs exactly once per server lifetime — not on every rerun.

    Does two things:
      1. Registers all .ttl files in data/uploads/ into the SPARQL engine
         so SPARQL mode works immediately without re-uploading.
      2. The ChromaDB vector store is already persisted to disk — the
         VectorStoreManager lazy-loads it automatically on first access,
         so no explicit restore is needed for RAG queries.
    """
    from src.rag.sparql_engine import get_sparql_engine

    upload_dir = Path(settings.upload_dir)
    restored = []
    if upload_dir.exists():
        for ttl_file in upload_dir.glob("*.ttl"):
            try:
                get_sparql_engine(str(ttl_file))
                restored.append(ttl_file.name)
            except Exception as e:
                pass
        for ttl_file in upload_dir.glob("*.turtle"):
            try:
                get_sparql_engine(str(ttl_file))
                restored.append(ttl_file.name)
            except Exception as e:
                pass
    return restored

_restored_ttl = _restore_on_startup()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def refresh_stats() -> dict:
    """Fetch fresh ChromaDB stats and cache in session state."""
    try:
        stats = vector_store_manager.collection_stats()
    except Exception:
        stats = {}
    st.session_state["stats"] = stats
    return stats


def ingest_file(uploaded_file) -> dict:
    """
    Save file permanently to data/uploads/, load it, ingest into ChromaDB.

    Fix: previously used tempfile which deleted the file immediately after
    ingestion. Files are now saved to settings.upload_dir so they persist
    between sessions and can be re-ingested without re-uploading.
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    save_path = upload_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        text_docs, table_docs = load_document(str(save_path))
        for doc in text_docs + table_docs:
            doc.metadata["file_name"] = uploaded_file.name
        text_n, table_n = vector_store_manager.ingest_documents(
            documents=text_docs,
            table_documents=table_docs if table_docs else None,
        )
        return {
            "ok":         True,
            "file_name":  uploaded_file.name,
            "saved_path": str(save_path),
            "text_pages": len(text_docs),
            "table_pages": len(table_docs),
            "text_chunks": text_n,
            "table_chunks": table_n,
        }
    except Exception as exc:
        # Remove the saved file if ingestion failed to avoid stale files
        if save_path.exists():
            save_path.unlink()
        return {"ok": False, "file_name": uploaded_file.name, "error": str(exc)}


def chip_html(label: str, kind: str = "") -> str:
    """Return an HTML chip span."""
    css = {"pdf":"chip chip-pdf","ttl":"chip chip-ttl","turtle":"chip chip-ttl",
           "table":"chip chip-table","page":"chip chip-page"}.get(kind, "chip")
    icon = {"pdf":"📄","ttl":"🐢","turtle":"🐢","table":"📊","page":"🔖"}.get(kind, "")
    return f'<span class="{css}">{icon} {label}</span>'


def render_message(msg: dict):
    """Render one chat message bubble."""
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])
    ts      = msg.get("ts", "")
    ts_html = (f'<div class="ts-tag">{ts}</div>'
               if st.session_state["show_timestamps"] and ts else "")

    if role == "user":
        st.markdown(
            f'<div class="msg-wrap-user">'
            f'  <div class="bubble-user">'
            f'    <div class="sender-tag">You</div>'
            f'    {content}{ts_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # Source chips
        chips_html = ""
        if st.session_state["show_sources"] and sources:
            seen_files: dict[str, str] = {}
            seen_pages: set = set()
            for s in sources:
                meta  = s.get("metadata", {})
                fname = meta.get("file_name", "?")
                ctype = meta.get("chunk_type", meta.get("source_type", ""))
                page  = meta.get("page")
                seen_files[fname] = ctype
                if page:
                    seen_pages.add((fname, page))
            chips = "".join(chip_html(fn, ct) for fn, ct in seen_files.items())
            chips += "".join(chip_html(f"p.{pg}", "page") for _, pg in sorted(seen_pages))
            chips_html = f'<div class="chips">{chips}</div>'

        # Guardrail badge
        guard_blocked = msg.get("guardrail_blocked", False)
        guard_code    = msg.get("guardrail_code", "ok")
        if guard_blocked:
            badge_html = (
                f'<div class="guardrail-badge">'
                f'⚠️ Blocked by guardrail &nbsp;·&nbsp; <code>{guard_code}</code>'
                f'</div>'
            )
        else:
            badge_html = ""

        st.markdown(
            f'<div class="msg-wrap-bot">'
            f'  <div class="bubble-bot">'
            f'    <div class="sender-tag">🤖 Assistant</div>'
            f'    {content}'
            f'    {chips_html}'
            f'    {badge_html}'
            f'    {ts_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Show generated SPARQL in an expander if present
        sparql_query = msg.get("sparql", "")
        if sparql_query:
            sparql_rows = msg.get("sparql_rows", 0)
            with st.expander(f"🔎 Generated SPARQL  ·  {sparql_rows} result(s)"):
                st.code(sparql_query, language="sparql")


def build_report_bytes() -> bytes | None:
    """Build Q-A pairs for PDF report generation."""
    msgs = st.session_state["messages"]
    pairs = [
        {"question": msgs[i-1]["content"],
         "answer":   m["content"],
         "sources":  m.get("sources", [])}
        for i, m in enumerate(msgs)
        if m["role"] == "assistant" and i > 0 and msgs[i-1]["role"] == "user"
    ]
    if not pairs:
        return None
    return generate_chat_summary_pdf(
        chat_history=pairs,
        document_sources=vector_store_manager.list_sources(),
        title="RAG Chat Summary Report",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand + New Chat button
    col_brand, col_new = st.columns([3, 1])
    with col_brand:
        st.markdown(
            '<p style="font-size:1.05rem;font-weight:800;color:#e6edf3;margin:0">🤖 RAG Chatbot</p>'
            '<p style="font-size:0.73rem;color:#6e7681;margin:2px 0 0">PDF &amp; Turtle RDF</p>',
            unsafe_allow_html=True,
        )
    with col_new:
        st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
        if st.button(
            "🗨️ New Chat",
            key="new_chat_btn",
            help="Clear current conversation and start fresh",
            use_container_width=True,
        ):
            st.session_state["messages"] = []
            st.session_state["typing"]   = False
            st.toast("New chat started.", icon="🗨️")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    # ── Upload ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="sec">📂 Upload Documents</p>', unsafe_allow_html=True)
    st.caption("Accepts `.pdf`, `.ttl`, `.turtle`")

    uploaded_files = st.file_uploader(
        "Drop files",
        type=["pdf", "ttl", "turtle"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state['uploader_key']}",
    )

    col_ingest, col_clear = st.columns(2)

    with col_ingest:
        ingest_clicked = st.button(
            "⬆️  Ingest files",
            disabled=not uploaded_files,
            help="Embed the uploaded files into ChromaDB",
            use_container_width=True,
        )

    with col_clear:
        clear_clicked = st.button(
            "✖  Clear selection",
            disabled=not uploaded_files,
            help="Remove selected files without ingesting them",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state["uploader_key"] += 1
        st.rerun()

    if ingest_clicked:
        prog = st.progress(0)
        total = len(uploaded_files)
        for idx, uf in enumerate(uploaded_files):
            with st.spinner(f"Processing **{uf.name}**…"):
                res = ingest_file(uf)
            prog.progress((idx + 1) / total)
            if res["ok"]:
                st.success(
                    f"✅ **{res['file_name']}**\n"
                    f"- Text:  {res['text_pages']} pages → **{res['text_chunks']}** chunks\n"
                    f"- Table: {res['table_pages']} pages → **{res['table_chunks']}** chunks"
                )
            else:
                st.error(f"❌ **{res['file_name']}**: {res['error']}")
        prog.empty()
        refresh_stats()
        # Auto-reset the uploader after ingestion
        st.session_state["uploader_key"] += 1
        st.rerun()

    st.divider()

    # ── Indexed sources ────────────────────────────────────────────────────────
    c_hdr, c_refresh, c_reset = st.columns([3, 1, 1])
    with c_hdr:
        st.markdown('<p class="sec" style="margin-top:0">📚 Indexed Sources</p>',
                    unsafe_allow_html=True)
    with c_refresh:
        if st.button("🔄", key="refresh_btn", help="Refresh source list"):
            refresh_stats()
    with c_reset:
        if st.button("🗑️", key="reset_vs_btn", help="Reset vector store — removes all ingested documents"):
            vector_store_manager.delete_collection()
            st.session_state["stats"] = {}
            st.session_state["uploader_key"] += 1
            st.toast("Vector store reset.", icon="🗑️")
            st.rerun()

    sources_list = vector_store_manager.list_sources()
    if sources_list:
        rows_html = "".join(
            f'<div class="src-row">'            f'  <div class="src-row-name">{"📄" if src.get("source_type") == "pdf" else "🐢"} {src["file_name"]}</div>'            f'  <div class="src-row-meta">text: {src.get("text_chunks", 0)} &nbsp;|&nbsp; table: {src.get("table_chunks", 0)}</div>'            f'</div>'
            for src in sources_list
        )
        st.markdown(
            f'<div style="max-height:180px;overflow-y:auto;padding-right:4px;">'            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="font-size:0.82rem;color:#484f58;font-style:italic">'            'No documents indexed yet.</p>',
            unsafe_allow_html=True,
        )


    st.divider()

    # ── Retrieval settings ─────────────────────────────────────────────────────
    st.markdown('<p class="sec">⚙️ Retrieval Settings</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="sec">🔍 Query Mode</p>', unsafe_allow_html=True)
    st.session_state["sparql_mode"] = st.toggle(
        "SPARQL mode (for Turtle files)",
        value=st.session_state["sparql_mode"],
        help=(
            "When ON: natural language is translated to SPARQL and run "
            "directly on the RDF graph — exact structured answers.\n\n"
            "When OFF: standard RAG similarity search — good for PDF and "
            "general questions, but imprecise for graph relationship queries."
        ),
    )
    if st.session_state["sparql_mode"]:
        st.caption("🐢 Querying RDF graph directly via SPARQL")
    else:
        st.caption("📄 Using vector similarity search (RAG)")

    st.session_state["top_k"] = st.slider(
        "Top-K chunks", min_value=1, max_value=20,
        value=st.session_state["top_k"], step=1,
        help="More chunks = broader context but slower response",
    )
    st.session_state["show_sources"] = st.toggle(
        "Show source chips", value=st.session_state["show_sources"],
    )
    st.session_state["show_timestamps"] = st.toggle(
        "Show timestamps", value=st.session_state["show_timestamps"],
    )

    st.divider()



    st.divider()

    # ── Danger zone ────────────────────────────────────────────────────────────
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.caption("These actions cannot be undone.")
        if st.button("🗑️ Clear vector store", key="clr_vs"):
            vector_store_manager.delete_collection()
            st.session_state["stats"] = {}
            st.warning("Vector store cleared.")
            st.rerun()
        if st.button("🧹 Clear chat history", key="clr_ch"):
            st.session_state["messages"] = []
            st.rerun()
        if st.button("🗂️ Delete uploaded files from disk", key="clr_files"):
            upload_dir = Path(settings.upload_dir)
            deleted = []
            for f in upload_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {".pdf", ".ttl", ".turtle"}:
                    f.unlink()
                    deleted.append(f.name)
            if deleted:
                st.session_state["uploader_key"] += 1
                st.warning(f"Deleted {len(deleted)} file(s) from disk.")
            else:
                st.info("No files found in uploads folder.")
            st.rerun()
        if st.button("♻️ Reset everything", key="clr_all"):
            vector_store_manager.delete_collection()
            st.session_state["messages"] = []
            st.session_state["stats"] = {}
            # Delete all uploaded files from disk
            upload_dir = Path(settings.upload_dir)
            for f in upload_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {".pdf", ".ttl", ".turtle"}:
                    f.unlink()
            st.session_state["uploader_key"] += 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Banner ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="banner">'
    '  <div class="banner-icon">🤖</div>'
    '  <div>'
    '    <p class="banner-title">PDF &amp; Turtle RAG Chatbot</p>'
    '    <p class="banner-sub">'
    '      Chat with your documents — fully local · zero API costs &nbsp;·&nbsp;'
    '      LangChain · ChromaDB · Ollama · pdfplumber · rdflib'
    '    </p>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Stats row ──────────────────────────────────────────────────────────────────
stats = refresh_stats()

sc1, sc2, sc3, sc4, sc5 = st.columns(5)
for col, num, label in [
    (sc1, stats.get("unique_sources", 0),              "Documents"),
    (sc2, stats.get("total_chunks",   0),              "Total Chunks"),
    (sc3, stats.get("text_chunks",    0),              "Text Chunks"),
    (sc4, stats.get("table_chunks",   0),              "Table Chunks"),
    (sc5, len(st.session_state["messages"]),           "Messages"),
]:
    col.markdown(
        f'<div class="stat-card">'
        f'  <div class="stat-num">{num}</div>'
        f'  <div class="stat-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Two-column layout ──────────────────────────────────────────────────────────
chat_col, meta_col = st.columns([3, 1], gap="large")

# ── CHAT COLUMN ───────────────────────────────────────────────────────────────
with chat_col:
    msgs = st.session_state["messages"]

    # Empty state
    if not msgs:
        st.markdown(
            '<div class="empty-state">'
            '  <div class="icon">💬</div>'
            '  <div class="title">Start a conversation</div>'
            '  <div class="sub">'
            '    1. Upload a <b>PDF</b> or <b>Turtle (.ttl)</b> file via the sidebar<br>'
            '    2. Click <b>Ingest files</b><br>'
            '    3. Type your question in the box below'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in msgs:
            render_message(msg)
            st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # Thinking animation
    if st.session_state.get("typing"):
        st.markdown(
            '<div class="msg-wrap-bot">'
            '  <div class="bubble-bot">'
            '    <div class="thinking-wrap">'
            '      <span class="thinking-label">Thinking</span>'
            '      <span class="dot"></span>'
            '      <span class="dot"></span>'
            '      <span class="dot"></span>'
            '    </div>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Source detail expander ────────────────────────────────────────────────
    last_bot = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
    if last_bot and last_bot.get("sources"):
        n = len(last_bot["sources"])
        with st.expander(f"📄 {n} source chunk{'s' if n > 1 else ''} used in last answer"):
            for i, src in enumerate(last_bot["sources"], 1):
                meta = src.get("metadata", {})
                st.markdown(
                    f"**Chunk {i}** &nbsp;—&nbsp; "
                    f"`{meta.get('file_name','?')}` &nbsp;|&nbsp; "
                    f"page `{meta.get('page','?')}` &nbsp;|&nbsp; "
                    f"type `{meta.get('chunk_type','?')}`"
                )
                st.markdown(
                    f'<div class="src-card">{src.get("content_preview","")[:500]}</div>',
                    unsafe_allow_html=True,
                )
                if i < n:
                    st.divider()

    # ── Download report ───────────────────────────────────────────────────────
    if msgs:
        pdf_bytes = build_report_bytes()
        if pdf_bytes:
            st.download_button(
                label="⬇️  Download chat summary as PDF",
                data=pdf_bytes,
                file_name=f"chat_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Save this conversation and source references as a formatted PDF",
            )

# ── META PANEL ────────────────────────────────────────────────────────────────
with meta_col:
    st.markdown("#### 📘 Quick Guide")
    st.markdown(
        '<div style="font-size:0.81rem;color:#8b949e;line-height:2.1">'
        '1️⃣ &nbsp;Upload a <b>.pdf</b> or <b>.ttl</b> file<br>'
        '2️⃣ &nbsp;Click <b>Ingest files</b><br>'
        '3️⃣ &nbsp;Type your question below<br>'
        '4️⃣ &nbsp;View sources in the expander<br>'
        '5️⃣ &nbsp;Export a <b>PDF report</b>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### 🏷️ Metadata")
    legend = [
        ("pdf",   "file.pdf",  "PDF document"),
        ("ttl",   "file.ttl",  "Turtle RDF"),
        ("table", "table",     "Table chunk"),
        ("page",  "p.3",       "Page number"),
    ]
    for kind, label, desc in legend:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:3px 0;border-bottom:1px solid #21262d">'
            f'  {chip_html(label, kind)}'
            f'  <span style="font-size:0.75rem;color:#6e7681">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("#### 🛠️ Tech Stack")
    STACK = [
        ("🦙", "Ollama",     "LLM"),
        ("📐", "Embed",      "Vectors"),
        ("🗄️", "ChromaDB",   "Store"),
        ("🔗", "LangChain",  "RAG"),
        ("📑", "pdfplumber", "PDF"),
        ("🐢", "rdflib",     "TTL"),
        ("📊", "ReportLab",  "Reports"),
    ]
    rows = "".join(
        f'<div class="stack-row">'
        f'  <span>{ico}</span>'
        f'  <span class="stack-name">{name}</span>'
        f'  <span class="stack-desc">{desc}</span>'
        f'</div>'
        for ico, name, desc in STACK
    )
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #21262d;'
        f'border-radius:10px;padding:10px 14px">{rows}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT
# ══════════════════════════════════════════════════════════════════════════════
no_docs = stats.get("total_chunks", 0) == 0

if no_docs:
    st.info(
        "📂 **No documents indexed.**  "
        "Upload a PDF or Turtle file in the sidebar and click **Ingest files**.",
        icon="⚠️",
    )

prompt = st.chat_input(
    placeholder="Ask anything about your documents…",
    disabled=no_docs,
)

# ── On new user message: store it and trigger a rerun that shows the animation
if prompt:
    st.session_state["messages"].append({
        "role":    "user",
        "content": prompt,
        "sources": [],
        "ts":      datetime.now().strftime("%H:%M"),
    })
    st.session_state["typing"] = True
    st.rerun()

# ── On rerun with typing=True: call the RAG chain and store the answer
if st.session_state["typing"]:
    msgs = st.session_state["messages"]
    if msgs and msgs[-1]["role"] == "user":
        question = msgs[-1]["content"]
        try:
            if st.session_state["sparql_mode"]:
                # ── SPARQL path ────────────────────────────────────────────
                from src.rag.sparql_engine import get_all_ttl_engines
                engines = get_all_ttl_engines()
                if not engines:
                    answer  = (
                        "⚠️ No Turtle files are loaded in the SPARQL engine.\n\n"
                        "Please upload and ingest a `.ttl` file first, "
                        "then ask your question again."
                    )
                    sources = []
                    result  = {"guardrail_blocked": False, "guardrail_code": "ok"}
                else:
                    # Use the most recently loaded TTL engine
                    engine      = list(engines.values())[-1]
                    sparql_result = engine.answer(question)
                    answer      = sparql_result["answer"]
                    sources     = []
                    result      = {
                        "guardrail_blocked": False,
                        "guardrail_code":    "ok",
                        "sparql":            sparql_result.get("sparql", ""),
                        "sparql_rows":       len(sparql_result.get("rows", [])),
                        "sparql_success":    sparql_result.get("success", False),
                    }
            else:
                # ── RAG path ───────────────────────────────────────────────
                result  = rag_chain.query_with_context(question, k=st.session_state["top_k"])
                answer  = result["answer"]
                sources = result.get("sources", [])
        except Exception as exc:
            answer  = (
                f"⚠️ **Error**: `{exc}`\n\n"
                "Make sure Ollama is running: `ollama serve`"
            )
            sources = []
            result  = {"guardrail_blocked": False, "guardrail_code": "ok"}

        st.session_state["messages"].append({
            "role":              "assistant",
            "content":           answer,
            "sources":           sources,
            "ts":                datetime.now().strftime("%H:%M"),
            "guardrail_blocked": result.get("guardrail_blocked", False),
            "guardrail_code":    result.get("guardrail_code", "ok"),
            "sparql":            result.get("sparql", ""),
            "sparql_rows":       result.get("sparql_rows", 0),
        })
    st.session_state["typing"] = False
    st.rerun()