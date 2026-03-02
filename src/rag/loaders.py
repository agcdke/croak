"""
Document loaders for PDF (.pdf) and Turtle RDF (.ttl) files.

Adapted from reference implementation (agcdke/toadlet-llm):
  - remove_unicode()            strips non-ASCII characters
  - convert_doctable_to_mdtext() fallback: table → Markdown text
  - extract_pdf_text_tables()   returns (joined_text: str, table_list: list[list[dict]])
  - PDFLoader.load()            wraps the above into LangChain Documents
  - TurtleLoader.load()         returns (text_docs, []) for API consistency
  - load_document()             unified entry point auto-detecting file type
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
import pandas as pd
from rdflib import Graph
from langchain_core.documents import Document
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (direct ports from reference)
# ══════════════════════════════════════════════════════════════════════════════

def remove_unicode(text: str) -> str:
    """
    Remove unicode characters.
    Mirrors reference remove_unicode() exactly.
    """
    return re.sub(r'[^\x00-\x7F]+', '', text)


def convert_doctable_to_mdtext(page) -> Optional[str]:
    """
    Convert document table to markdown text format.
    Used as fallback when JSON serialisation fails.
    Mirrors reference convert_doctable_to_mdtext() exactly.

    page: pdfplumber page object
    """
    for table in page.find_tables():
        df = pd.DataFrame(table.extract())
        df.columns = df.iloc[0]
        markdown_df = df.drop(0).to_markdown(index=False)
        return markdown_df
    return None


def extract_pdf_text_tables(
    dir_path: str,
    filename: str,
) -> Tuple[Optional[str], List[List[dict]]]:
    """
    Extract text and tables from a PDF file.
    Mirrors reference extract_pdf_text_tables() exactly.

    Args:
        dir_path: directory path containing the PDF file
        filename: PDF filename

    Returns:
        joined_text : str  — all page text joined with spaces, or None
        table_list  : list[list[dict]] — one inner list per page that has tables;
                      each inner list holds one dict per table row
    """
    text_list  = []
    table_list = []

    pdf_file = os.path.join(dir_path, filename)
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            # ── Text ──────────────────────────────────────────────────────
            if page_text:
                text_list.append(page_text)

            # ── Tables ────────────────────────────────────────────────────
            table = page.extract_tables()
            if table:
                # Assume first row is header (UNECE STANDARD pattern from reference)
                table_df = pd.DataFrame(table[0][1:], columns=table[0][0])
                table_df = table_df.reset_index(drop=True)
                try:
                    table_doc_list = [
                        json.loads(
                            remove_unicode(
                                row.to_json(force_ascii=False, orient="index")
                            )
                        )
                        for _, row in table_df.iterrows()
                    ]
                    table_list.append(table_doc_list)

                except ValueError as te:
                    logger.warning(
                        f"Convert doc-table to markdown-text format as "
                        f"ValueError occurred: {te}"
                    )
                    markdown_df = convert_doctable_to_mdtext(page=page)
                    if markdown_df:
                        text_list.append(markdown_df)

                except Exception as e:
                    logging.info(f"An exception occurred: {str(e)}")

    joined_text = " ".join(text_list) if text_list else None
    if not table_list:
        table_list = []

    return joined_text, table_list


# ══════════════════════════════════════════════════════════════════════════════
# PDF LOADER
# ══════════════════════════════════════════════════════════════════════════════

class PDFLoader:
    """
    Loads PDF documents using pdfplumber.

    Wraps extract_pdf_text_tables() (ported from reference) into LangChain
    Document objects so the rest of the pipeline (VectorStoreManager) can
    consume them without knowing about the raw extraction details.

    Returns:
        text_docs  : list[Document]  — one Document containing all joined page text
        table_docs : list[Document]  — one Document per page that had tables,
                                       page_content = JSON-encoded list of row dicts
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.dir_path  = str(Path(file_path).parent)

    def load(self) -> Tuple[List[Document], List[Document]]:
        logger.info(f"Loading PDF: {self.file_path}")

        joined_text, table_list = extract_pdf_text_tables(
            dir_path=self.dir_path,
            filename=self.file_name,
        )

        base_meta = {
            "source":      self.file_path,
            "source_type": "pdf",
            "file_name":   self.file_name,
        }

        # ── Text Document ──────────────────────────────────────────────────
        text_docs: List[Document] = []
        if joined_text:
            text_docs.append(
                Document(
                    page_content=joined_text,
                    metadata={**base_meta, "chunk_type": "text"},
                )
            )

        # ── Table Documents ────────────────────────────────────────────────
        # table_list is list[list[dict]] — one inner list per page with tables.
        # Store each page's list of row-dicts as a JSON string in one Document.
        # VectorStoreManager.split_table_documents() will call split_table_to_json()
        # on each Document's page_content to produce final ChromaDB chunks.
        table_docs: List[Document] = []
        for page_idx, row_dicts in enumerate(table_list):
            if row_dicts:
                table_docs.append(
                    Document(
                        page_content=json.dumps(row_dicts),
                        metadata={
                            **base_meta,
                            "chunk_type": "table",
                            "table_page": page_idx,
                            "row_count":  len(row_dicts),
                        },
                    )
                )

        logger.info(
            f"PDF loaded — text_docs={len(text_docs)}, "
            f"table_docs={len(table_docs)} from '{self.file_name}'"
        )
        return text_docs, table_docs


# ══════════════════════════════════════════════════════════════════════════════
# TURTLE LOADER
# ══════════════════════════════════════════════════════════════════════════════

class TurtleLoader:
    """
    Loads Turtle RDF (.ttl) files and converts triples to human-readable
    text Documents suitable for embedding.

    Key fix — why Turtle retrieval previously failed:
      Raw URIs like "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" are
      opaque to embedding models. nomic-embed-text and similar models encode
      them as near-random vectors with no semantic meaning, so similarity
      search never finds relevant chunks.

    Solution — two-stage conversion:
      1. _uri_to_label()  strips namespace prefixes from URIs and converts
         CamelCase / snake_case local names to plain English words.
         e.g. "http://schema.org/Person"  →  "Person"
              "http://schema.org/birthDate" →  "birth date"
      2. _subject_to_sentences() groups all triples about the same subject
         into one natural-language paragraph:
         "Person John Smith. Has birth date 1990-01-01. Has email john@x.com."
         This is far more embeddable than 50 disconnected raw triples.

    Chunking:
      Groups sentences by subject (entity-centric chunks) rather than fixed
      50-triple windows. Each chunk describes one entity completely, so the
      retriever returns coherent facts rather than arbitrary slices.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name

    # ── URI → readable label ───────────────────────────────────────────────────

    @staticmethod
    def _uri_to_label(uri_ref) -> str:
        """
        Convert a URI or literal to a readable English label.

        URIRef  →  local name with CamelCase / snake_case split into words
        Literal →  its string value directly
        BNode   →  "entity"
        """
        from rdflib import URIRef, Literal, BNode

        if isinstance(uri_ref, Literal):
            return str(uri_ref).strip()

        if isinstance(uri_ref, BNode):
            return "entity"

        # URIRef: take the fragment (#name) or last path segment (/name)
        s = str(uri_ref)
        if "#" in s:
            local = s.rsplit("#", 1)[-1]
        elif "/" in s:
            local = s.rsplit("/", 1)[-1]
        else:
            local = s

        # CamelCase → "Camel Case"
        local = re.sub(r"([a-z])([A-Z])", r"\1 \2", local)
        # snake_case / kebab-case → spaces
        local = re.sub(r"[_\-]+", " ", local)
        return local.strip() or s

    def _triple_to_sentence(self, s, p, o) -> str:
        """
        Convert one (subject, predicate, object) triple to a readable sentence.
        e.g. "John Smith  has birth date  1990-01-01."
        """
        subj  = self._uri_to_label(s)
        pred  = self._uri_to_label(p).lower()
        obj   = self._uri_to_label(o)

        # Prettify common RDF predicates
        pred = (pred
                .replace("rdf type", "is a")
                .replace("type", "is a")
                .replace("label", "is called")
                .replace("comment", "is described as")
                .replace("subclass of", "is a subclass of")
                .replace("domain", "applies to")
                .replace("range", "has values of type"))

        return f"{subj} {pred} {obj}."

    def load(self) -> Tuple[List[Document], List[Document]]:
        logger.info(f"Loading Turtle file: {self.file_path}")

        from rdflib import URIRef, BNode
        from collections import defaultdict

        g = Graph()
        g.parse(self.file_path, format="turtle")

        # ── Group triples by subject (entity-centric chunking) ─────────────
        subject_sentences: dict = defaultdict(list)
        for s, p, o in g:
            sentence = remove_unicode(self._triple_to_sentence(s, p, o))
            subject_sentences[str(s)].append(sentence)

        # ── Build one Document per subject ─────────────────────────────────
        # If a subject has many predicates, split into chunks of 20 sentences
        # so no single chunk exceeds the embedding model's sweet spot.
        SENTENCES_PER_CHUNK = 20
        docs: List[Document] = []
        base_meta = {
            "source":      self.file_path,
            "source_type": "turtle",
            "file_name":   self.file_name,
            "chunk_type":  "text",
        }

        for subj_uri, sentences in subject_sentences.items():
            subj_label = self._uri_to_label(
                URIRef(subj_uri) if subj_uri.startswith("http") else BNode(subj_uri)
            )
            for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
                chunk_sentences = sentences[i: i + SENTENCES_PER_CHUNK]
                # Header line gives the embedding model clear subject context
                header  = f"Facts about {subj_label}:"
                content = header + "\n" + " ".join(chunk_sentences)
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            **base_meta,
                            "subject":       subj_label,
                            "subject_uri":   subj_uri,
                            "sentence_range": f"{i}-{i + len(chunk_sentences)}",
                        },
                    )
                )

        total_triples = sum(len(v) for v in subject_sentences.values())
        logger.info(
            f"TTL loaded — {total_triples} triples → "
            f"{len(docs)} entity chunks from '{self.file_name}'"
        )

        # Register this file in the SPARQL engine so it can answer
        # structured queries directly on the graph (bypasses RAG embedding)
        try:
            from src.rag.sparql_engine import get_sparql_engine
            get_sparql_engine(self.file_path)
            logger.info(f"SPARQL engine registered for: {self.file_name}")
        except Exception as e:
            logger.warning(f"SPARQL engine registration failed: {e}")

        return docs, []


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_document(file_path: str) -> Tuple[List[Document], List[Document]]:
    """
    Auto-detect file type and load the document.

    Returns:
        (text_documents, table_documents)
        Pass both directly to VectorStoreManager.ingest_documents().
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return PDFLoader(file_path).load()
    elif suffix in (".ttl", ".turtle"):
        return TurtleLoader(file_path).load()
    else:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. Supported: .pdf, .ttl, .turtle"
        )
