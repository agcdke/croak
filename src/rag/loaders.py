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
    Domain-aware Turtle RDF loader.

    Detects entity types in the graph and applies specialised chunking
    strategies per type so embedding quality is maximised for each domain.

    Supported entity types (auto-detected from graph content):
    ┌─────────────────────────┬────────────────────────────────────────────┐
    │ Type                    │ Strategy                                   │
    ├─────────────────────────┼────────────────────────────────────────────┤
    │ ex:InspectionReport     │ One structured prose paragraph per report  │
    │ (fred_synth_kg.ttl)     │ with all fields: date, product, supplier,  │
    │                         │ defect%, rejected, origin country          │
    ├─────────────────────────┼────────────────────────────────────────────┤
    │ skos:Concept (AGROVOC)  │ One chunk per concept: preferred English   │
    │                         │ label + all synonyms + hierarchy links     │
    ├─────────────────────────┼────────────────────────────────────────────┤
    │ Generic entity          │ Entity-centric chunks of ≤20 sentences,   │
    │ (any other TTL)         │ each prefixed with "Facts about X:"        │
    └─────────────────────────┴────────────────────────────────────────────┘

    All types also register the file in the SPARQL engine for exact
    structured queries that bypass RAG.
    """

    # Known namespace prefixes for readable label generation
    NS_PREFIXES = {
        "http://www.w3.org/2004/02/skos/core#":            "skos:",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#":     "rdf:",
        "http://www.w3.org/2000/01/rdf-schema#":           "rdfs:",
        "http://purl.org/dc/terms/":                       "dcterms:",
        "https://example.org/kg/":                         "ex:",
        "http://aims.fao.org/aos/agrovoc/":                "agrovoc:",
        "http://publications.europa.eu/resource/authority/country/": "",  # just use code
    }

    # EU country code → full name (for readable inspection report chunks)
    EU_COUNTRY_CODES = {
        "AUT":"Austria","BEL":"Belgium","BGR":"Bulgaria","HRV":"Croatia",
        "CYP":"Cyprus","CZE":"Czechia","DNK":"Denmark","EST":"Estonia",
        "FIN":"Finland","FRA":"France","DEU":"Germany","GRC":"Greece",
        "HUN":"Hungary","IRL":"Ireland","ITA":"Italy","LVA":"Latvia",
        "LTU":"Lithuania","LUX":"Luxembourg","MLT":"Malta","NLD":"Netherlands",
        "POL":"Poland","PRT":"Portugal","ROU":"Romania","SVK":"Slovakia",
        "SVN":"Slovenia","ESP":"Spain","SWE":"Sweden","GBR":"United Kingdom",
        "USA":"United States","CAN":"Canada","BRA":"Brazil","ARG":"Argentina",
        "MAR":"Morocco","ZAF":"South Africa","EGY":"Egypt","TUR":"Turkey",
        "ISR":"Israel","CHN":"China","JPN":"Japan","IND":"India",
        "MEX":"Mexico","COL":"Colombia","PER":"Peru","CHL":"Chile",
        "KEN":"Kenya","ETH":"Ethiopia","SEN":"Senegal","GHA":"Ghana",
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self._graph: Optional[Graph] = None

    # ── URI helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _uri_to_label(uri_ref) -> str:
        """URI/Literal → readable English string."""
        from rdflib import URIRef, Literal, BNode
        if isinstance(uri_ref, Literal):
            return str(uri_ref).strip()
        if isinstance(uri_ref, BNode):
            return "entity"
        s = str(uri_ref)
        local = s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]
        local = re.sub(r"([a-z])([A-Z])", r"\1 \2", local)
        local = re.sub(r"[_\-]+", " ", local)
        return local.strip() or s

    def _country_label(self, uri: str) -> str:
        """Convert EU country authority URI to full country name."""
        code = uri.rsplit("/", 1)[-1]
        return self.EU_COUNTRY_CODES.get(code, code)

    def _pref_label_en(self, g: Graph, uri) -> str:
        """Get the English prefLabel for a URI, falling back to URI local name."""
        from rdflib import URIRef, Literal
        from rdflib.namespace import SKOS
        for _, _, o in g.triples((uri, SKOS.prefLabel, None)):
            if isinstance(o, Literal) and o.language == "en" and str(o).strip():
                return str(o).strip()
        return self._uri_to_label(uri)

    # ── Domain detector ────────────────────────────────────────────────────────

    def _detect_domains(self, g: Graph) -> set:
        """Return set of domain names found in graph: 'inspection', 'skos', 'generic'."""
        from rdflib import URIRef
        from rdflib.namespace import RDF
        domains = set()
        inspection_type = URIRef("https://example.org/kg/InspectionReport")
        skos_concept    = URIRef("http://www.w3.org/2004/02/skos/core#Concept")
        for _, _, o in g.triples((None, RDF.type, None)):
            if o == inspection_type:
                domains.add("inspection")
            elif o == skos_concept:
                domains.add("skos")
        if not domains:
            domains.add("generic")
        return domains

    # ── Chunking strategies ────────────────────────────────────────────────────

    def _chunk_inspection_reports(self, g: Graph) -> List[Document]:
        """
        One Document per InspectionReport with all fields as a readable paragraph.

        Example output:
          Inspection Report 1000 | Date: 2023-02-16 | Product: Celery
          Supplier: Camposeven SAT | Packer: camposeven | Origin: Spain
          Batch: 308253 | Product ID: 58812 | Rejected: Yes | Borderline: No
          Defect: 74% — Minimum quality requirement "whole" not met
          Detail: with mechanical damage on stalks
        """
        from rdflib import URIRef, Literal
        from rdflib.namespace import RDF, RDFS, SKOS

        EX       = "https://example.org/kg/"
        DCTERMS  = "http://purl.org/dc/terms/"
        REPORT_T = URIRef(f"{EX}InspectionReport")

        # Build prefLabel lookup for AGROVOC product URIs
        product_labels: dict = {}
        for s, _, o in g.triples((None, SKOS.prefLabel, None)):
            if isinstance(o, Literal) and o.language == "en" and str(o).strip():
                uri = str(s)
                if uri not in product_labels:
                    product_labels[uri] = str(o).strip()

        docs: List[Document] = []
        base_meta = {
            "source":      self.file_path,
            "source_type": "turtle",
            "file_name":   self.file_name,
            "chunk_type":  "text",
            "entity_type": "InspectionReport",
        }

        def _val(g, subj, pred_local, ns=EX):
            """Get first string value for a predicate."""
            pred = URIRef(f"{ns}{pred_local}")
            for _, _, o in g.triples((subj, pred, None)):
                return str(o).strip()
            return ""

        for report_uri, _, _ in g.triples((None, RDF.type, REPORT_T)):
            report_id  = str(report_uri).rsplit("/", 1)[-1]
            label      = _val(g, report_uri, "label",   ns="http://www.w3.org/2000/01/rdf-schema#")
            date       = _val(g, report_uri, "date",    ns=DCTERMS)
            subject    = _val(g, report_uri, "subject", ns=DCTERMS)

            # Product: resolve AGROVOC URI to English label
            product_uri_node = next(
                (o for _, _, o in g.triples((report_uri, URIRef(f"{EX}hasProduct"), None))), None
            )
            product = ""
            if product_uri_node:
                product = product_labels.get(str(product_uri_node), "") or subject

            # Country: resolve EU authority URI to country name
            country_node = next(
                (o for _, _, o in g.triples((report_uri, URIRef(f"{EX}originCountry"), None))), None
            )
            country = self._country_label(str(country_node)) if country_node else ""

            supplier      = _val(g, report_uri, "supplierName")
            packer        = _val(g, report_uri, "packerName")
            batch         = _val(g, report_uri, "batchNumber")
            product_id    = _val(g, report_uri, "productId")
            rejected_raw  = _val(g, report_uri, "rejected")
            borderline_raw= _val(g, report_uri, "borderline")
            defect_pct    = _val(g, report_uri, "defectPercentageText")
            defect_pct_num= _val(g, report_uri, "defectPercentage")
            description   = _val(g, report_uri, "description")
            detail        = _val(g, report_uri, "detail")
            additional    = _val(g, report_uri, "additionalInfo")
            exact_desc    = _val(g, report_uri, "exactProductDescription")
            code          = _val(g, report_uri, "code")

            rejected   = "Yes" if rejected_raw.lower() == "true" else "No"
            borderline = "Yes" if borderline_raw.lower() == "true" else "No"

            # Build readable paragraph — every non-empty field included
            lines = [
                f"Inspection Report {report_id}",
                f"Date: {date}" if date else "",
                f"Product: {product or subject}" if (product or subject) else "",
                f"Exact product description: {exact_desc}" if exact_desc else "",
                f"Supplier: {supplier}" if supplier else "",
                f"Packer: {packer}" if packer else "",
                f"Origin country: {country}" if country else "",
                f"Batch number: {batch}" if batch else "",
                f"Product ID: {product_id}" if product_id else "",
                f"Code: {code}" if code else "",
                f"Rejected: {rejected}",
                f"Borderline: {borderline}",
                f"Defect percentage: {defect_pct} ({defect_pct_num})" if defect_pct else "",
                f"Description: {description}" if description else "",
                f"Detail: {detail}" if detail else "",
                f"Additional info: {additional}" if additional else "",
            ]
            content = "\n".join(remove_unicode(l) for l in lines if l)

            docs.append(Document(
                page_content=content,
                metadata={
                    **base_meta,
                    "report_id":   report_id,
                    "date":        date,
                    "product":     product or subject,
                    "supplier":    supplier,
                    "country":     country,
                    "rejected":    rejected,
                },
            ))

        logger.info(f"  Inspection reports chunked: {len(docs)}")
        return docs

    def _chunk_skos_concepts(self, g: Graph) -> List[Document]:
        """
        One Document per skos:Concept with English labels + hierarchy.

        Example output:
          SKOS Concept: garlic
          Preferred label (en): garlic
          Also known as: garlic mix, Garlic specialties, Al ajillo
          Broader concept: alliums
        """
        from rdflib import URIRef, Literal
        from rdflib.namespace import RDF, SKOS

        SKOS_CONCEPT = URIRef("http://www.w3.org/2004/02/skos/core#Concept")
        docs: List[Document] = []
        base_meta = {
            "source":      self.file_path,
            "source_type": "turtle",
            "file_name":   self.file_name,
            "chunk_type":  "text",
            "entity_type": "SKOSConcept",
        }

        for concept_uri, _, _ in g.triples((None, RDF.type, SKOS_CONCEPT)):
            # Collect all English preferred labels
            en_pref = [
                str(o) for _, _, o in g.triples((concept_uri, SKOS.prefLabel, None))
                if isinstance(o, Literal) and o.language == "en" and str(o).strip()
            ]
            if not en_pref:
                continue   # skip concepts with no English label

            # Alt labels (synonyms) in English
            en_alt = [
                str(o) for _, _, o in g.triples((concept_uri, SKOS.altLabel, None))
                if isinstance(o, Literal) and o.language == "en" and str(o).strip()
            ]

            # Broader/narrower/related — resolve to English labels
            broader = [
                self._pref_label_en(g, o)
                for _, _, o in g.triples((concept_uri, SKOS.broader, None))
            ]
            narrower = [
                self._pref_label_en(g, o)
                for _, _, o in g.triples((concept_uri, SKOS.narrower, None))
            ]
            related = [
                self._pref_label_en(g, o)
                for _, _, o in g.triples((concept_uri, SKOS.related, None))
            ]

            primary = en_pref[0]
            lines = [f"SKOS Concept: {primary}"]
            if len(en_pref) > 1:
                lines.append(f"Preferred labels: {', '.join(en_pref)}")
            else:
                lines.append(f"Preferred label: {primary}")
            if en_alt:
                lines.append(f"Also known as: {', '.join(en_alt)}")
            if broader:
                lines.append(f"Broader concept: {', '.join(broader)}")
            if narrower:
                lines.append(f"Narrower concepts: {', '.join(narrower)}")
            if related:
                lines.append(f"Related concepts: {', '.join(related)}")

            content = remove_unicode("\n".join(lines))
            docs.append(Document(
                page_content=content,
                metadata={
                    **base_meta,
                    "subject":     primary,
                    "subject_uri": str(concept_uri),
                },
            ))

        logger.info(f"  SKOS concepts chunked: {len(docs)}")
        return docs

    def _chunk_generic(self, g: Graph) -> List[Document]:
        """Entity-centric chunks for any TTL that doesn't match a known domain."""
        from rdflib import URIRef, BNode
        from collections import defaultdict

        subject_sentences: dict = defaultdict(list)
        for s, p, o in g:
            subj = self._uri_to_label(s)
            pred = self._uri_to_label(p).lower()
            obj  = self._uri_to_label(o)
            pred = pred.replace("rdf type", "is a").replace("type", "is a")
            subject_sentences[str(s)].append(remove_unicode(f"{subj} {pred} {obj}."))

        CHUNK = 20
        docs: List[Document] = []
        base_meta = {
            "source":      self.file_path,
            "source_type": "turtle",
            "file_name":   self.file_name,
            "chunk_type":  "text",
            "entity_type": "generic",
        }
        for subj_uri, sentences in subject_sentences.items():
            label = self._uri_to_label(
                URIRef(subj_uri) if subj_uri.startswith("http") else BNode(subj_uri)
            )
            for i in range(0, len(sentences), CHUNK):
                chunk = sentences[i: i + CHUNK]
                docs.append(Document(
                    page_content=f"Facts about {label}:\n" + " ".join(chunk),
                    metadata={**base_meta, "subject": label, "subject_uri": subj_uri},
                ))

        logger.info(f"  Generic entity chunks: {len(docs)}")
        return docs

    # ── Main load entry point ──────────────────────────────────────────────────

    def load(self) -> Tuple[List[Document], List[Document]]:
        logger.info(f"Loading Turtle file: {self.file_path}")

        g = Graph()
        g.parse(self.file_path, format="turtle")
        logger.info(f"  Graph parsed: {len(g)} triples")

        domains = self._detect_domains(g)
        logger.info(f"  Detected domains: {domains}")

        docs: List[Document] = []

        if "inspection" in domains:
            docs.extend(self._chunk_inspection_reports(g))

        if "skos" in domains:
            docs.extend(self._chunk_skos_concepts(g))

        if "generic" in domains:
            docs.extend(self._chunk_generic(g))

        logger.info(
            f"TTL loaded — {len(g)} triples → {len(docs)} chunks "
            f"from '{self.file_name}' (domains: {domains})"
        )

        # Register in SPARQL engine for exact structured queries
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