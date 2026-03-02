"""
SPARQL Engine for Turtle RDF files.

WHY RAG FAILS FOR SPARQL QUERIES:
  A chatbot user typing "What is broader than turbidimetry?" expects an exact
  structured answer from the graph. RAG similarity search cannot do this:
    - It embeds the question and retrieves approximate chunks by cosine distance
    - It never traverses graph relationships (broader/narrower/related)
    - It cannot count, filter by language, or follow multi-hop paths
    - Results are probabilistic and often wrong for factual graph queries

THIS MODULE:
  Adds a real SPARQL query layer that runs directly on the loaded RDF graph.
  The LLM is used only to translate natural language → SPARQL, then SPARQL
  is executed exactly against rdflib, and results are returned as facts.

ARCHITECTURE:
  User question (NL)
      │
      ▼
  nl_to_sparql()     ← LLM generates SPARQL from question + schema context
      │
      ▼
  execute_sparql()   ← rdflib runs SPARQL on in-memory graph — exact results
      │
      ▼
  format_results()   ← LLM formats raw SPARQL results as readable answer
      │
      ▼
  Final answer (with exact facts from the graph)
"""

import re
from pathlib import Path
from typing import Optional
from loguru import logger

from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

from src.config import settings


# ── SKOS namespace context given to LLM when generating SPARQL ─────────────────
SCHEMA_CONTEXT = """
You are querying an AGROVOC SKOS thesaurus stored as an RDF graph.

PREFIXES (always include these in every SPARQL query):
  PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
  PREFIX dct:  <http://purl.org/dc/terms/>
  PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

KEY PREDICATES:
  skos:prefLabel    — preferred label (use FILTER(lang(?label)="en") for English)
  skos:altLabel     — alternative/synonym label
  skos:broader      — broader (parent) concept
  skos:narrower     — narrower (child) concept
  skos:related      — related concept
  skos:definition   — definition URI (not literal text in this dataset)
  skos:inScheme     — concept belongs to a scheme
  skos:Concept      — the type of all concepts
  dct:created       — creation date
  dct:modified      — last modified date

IMPORTANT RULES:
  1. All concepts are URIs like <http://aims.fao.org/aos/agrovoc/c_XXXXX>
  2. Labels are language-tagged literals: "turbidimetry"@en
  3. Always use FILTER(lang(?label) = "en") to get English labels
  4. For broader/narrower, JOIN on prefLabel to get readable names
  5. Use OPTIONAL for fields that may not exist on every concept
  6. Output ONLY the SPARQL query, no explanation, no markdown fences
"""


class SPARQLEngine:
    """
    Loads a Turtle file into an rdflib Graph and provides:
      - execute_sparql(query)   — run raw SPARQL, return list of result dicts
      - answer(question)        — NL question → SPARQL → execute → NL answer
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self._graph: Optional[Graph] = None
        logger.info(f"SPARQLEngine initialised for: {file_path}")

    @property
    def graph(self) -> Graph:
        """Lazy-load the RDF graph."""
        if self._graph is None:
            logger.info(f"Parsing RDF graph: {self.file_path}")
            self._graph = Graph()
            self._graph.parse(self.file_path, format="turtle")
            logger.info(f"Graph loaded: {len(self._graph)} triples")
        return self._graph

    # ── SPARQL execution ───────────────────────────────────────────────────────

    def execute_sparql(self, sparql_query: str) -> list[dict]:
        """
        Execute a SPARQL SELECT query on the loaded graph.
        Returns a list of dicts {variable_name: value_string}.
        """
        try:
            results = self.graph.query(sparql_query)
            rows = []
            for row in results:
                row_dict = {}
                for var in results.vars:
                    val = row[var]
                    row_dict[str(var)] = str(val) if val is not None else ""
                rows.append(row_dict)
            logger.info(f"SPARQL returned {len(rows)} rows")
            return rows
        except Exception as e:
            logger.error(f"SPARQL execution error: {e}")
            raise ValueError(f"SPARQL error: {e}")

    # ── NL → SPARQL → execute → NL answer ─────────────────────────────────────

    def nl_to_sparql(self, question: str) -> str:
        """Use the LLM to translate a natural language question to SPARQL."""
        from src.rag.llm import get_llm

        llm = get_llm()
        prompt = (
            f"{SCHEMA_CONTEXT}\n\n"
            f"Translate this question to a SPARQL SELECT query:\n"
            f"Question: {question}\n\n"
            f"SPARQL query:"
        )
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown fences if LLM added them
        raw = re.sub(r"^```(?:sparql|sql)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    def results_to_answer(self, question: str, sparql: str, rows: list[dict]) -> str:
        """Use the LLM to turn raw SPARQL results into a readable answer."""
        from src.rag.llm import get_llm

        llm = get_llm()

        if not rows:
            results_text = "No results found."
        else:
            lines = []
            for i, row in enumerate(rows[:50], 1):   # cap at 50 rows
                lines.append(f"  {i}. " + " | ".join(
                    f"{k}: {v}" for k, v in row.items() if v
                ))
            results_text = "\n".join(lines)

        prompt = (
            f"The user asked: {question}\n\n"
            f"SPARQL query used:\n{sparql}\n\n"
            f"Results from the knowledge graph:\n{results_text}\n\n"
            f"Write a clear, concise answer to the user's question based on "
            f"these exact results. Do not add information not present in the "
            f"results. If no results, say so clearly."
        )
        response = llm.invoke(prompt)
        return response.content.strip()

    def answer(self, question: str) -> dict:
        """
        Full pipeline: NL question → SPARQL → execute → NL answer.

        Returns:
            {
                "answer":  str,           # human-readable answer
                "sparql":  str,           # generated SPARQL (for transparency)
                "rows":    list[dict],    # raw results
                "success": bool,
                "error":   str | None,
            }
        """
        logger.info(f"SPARQLEngine.answer(): '{question[:80]}'")

        # Step 1: generate SPARQL
        try:
            sparql = self.nl_to_sparql(question)
            logger.info(f"Generated SPARQL:\n{sparql}")
        except Exception as e:
            return {
                "answer":  f"Could not generate SPARQL query: {e}",
                "sparql":  "",
                "rows":    [],
                "success": False,
                "error":   str(e),
            }

        # Step 2: execute
        try:
            rows = self.execute_sparql(sparql)
        except Exception as e:
            return {
                "answer":  f"SPARQL execution failed: {e}\n\nGenerated query:\n```sparql\n{sparql}\n```",
                "sparql":  sparql,
                "rows":    [],
                "success": False,
                "error":   str(e),
            }

        # Step 3: format answer
        answer_text = self.results_to_answer(question, sparql, rows)

        return {
            "answer":  answer_text,
            "sparql":  sparql,
            "rows":    rows,
            "success": True,
            "error":   None,
        }


# ── Registry: file_path → SPARQLEngine ────────────────────────────────────────
# Keeps one loaded graph per TTL file — avoids re-parsing on every query.

_engines: dict[str, SPARQLEngine] = {}


def get_sparql_engine(file_path: str) -> SPARQLEngine:
    """Return a cached SPARQLEngine for the given TTL file."""
    if file_path not in _engines:
        _engines[file_path] = SPARQLEngine(file_path)
    return _engines[file_path]


def get_all_ttl_engines() -> dict[str, SPARQLEngine]:
    """Return all currently loaded SPARQL engines keyed by file path."""
    return _engines
