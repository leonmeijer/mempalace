"""
knowledge_graph.py — Temporal Entity-Relationship Graph for MemPalace
=====================================================================

Real knowledge graph with:
  - Entity nodes (people, projects, tools, concepts)
  - Typed relationship edges (daughter_of, does, loves, works_on, etc.)
  - Temporal validity (valid_from → valid_to — knows WHEN facts are true)
  - Closet references (links back to the verbatim memory)

Storage: IndentiaGraph via SPARQL 1.2 (port 7001)
Namespaces: https://id.indentia.ai/ (ADR-121)

Usage:
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "loves", "chess", valid_from="2025-10-01")

    # Query: everything about Max
    kg.query_entity("Max")

    # Query: what was true about Max in January 2026?
    kg.query_entity("Max", as_of="2026-01-15")

    # Query: who is connected to Alice?
    kg.query_entity("Alice", direction="both")

    # Invalidate: Max's sports injury resolved
    kg.invalidate("Max", "has_issue", "sports_injury", ended="2026-02-15")
"""

import hashlib
import json
import logging
import os
import threading
from datetime import date, datetime
from urllib.parse import quote as _url_quote

import requests

logger = logging.getLogger(__name__)

# ── ADR-121 namespaces ────────────────────────────────────────────────────────
_BASE = "https://id.indentia.ai/"
_ENTITY_BASE = f"{_BASE}memory/entity/mempalace/"
_PRED_BASE = f"{_BASE}memory/predicate/mempalace/"
_ONTOLOGY = f"{_BASE}memory/ontology/"
_GRAPH_ENTITIES = f"{_BASE}sources/mempalace/entities"
_GRAPH_KG = f"{_BASE}sources/mempalace/kg"

# Common SPARQL prefix block used in every query/update
_PREFIXES = f"""PREFIX mp: <{_ONTOLOGY}>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

DEFAULT_SPARQL_URL = os.getenv("INDENTIAGRAPH_SPARQL_URL", "http://localhost:7001")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _esc(value: str) -> str:
    """Escape a string for use inside a SPARQL double-quoted literal."""
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\r")
    value = value.replace("\t", "\\t")
    return value


def _entity_id(name: str) -> str:
    """Normalize a name to a safe IRI local part."""
    return _url_quote(
        name.lower().replace(" ", "_").replace("'", ""),
        safe="-._~",
    )


def _entity_uri(name: str) -> str:
    return f"{_ENTITY_BASE}{_entity_id(name)}"


def _pred_uri(predicate: str) -> str:
    pred_norm = predicate.lower().replace(" ", "_")
    return f"{_PRED_BASE}{_url_quote(pred_norm, safe='-._~')}"


def _pred_local(pred_uri: str) -> str:
    """Extract the local part from a predicate URI."""
    return pred_uri.split("/")[-1]


def _sparql_binding(b: dict) -> str | None:
    """Extract the string value from a SPARQL result binding, or None."""
    if b is None:
        return None
    return b.get("value")


class KnowledgeGraph:
    def __init__(self, db_path: str = None, sparql_url: str = None):
        # db_path is accepted but ignored — kept for API compatibility
        # with callers that still pass it (e.g. mcp_server.py).
        _ = db_path
        self._sparql_url = (sparql_url or DEFAULT_SPARQL_URL).rstrip("/")
        self._query_ep = f"{self._sparql_url}/sparql"
        self._update_ep = f"{self._sparql_url}/update"
        self._session = requests.Session()
        self._lock = threading.Lock()

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _query(self, sparql: str) -> dict:
        """Execute a SPARQL SELECT or ASK query."""
        with self._lock:
            resp = self._session.post(
                self._query_ep,
                data=(_PREFIXES + sparql).encode("utf-8"),
                headers={
                    "Content-Type": "application/sparql-query",
                    "Accept": "application/sparql-results+json",
                },
                timeout=15,
            )
        resp.raise_for_status()
        return resp.json()

    def _update(self, sparql: str) -> None:
        """Execute a SPARQL UPDATE (INSERT/DELETE)."""
        with self._lock:
            resp = self._session.post(
                self._update_ep,
                data=(_PREFIXES + sparql).encode("utf-8"),
                headers={"Content-Type": "application/sparql-update"},
                timeout=15,
            )
        resp.raise_for_status()

    # ── Write operations ──────────────────────────────────────────────────────

    def _entity_exists(self, uri: str) -> bool:
        result = self._query(
            f"""ASK {{
  GRAPH <{_GRAPH_ENTITIES}> {{
    <{uri}> mp:name ?n .
  }}
}}"""
        )
        return result.get("boolean", False)

    def add_entity(self, name: str, entity_type: str = "unknown", properties: dict = None):
        """Add or replace an entity node."""
        uri = _entity_uri(name)
        props_json = _esc(json.dumps(properties or {}))
        created = datetime.now().isoformat()
        ename = _esc(name)
        etype = _esc(entity_type)

        # DELETE existing annotations first (replace semantics)
        self._update(
            f"""DELETE {{
  GRAPH <{_GRAPH_ENTITIES}> {{
    <{uri}> ?p ?o .
  }}
}}
WHERE {{
  GRAPH <{_GRAPH_ENTITIES}> {{
    <{uri}> ?p ?o .
  }}
}}"""
        )

        self._update(
            f"""INSERT DATA {{
  GRAPH <{_GRAPH_ENTITIES}> {{
    <{uri}> a mp:Entity ;
      mp:name "{ename}" ;
      mp:type "{etype}" ;
      mp:properties "{props_json}" ;
      mp:createdAt "{created}" .
  }}
}}"""
        )
        return _entity_id(name)

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source_closet: str = None,
        source_file: str = None,
    ):
        """Add a relationship triple: subject → predicate → object."""
        sub_uri = _entity_uri(subject)
        obj_uri = _entity_uri(obj)
        pred_uri = _pred_uri(predicate)

        # Auto-create entities if missing
        if not self._entity_exists(sub_uri):
            self.add_entity(subject)
        if not self._entity_exists(obj_uri):
            self.add_entity(obj)

        # Check for existing active triple — return its ID if found
        existing_id = self._query(
            f"""SELECT ?tid WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    <{sub_uri}> <{pred_uri}> <{obj_uri}> .
    << <{sub_uri}> <{pred_uri}> <{obj_uri}> >> mp:tripleId ?tid .
    FILTER NOT EXISTS {{
      << <{sub_uri}> <{pred_uri}> <{obj_uri}> >> mp:validTo ?vt .
    }}
  }}
}}"""
        )
        rows = existing_id.get("results", {}).get("bindings", [])
        if rows:
            return rows[0]["tid"]["value"]

        triple_id = (
            f"t_{_entity_id(subject)}_{_entity_id(predicate)}_{_entity_id(obj)}_"
            f"{hashlib.sha256(f'{valid_from}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
        )
        extracted_at = datetime.now().isoformat()

        # Build RDF-star annotations — each as a separate statement to avoid
        # Turtle ';'-chaining issues when inserting optional properties.
        _ann = f"<< <{sub_uri}> <{pred_uri}> <{obj_uri}> >>"
        annotations: list[str] = [
            f'{_ann} mp:tripleId "{triple_id}" .',
            f'{_ann} mp:confidence "{confidence}"^^xsd:decimal .',
            f'{_ann} mp:extractedAt "{extracted_at}" .',
        ]
        if valid_from:
            annotations.append(f'{_ann} mp:validFrom "{_esc(valid_from)}" .')
        if valid_to:
            annotations.append(f'{_ann} mp:validTo "{_esc(valid_to)}" .')
        if source_closet:
            annotations.append(f'{_ann} mp:sourceCloset "{_esc(source_closet)}" .')
        if source_file:
            annotations.append(f'{_ann} mp:sourceFile "{_esc(source_file)}" .')

        annotations_str = "\n    ".join(annotations)

        self._update(
            f"""INSERT DATA {{
  GRAPH <{_GRAPH_KG}> {{
    <{sub_uri}> <{pred_uri}> <{obj_uri}> .
    {annotations_str}
  }}
}}"""
        )
        return triple_id

    def invalidate(self, subject: str, predicate: str, obj: str, ended: str = None):
        """Mark a relationship as no longer valid (set valid_to date)."""
        sub_uri = _entity_uri(subject)
        obj_uri = _entity_uri(obj)
        pred_uri = _pred_uri(predicate)
        ended = ended or date.today().isoformat()

        self._update(
            f"""INSERT {{
  GRAPH <{_GRAPH_KG}> {{
    << <{sub_uri}> <{pred_uri}> <{obj_uri}> >> mp:validTo "{_esc(ended)}" .
  }}
}}
WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    <{sub_uri}> <{pred_uri}> <{obj_uri}> .
    FILTER NOT EXISTS {{
      << <{sub_uri}> <{pred_uri}> <{obj_uri}> >> mp:validTo ?vt .
    }}
  }}
}}"""
        )

    # ── Query operations ──────────────────────────────────────────────────────

    def query_entity(self, name: str, as_of: str = None, direction: str = "outgoing"):
        """Get all relationships for an entity.

        direction: "outgoing" (entity → ?), "incoming" (? → entity), "both"
        as_of: date string — only return facts valid at that time
        """
        eid_uri = _entity_uri(name)
        results = []

        temporal_filter = ""
        if as_of:
            d = _esc(as_of)
            temporal_filter = f"""
    FILTER(!BOUND(?vf) || ?vf <= "{d}")
    FILTER(!BOUND(?vt) || ?vt >= "{d}")"""

        if direction in ("outgoing", "both"):
            sparql = f"""SELECT ?pred ?obj ?objName ?vf ?vt ?conf ?closet WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    <{eid_uri}> ?pred ?obj .
    OPTIONAL {{ << <{eid_uri}> ?pred ?obj >> mp:validFrom ?vf }}
    OPTIONAL {{ << <{eid_uri}> ?pred ?obj >> mp:validTo   ?vt }}
    OPTIONAL {{ << <{eid_uri}> ?pred ?obj >> mp:confidence ?conf }}
    OPTIONAL {{ << <{eid_uri}> ?pred ?obj >> mp:sourceCloset ?closet }}
    FILTER(STRSTARTS(STR(?pred), "{_PRED_BASE}"))
    FILTER(STRSTARTS(STR(?obj),  "{_ENTITY_BASE}"))
  }}{temporal_filter}
  OPTIONAL {{
    GRAPH <{_GRAPH_ENTITIES}> {{ ?obj mp:name ?objName }}
  }}
}}"""
            data = self._query(sparql)
            for b in data.get("results", {}).get("bindings", []):
                pred_local = _pred_local(_sparql_binding(b.get("pred")) or "")
                obj_label = _sparql_binding(b.get("objName")) or _sparql_binding(b.get("obj")) or ""
                vt = _sparql_binding(b.get("vt"))
                results.append(
                    {
                        "direction": "outgoing",
                        "subject": name,
                        "predicate": pred_local,
                        "object": obj_label,
                        "valid_from": _sparql_binding(b.get("vf")),
                        "valid_to": vt,
                        "confidence": float(_sparql_binding(b.get("conf")) or 1.0),
                        "source_closet": _sparql_binding(b.get("closet")),
                        "current": vt is None,
                    }
                )

        if direction in ("incoming", "both"):
            sparql = f"""SELECT ?pred ?sub ?subName ?vf ?vt ?conf ?closet WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    ?sub ?pred <{eid_uri}> .
    OPTIONAL {{ << ?sub ?pred <{eid_uri}> >> mp:validFrom ?vf }}
    OPTIONAL {{ << ?sub ?pred <{eid_uri}> >> mp:validTo   ?vt }}
    OPTIONAL {{ << ?sub ?pred <{eid_uri}> >> mp:confidence ?conf }}
    OPTIONAL {{ << ?sub ?pred <{eid_uri}> >> mp:sourceCloset ?closet }}
    FILTER(STRSTARTS(STR(?pred), "{_PRED_BASE}"))
    FILTER(STRSTARTS(STR(?sub),  "{_ENTITY_BASE}"))
  }}{temporal_filter}
  OPTIONAL {{
    GRAPH <{_GRAPH_ENTITIES}> {{ ?sub mp:name ?subName }}
  }}
}}"""
            data = self._query(sparql)
            for b in data.get("results", {}).get("bindings", []):
                pred_local = _pred_local(_sparql_binding(b.get("pred")) or "")
                sub_label = _sparql_binding(b.get("subName")) or _sparql_binding(b.get("sub")) or ""
                vt = _sparql_binding(b.get("vt"))
                results.append(
                    {
                        "direction": "incoming",
                        "subject": sub_label,
                        "predicate": pred_local,
                        "object": name,
                        "valid_from": _sparql_binding(b.get("vf")),
                        "valid_to": vt,
                        "confidence": float(_sparql_binding(b.get("conf")) or 1.0),
                        "source_closet": _sparql_binding(b.get("closet")),
                        "current": vt is None,
                    }
                )

        return results

    def query_relationship(self, predicate: str, as_of: str = None):
        """Get all triples with a given relationship type."""
        pred_uri = _pred_uri(predicate)

        temporal_filter = ""
        if as_of:
            d = _esc(as_of)
            temporal_filter = f"""
    FILTER(!BOUND(?vf) || ?vf <= "{d}")
    FILTER(!BOUND(?vt) || ?vt >= "{d}")"""

        sparql = f"""SELECT ?sub ?obj ?subName ?objName ?vf ?vt WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    ?sub <{pred_uri}> ?obj .
    OPTIONAL {{ << ?sub <{pred_uri}> ?obj >> mp:validFrom ?vf }}
    OPTIONAL {{ << ?sub <{pred_uri}> ?obj >> mp:validTo   ?vt }}
  }}{temporal_filter}
  OPTIONAL {{ GRAPH <{_GRAPH_ENTITIES}> {{ ?sub mp:name ?subName }} }}
  OPTIONAL {{ GRAPH <{_GRAPH_ENTITIES}> {{ ?obj mp:name ?objName }} }}
}}"""
        data = self._query(sparql)
        results = []
        pred_local = _pred_local(pred_uri)
        for b in data.get("results", {}).get("bindings", []):
            sub_label = _sparql_binding(b.get("subName")) or _sparql_binding(b.get("sub")) or ""
            obj_label = _sparql_binding(b.get("objName")) or _sparql_binding(b.get("obj")) or ""
            vt = _sparql_binding(b.get("vt"))
            results.append(
                {
                    "subject": sub_label,
                    "predicate": pred_local,
                    "object": obj_label,
                    "valid_from": _sparql_binding(b.get("vf")),
                    "valid_to": vt,
                    "current": vt is None,
                }
            )
        return results

    def timeline(self, entity_name: str = None):
        """Get all facts in chronological order, optionally filtered by entity."""
        if entity_name:
            eid_uri = _entity_uri(entity_name)
            entity_filter = f"""
    FILTER(
      (STRSTARTS(STR(?sub), "{_ENTITY_BASE}") && ?sub = <{eid_uri}>)
      || (STRSTARTS(STR(?obj), "{_ENTITY_BASE}") && ?obj = <{eid_uri}>)
    )"""
        else:
            entity_filter = ""

        sparql = f"""SELECT ?sub ?pred ?obj ?subName ?objName ?vf ?vt WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    ?sub ?pred ?obj .
    FILTER(STRSTARTS(STR(?pred), "{_PRED_BASE}"))
    OPTIONAL {{ << ?sub ?pred ?obj >> mp:validFrom ?vf }}
    OPTIONAL {{ << ?sub ?pred ?obj >> mp:validTo   ?vt }}
  }}{entity_filter}
  OPTIONAL {{ GRAPH <{_GRAPH_ENTITIES}> {{ ?sub mp:name ?subName }} }}
  OPTIONAL {{ GRAPH <{_GRAPH_ENTITIES}> {{ ?obj mp:name ?objName }} }}
}}
ORDER BY ASC(?vf)
LIMIT 100"""
        data = self._query(sparql)
        results = []
        for b in data.get("results", {}).get("bindings", []):
            sub_label = _sparql_binding(b.get("subName")) or _sparql_binding(b.get("sub")) or ""
            obj_label = _sparql_binding(b.get("objName")) or _sparql_binding(b.get("obj")) or ""
            pred_local = _pred_local(_sparql_binding(b.get("pred")) or "")
            vt = _sparql_binding(b.get("vt"))
            results.append(
                {
                    "subject": sub_label,
                    "predicate": pred_local,
                    "object": obj_label,
                    "valid_from": _sparql_binding(b.get("vf")),
                    "valid_to": vt,
                    "current": vt is None,
                }
            )
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self):
        entity_count_result = self._query(
            f"""SELECT (COUNT(DISTINCT ?e) AS ?cnt) WHERE {{
  GRAPH <{_GRAPH_ENTITIES}> {{ ?e a mp:Entity }}
}}"""
        )
        triple_count_result = self._query(
            f"""SELECT (COUNT(*) AS ?cnt) WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    ?s ?p ?o .
    FILTER(STRSTARTS(STR(?p), "{_PRED_BASE}"))
  }}
}}"""
        )
        current_count_result = self._query(
            f"""SELECT (COUNT(*) AS ?cnt) WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    ?s ?p ?o .
    FILTER(STRSTARTS(STR(?p), "{_PRED_BASE}"))
    FILTER NOT EXISTS {{ << ?s ?p ?o >> mp:validTo ?vt }}
  }}
}}"""
        )
        preds_result = self._query(
            f"""SELECT DISTINCT ?pred WHERE {{
  GRAPH <{_GRAPH_KG}> {{
    ?s ?pred ?o .
    FILTER(STRSTARTS(STR(?pred), "{_PRED_BASE}"))
  }}
}}
ORDER BY ?pred"""
        )

        def _first_cnt(res):
            bindings = res.get("results", {}).get("bindings", [])
            if bindings:
                return int(_sparql_binding(bindings[0].get("cnt")) or 0)
            return 0

        entities = _first_cnt(entity_count_result)
        triples = _first_cnt(triple_count_result)
        current = _first_cnt(current_count_result)
        expired = triples - current
        predicates = [
            _pred_local(_sparql_binding(b.get("pred")) or "")
            for b in preds_result.get("results", {}).get("bindings", [])
        ]
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates,
        }

    def close(self):
        """Close the HTTP session."""
        with self._lock:
            self._session.close()

    # ── Seed from known facts ─────────────────────────────────────────────────

    def seed_from_entity_facts(self, entity_facts: dict):
        """Seed the knowledge graph from fact_checker.py ENTITY_FACTS."""
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            etype = facts.get("type", "person")
            self.add_entity(
                name,
                etype,
                {
                    "gender": facts.get("gender", ""),
                    "birthday": facts.get("birthday", ""),
                },
            )

            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name, "child_of", parent.capitalize(), valid_from=facts.get("birthday")
                )

            partner = facts.get("partner")
            if partner:
                self.add_triple(name, "married_to", partner.capitalize())

            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                self.add_triple(
                    name,
                    "is_child_of",
                    facts.get("parent", "").capitalize() or name,
                    valid_from=facts.get("birthday"),
                )
            elif relationship == "husband":
                self.add_triple(name, "is_partner_of", facts.get("partner", name).capitalize())
            elif relationship == "brother":
                self.add_triple(name, "is_sibling_of", facts.get("sibling", name).capitalize())
            elif relationship == "dog":
                self.add_triple(name, "is_pet_of", facts.get("owner", name).capitalize())
                self.add_entity(name, "animal")

            for interest in facts.get("interests", []):
                self.add_triple(name, "loves", interest.capitalize(), valid_from="2025-01-01")
