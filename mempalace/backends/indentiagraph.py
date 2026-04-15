"""IndentiaGraph-backed MemPalace collection adapter.

Uses IndentiaGraph's Elasticsearch-compatible API (port 9200) for palace
drawer/closet storage with native HNSW vector indexing.

Embeddings are generated locally via sentence-transformers — no API key,
no data leaves the machine (same model ChromaDB used internally).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from .base import BaseCollection

logger = logging.getLogger(__name__)

# Metadata fields stored as ES document fields alongside content + embedding.
_METADATA_FIELDS = [
    "wing",
    "room",
    "source_file",
    "chunk_index",
    "added_by",
    "filed_at",
    "normalize_version",
    "source_mtime",
    "hall",
    "entities",
    "generated_by",
    # Diary-entry fields (mcp_server.tool_diary_write)
    "topic",
    "type",
    "agent",
    "date",
    # Conversation-miner fields
    "ingest_mode",
    "extract_mode",
]

_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
            "wing": {"type": "keyword"},
            "room": {"type": "keyword"},
            "source_file": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "added_by": {"type": "keyword"},
            "filed_at": {"type": "keyword"},
            "normalize_version": {"type": "integer"},
            "source_mtime": {"type": "float"},
            "hall": {"type": "keyword"},
            "entities": {"type": "keyword"},
            "generated_by": {"type": "keyword"},
            "topic": {"type": "keyword"},
            "type": {"type": "keyword"},
            "agent": {"type": "keyword"},
            "date": {"type": "keyword"},
            "ingest_mode": {"type": "keyword"},
            "extract_mode": {"type": "keyword"},
        }
    },
}


def _build_es_filter(where: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a ChromaDB-style where-dict into an ES bool/filter clause.

    ChromaDB formats:
      {"wing": "x"}                              — simple equality
      {"$and": [{"wing": "x"}, {"room": "y"}]}  — AND
      {"$or": [{"wing": "x"}, {"wing": "y"}]}   — OR (rare but handle it)
      {"chunk_index": {"$in": [0, 1, 2]}}        — terms (multi-value match)
    """
    if not where:
        return {}

    if "$and" in where:
        clauses = [_build_es_filter(c) for c in where["$and"]]
        return {"bool": {"filter": clauses}}

    if "$or" in where:
        clauses = [_build_es_filter(c) for c in where["$or"]]
        return {"bool": {"should": clauses, "minimum_should_match": 1}}

    # Single key — may be equality OR a ChromaDB operator dict
    if len(where) == 1:
        key, val = next(iter(where.items()))
        if isinstance(val, dict):
            # ChromaDB operator: {"$in": [...]} or {"$gte": x} etc.
            if "$in" in val:
                return {"terms": {key: val["$in"]}}
            if "$nin" in val:
                return {"bool": {"must_not": [{"terms": {key: val["$nin"]}}]}}
            if "$eq" in val:
                return {"term": {key: val["$eq"]}}
            if "$ne" in val:
                return {"bool": {"must_not": [{"term": {key: val["$ne"]}}]}}
            if "$gt" in val:
                return {"range": {key: {"gt": val["$gt"]}}}
            if "$gte" in val:
                return {"range": {key: {"gte": val["$gte"]}}}
            if "$lt" in val:
                return {"range": {key: {"lt": val["$lt"]}}}
            if "$lte" in val:
                return {"range": {key: {"lte": val["$lte"]}}}
        return {"term": {key: val}}

    # Multiple key-value pairs → implicit AND
    clauses = [{"term": {k: v}} for k, v in where.items()]
    return {"bool": {"filter": clauses}}


def _es_score_to_chroma_distance(score: float) -> float:
    """Convert ES cosine similarity score to ChromaDB cosine distance.

    ES scores normalized cosine similarity as (1 + dot_product) / 2 ∈ [0, 1].
    ChromaDB cosine distance = 1 - cos_sim = 2 * (1 - es_score) ∈ [0, 2].
    """
    return 2.0 * (1.0 - score)


def _doc_to_meta(source: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata dict from an ES _source document."""
    meta = {}
    for field in _METADATA_FIELDS:
        val = source.get(field)
        if val is not None:
            meta[field] = val
    return meta


class IndentiaGraphCollection(BaseCollection):
    """Adapter over an IndentiaGraph (ES-compatible) index."""

    def __init__(self, es_client, index_name: str):
        self._es = es_client
        self._index = index_name
        self._embedder = None  # lazy-loaded

    def _get_embedder(self):
        if self._embedder is None:
            from ..embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, *, documents: List[str], ids: List[str], metadatas=None) -> None:
        """Insert new records (same as upsert for ES)."""
        self.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def upsert(
        self,
        *,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or replace documents, generating embeddings locally."""
        embedder = self._get_embedder()
        embeddings = embedder.embed_batch(documents)
        metas = metadatas or [{}] * len(documents)

        for doc, doc_id, embedding, meta in zip(documents, ids, embeddings, metas):
            body: Dict[str, Any] = {"content": doc, "embedding": embedding}
            body.update({k: v for k, v in meta.items() if v is not None})
            self._es.index(
                index=self._index,
                id=doc_id,
                document=body,
                refresh="wait_for",
            )

    def update(self, **kwargs: Any) -> None:
        """Update existing record. Raises ValueError if the ID is missing."""
        ids = kwargs.get("ids", [])
        documents = kwargs.get("documents")
        metadatas = kwargs.get("metadatas")

        if not ids:
            raise ValueError("update() requires at least one id")

        for i, doc_id in enumerate(ids):
            # Fetch current document to verify it exists
            try:
                current = self._es.get(index=self._index, id=doc_id)
            except Exception as exc:
                raise ValueError(f"ID not found for update: {doc_id}") from exc

            current_source = current["_source"]
            update_body: Dict[str, Any] = {}

            if documents is not None:
                new_doc = documents[i]
                update_body["content"] = new_doc
                update_body["embedding"] = self._get_embedder().embed(new_doc)

            if metadatas is not None:
                meta = metadatas[i]
                update_body.update({k: v for k, v in meta.items() if v is not None})

            if update_body:
                # Merge: start from existing, apply updates
                merged = dict(current_source)
                merged.update(update_body)
                self._es.index(
                    index=self._index,
                    id=doc_id,
                    document=merged,
                    refresh="wait_for",
                )

    def delete(self, **kwargs: Any) -> None:
        """Delete by IDs or where-filter.

        IndentiaGraph's delete_by_query endpoint has a serialization bug with
        complex query bodies, so we implement delete-by-filter as a two-step
        search-then-delete-by-ID operation.
        """
        ids = kwargs.get("ids")
        where = kwargs.get("where")

        if ids:
            target_ids = list(ids)
        elif where:
            # Search for IDs matching the filter, then delete by ID
            es_filter = _build_es_filter(where)
            resp = self._es.search(
                index=self._index,
                body={"query": es_filter, "size": 10000, "_source": False},
            )
            target_ids = [h["_id"] for h in resp["hits"]["hits"]]
        else:
            return

        if not target_ids:
            return

        # Delete one-by-one; refresh on last to avoid too many refresh calls
        for i, doc_id in enumerate(target_ids):
            refresh = "wait_for" if (i == len(target_ids) - 1) else "false"
            try:
                self._es.delete(index=self._index, id=doc_id, refresh=refresh)
            except Exception:
                pass  # 404 is fine — doc was already gone

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(self, **kwargs: Any) -> Dict[str, Any]:
        """Vector similarity search.

        Accepts ChromaDB-style kwargs:
          query_texts: list of query strings (only first is used)
          n_results: number of results to return
          include: list of ["documents", "metadatas", "distances"]
          where: metadata filter dict
        """
        query_texts = kwargs.get("query_texts", [])
        n_results = kwargs.get("n_results", 5)
        include = kwargs.get("include", ["documents", "metadatas", "distances"])
        where = kwargs.get("where")

        if not query_texts:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_vec = self._get_embedder().embed(query_texts[0])

        knn_clause: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vec,
            "k": n_results,
            "num_candidates": max(n_results * 10, 100),
        }
        if where:
            knn_clause["filter"] = _build_es_filter(where)

        # Determine which source fields to fetch
        source_fields = ["content"] + _METADATA_FIELDS

        response = self._es.search(
            index=self._index,
            body={
                "knn": knn_clause,
                "size": n_results,
                "_source": source_fields,
            },
        )

        hits = response["hits"]["hits"]
        ids_row: List[str] = []
        docs_row: List[str] = []
        metas_row: List[Dict[str, Any]] = []
        dists_row: List[float] = []

        for hit in hits:
            ids_row.append(hit["_id"])
            src = hit.get("_source", {})
            docs_row.append(src.get("content", ""))
            metas_row.append(_doc_to_meta(src))
            dists_row.append(_es_score_to_chroma_distance(hit.get("_score", 0.0)))

        result: Dict[str, Any] = {"ids": [ids_row]}
        if "documents" in include:
            result["documents"] = [docs_row]
        else:
            result["documents"] = [None]
        if "metadatas" in include:
            result["metadatas"] = [metas_row]
        else:
            result["metadatas"] = [None]
        if "distances" in include:
            result["distances"] = [dists_row]
        else:
            result["distances"] = [None]

        return result

    def get(self, **kwargs: Any) -> Dict[str, Any]:
        """Fetch documents by IDs or metadata filter.

        Accepts ChromaDB-style kwargs:
          ids: list of IDs to fetch
          where: metadata filter dict
          include: list of ["documents", "metadatas"]
          limit: max number of results (default 1000)
          offset: pagination offset (default 0)
        """
        ids = kwargs.get("ids")
        where = kwargs.get("where")
        include = kwargs.get("include", ["documents", "metadatas"])
        limit = kwargs.get("limit", 1000)
        offset = kwargs.get("offset", 0)

        # Build query
        if ids:
            query = {"ids": {"values": list(ids)}}
        elif where:
            query = _build_es_filter(where)
        else:
            query = {"match_all": {}}

        # Determine source fields
        if not include:
            source_fields: Any = False  # IDs only
        else:
            source_fields = ["content"] + _METADATA_FIELDS

        response = self._es.search(
            index=self._index,
            body={
                "query": query,
                "size": limit,
                "from": offset,
                "_source": source_fields,
            },
        )

        hits = response["hits"]["hits"]
        result_ids: List[str] = [h["_id"] for h in hits]
        result_docs: Optional[List[str]] = None
        result_metas: Optional[List[Dict[str, Any]]] = None

        if "documents" in include:
            result_docs = [h.get("_source", {}).get("content", "") for h in hits]
        if "metadatas" in include:
            result_metas = [_doc_to_meta(h.get("_source", {})) for h in hits]

        return {
            "ids": result_ids,
            "documents": result_docs,
            "metadatas": result_metas,
        }

    def count(self) -> int:
        """Return total document count in this index.

        Uses search(size=0) instead of count() because IndentiaGraph's
        content-type guard rejects the bare count request sent by
        elasticsearch-py (which omits the JSON body and therefore the header).
        """
        try:
            response = self._es.search(
                index=self._index,
                body={"query": {"match_all": {}}, "size": 0},
            )
            return response["hits"]["total"]["value"]
        except Exception:
            return 0


class IndentiaGraphBackend:
    """Factory for MemPalace's IndentiaGraph storage backend."""

    def __init__(
        self,
        es_url: Optional[str] = None,
    ):
        self._es_url = es_url or os.getenv(
            "INDENTIAGRAPH_ES_URL", "http://localhost:9200"
        )
        self._client_cache: Dict[str, Any] = {}

    def _client(self):
        """Return a shared Elasticsearch client."""
        if "default" not in self._client_cache:
            self._client_cache["default"] = self._make_es_client(self._es_url)
        return self._client_cache["default"]

    @staticmethod
    def _make_es_client(es_url: str):
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise ImportError(
                "elasticsearch-py is required for the IndentiaGraph backend. "
                "Install it with: pip install elasticsearch"
            ) from exc

        return Elasticsearch(
            hosts=[es_url],
            verify_certs=False,
            request_timeout=30,
        )

    @staticmethod
    def make_client(es_url: Optional[str] = None):
        """Create and return a fresh Elasticsearch client."""
        url = es_url or os.getenv("INDENTIAGRAPH_ES_URL", "http://localhost:9200")
        return IndentiaGraphBackend._make_es_client(url)

    @staticmethod
    def backend_version() -> str:
        """Return IndentiaGraph server version via health endpoint."""
        import requests as _requests

        sparql_url = os.getenv("INDENTIAGRAPH_SPARQL_URL", "http://localhost:7001")
        try:
            resp = _requests.get(f"{sparql_url}/health", timeout=5)
            data = resp.json()
            return str(data.get("version", "indentiagraph"))
        except Exception:
            return "indentiagraph (unknown version)"

    @staticmethod
    def _derive_index_name(palace_path: str, collection_name: str) -> str:
        """Derive a stable, unique ES index name from palace path + collection name.

        Each palace directory gets its own index so collections don't collide
        across different palace paths (matching ChromaDB's PersistentClient
        isolation by directory).
        """
        path_hash = hashlib.sha256(
            os.path.abspath(palace_path).encode()
        ).hexdigest()[:12]
        return f"{collection_name.lower()}_{path_hash}"

    def _ensure_index(self, es, index_name: str) -> None:
        """Create the ES index with the correct mapping if it doesn't exist."""
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=_INDEX_MAPPING)
            logger.info("Created IndentiaGraph index: %s", index_name)

    def get_collection(
        self,
        palace_path: str,
        collection_name: str,
        create: bool = False,
    ) -> IndentiaGraphCollection:
        es = self._client()
        index_name = self._derive_index_name(palace_path, collection_name)

        if not create and not es.indices.exists(index=index_name):
            raise FileNotFoundError(
                f"IndentiaGraph index '{index_name}' does not exist. "
                "Run with create=True or mine the palace first."
            )

        if create:
            self._ensure_index(es, index_name)

        return IndentiaGraphCollection(es, index_name)

    def get_or_create_collection(
        self, palace_path: str, collection_name: str
    ) -> IndentiaGraphCollection:
        return self.get_collection(palace_path, collection_name, create=True)

    def delete_collection(self, palace_path: str, collection_name: str) -> None:
        es = self._client()
        index_name = self._derive_index_name(palace_path, collection_name)
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            logger.info("Deleted IndentiaGraph index: %s", index_name)

    def create_collection(
        self, palace_path: str, collection_name: str, hnsw_space: str = "cosine"
    ) -> IndentiaGraphCollection:
        """Explicitly create a collection (fails if it already exists)."""
        es = self._client()
        index_name = self._derive_index_name(palace_path, collection_name)
        if es.indices.exists(index=index_name):
            raise ValueError(f"Index '{index_name}' already exists.")
        es.indices.create(index=index_name, body=_INDEX_MAPPING)
        return IndentiaGraphCollection(es, index_name)
