"""Tests for mempalace.backends.indentiagraph — IndentiaGraph storage backend.

All ES calls are mocked — no real server needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from mempalace.backends.indentiagraph import (
    IndentiaGraphBackend,
    IndentiaGraphCollection,
    _build_es_filter,
    _es_score_to_chroma_distance,
)


# ── ES filter translation ────────────────────────────────────────────────


class TestBuildEsFilter:
    def test_simple_equality(self):
        result = _build_es_filter({"wing": "people"})
        assert result == {"term": {"wing": "people"}}

    def test_and_operator(self):
        result = _build_es_filter({"$and": [{"wing": "people"}, {"room": "alice"}]})
        assert result == {
            "bool": {
                "filter": [
                    {"term": {"wing": "people"}},
                    {"term": {"room": "alice"}},
                ]
            }
        }

    def test_or_operator(self):
        result = _build_es_filter({"$or": [{"wing": "a"}, {"wing": "b"}]})
        assert result == {
            "bool": {
                "should": [
                    {"term": {"wing": "a"}},
                    {"term": {"wing": "b"}},
                ],
                "minimum_should_match": 1,
            }
        }

    def test_empty_where(self):
        assert _build_es_filter({}) == {}

    def test_multiple_key_value_implicit_and(self):
        result = _build_es_filter({"wing": "x", "room": "y"})
        assert result == {
            "bool": {
                "filter": [
                    {"term": {"wing": "x"}},
                    {"term": {"room": "y"}},
                ]
            }
        }

    def test_in_operator(self):
        result = _build_es_filter({"chunk_index": {"$in": [0, 1, 2]}})
        assert result == {"terms": {"chunk_index": [0, 1, 2]}}

    def test_in_operator_inside_and(self):
        result = _build_es_filter(
            {"$and": [{"source_file": "f.py"}, {"chunk_index": {"$in": [0, 1]}}]}
        )
        assert result == {
            "bool": {
                "filter": [
                    {"term": {"source_file": "f.py"}},
                    {"terms": {"chunk_index": [0, 1]}},
                ]
            }
        }


# ── Distance conversion ────────────────────────────────────────────────


class TestScoreToDistance:
    def test_perfect_match(self):
        assert _es_score_to_chroma_distance(1.0) == 0.0

    def test_opposite(self):
        assert _es_score_to_chroma_distance(0.0) == 2.0

    def test_midpoint(self):
        assert _es_score_to_chroma_distance(0.5) == pytest.approx(1.0)


# ── Mock helpers ─────────────────────────────────────────────────────────


def _make_mock_es():
    """Return a MagicMock ES client for use in unit tests."""
    es = MagicMock()
    es.indices.exists.return_value = True
    es.count.return_value = {"count": 0}
    es.search.return_value = {"hits": {"hits": []}}
    return es


def _make_fake_embedder():
    """Return a mock Embedder that returns deterministic unit vectors."""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 384
    embedder.embed_batch.return_value = [[0.1] * 384]
    return embedder


# ── IndentiaGraphCollection — write ops ──────────────────────────────────


class TestIndentiaGraphCollectionUpsert:
    def test_upsert_calls_es_index(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        with patch.object(col, "_get_embedder", return_value=_make_fake_embedder()):
            col.upsert(documents=["hello world"], ids=["id1"], metadatas=[{"wing": "w"}])

        es.index.assert_called_once()
        call_kwargs = es.index.call_args.kwargs
        assert call_kwargs["id"] == "id1"
        assert call_kwargs["index"] == "test_index"
        doc = call_kwargs["document"]
        assert doc["content"] == "hello world"
        assert doc["wing"] == "w"
        assert "embedding" in doc
        assert call_kwargs["refresh"] == "wait_for"

    def test_add_delegates_to_upsert(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        with patch.object(col, "upsert") as mock_upsert:
            col.add(documents=["doc"], ids=["id1"], metadatas=[{"wing": "w"}])
            mock_upsert.assert_called_once_with(
                documents=["doc"], ids=["id1"], metadatas=[{"wing": "w"}]
            )

    def test_upsert_batch_calls_index_per_item(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        embedder = MagicMock()
        embedder.embed_batch.return_value = [[0.1] * 384, [0.2] * 384]
        with patch.object(col, "_get_embedder", return_value=embedder):
            col.upsert(
                documents=["doc1", "doc2"],
                ids=["id1", "id2"],
                metadatas=[{"wing": "a"}, {"wing": "b"}],
            )
        assert es.index.call_count == 2

    def test_upsert_without_metadatas(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        with patch.object(col, "_get_embedder", return_value=_make_fake_embedder()):
            col.upsert(documents=["doc"], ids=["id1"])

        doc = es.index.call_args.kwargs["document"]
        assert doc["content"] == "doc"


# ── IndentiaGraphCollection — query ──────────────────────────────────────


class TestIndentiaGraphCollectionQuery:
    def test_query_returns_chroma_format(self):
        es = _make_mock_es()
        es.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 0.9,
                        "_source": {"content": "result text", "wing": "people", "room": "alice"},
                    }
                ]
            }
        }
        col = IndentiaGraphCollection(es, "test_index")
        with patch.object(col, "_get_embedder", return_value=_make_fake_embedder()):
            result = col.query(query_texts=["test query"], n_results=5)

        assert result["ids"] == [["doc1"]]
        assert result["documents"] == [["result text"]]
        assert result["distances"][0][0] == pytest.approx(2.0 * (1.0 - 0.9))
        assert result["metadatas"][0][0]["wing"] == "people"
        assert result["metadatas"][0][0]["room"] == "alice"

    def test_query_empty_texts_returns_empty_without_calling_es(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        result = col.query(query_texts=[], n_results=5)
        assert result["ids"] == [[]]
        es.search.assert_not_called()

    def test_query_with_where_filter_passes_filter_to_knn(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        with patch.object(col, "_get_embedder", return_value=_make_fake_embedder()):
            col.query(query_texts=["q"], n_results=3, where={"wing": "people"})

        call_body = es.search.call_args.kwargs["body"]
        assert "filter" in call_body["knn"]
        assert call_body["knn"]["filter"] == {"term": {"wing": "people"}}

    def test_query_without_where_has_no_filter_in_knn(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        with patch.object(col, "_get_embedder", return_value=_make_fake_embedder()):
            col.query(query_texts=["q"], n_results=5)

        call_body = es.search.call_args.kwargs["body"]
        assert "filter" not in call_body["knn"]


# ── IndentiaGraphCollection — get ────────────────────────────────────────


class TestIndentiaGraphCollectionGet:
    def test_get_by_ids_uses_ids_query(self):
        es = _make_mock_es()
        es.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "id1", "_source": {"content": "doc", "wing": "w", "room": "r"}}
                ]
            }
        }
        col = IndentiaGraphCollection(es, "test_index")
        result = col.get(ids=["id1"])

        assert result["ids"] == ["id1"]
        assert result["documents"] == ["doc"]
        assert result["metadatas"][0]["wing"] == "w"

        call_body = es.search.call_args.kwargs["body"]
        assert call_body["query"] == {"ids": {"values": ["id1"]}}

    def test_get_by_where_uses_filter_query(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        col.get(where={"wing": "x"})

        call_body = es.search.call_args.kwargs["body"]
        assert call_body["query"] == {"term": {"wing": "x"}}

    def test_get_all_uses_match_all(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        col.get()

        call_body = es.search.call_args.kwargs["body"]
        assert call_body["query"] == {"match_all": {}}

    def test_get_pagination_passes_size_and_from(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        col.get(limit=50, offset=100)

        call_body = es.search.call_args.kwargs["body"]
        assert call_body["size"] == 50
        assert call_body["from"] == 100

    def test_get_returns_flat_lists(self):
        es = _make_mock_es()
        es.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "id1", "_source": {"content": "a", "wing": "w"}},
                    {"_id": "id2", "_source": {"content": "b", "wing": "w"}},
                ]
            }
        }
        col = IndentiaGraphCollection(es, "test_index")
        result = col.get()

        # get() returns flat lists, not nested lists like query()
        assert result["ids"] == ["id1", "id2"]
        assert result["documents"] == ["a", "b"]
        assert isinstance(result["metadatas"], list)
        assert len(result["metadatas"]) == 2


# ── IndentiaGraphCollection — delete ─────────────────────────────────────


class TestIndentiaGraphCollectionDelete:
    def test_delete_by_ids(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        col.delete(ids=["id1", "id2"])

        # delete() now calls es.delete() per ID (delete_by_query has a server
        # bug in IndentiaGraph — see IndentiaGraphCollection.delete docstring)
        assert es.delete.call_count == 2

    def test_delete_by_where(self):
        es = _make_mock_es()
        # Simulate one matching document returned by search
        es.search.return_value = {"hits": {"hits": [{"_id": "matched_id"}]}}
        col = IndentiaGraphCollection(es, "test_index")
        col.delete(where={"wing": "old_wing"})

        # Should have searched for IDs first, then deleted each one
        es.search.assert_called()
        es.delete.assert_called_once()

    def test_delete_noop_when_no_args(self):
        es = _make_mock_es()
        col = IndentiaGraphCollection(es, "test_index")
        col.delete()
        es.delete_by_query.assert_not_called()
        es.delete.assert_not_called()


# ── IndentiaGraphCollection — count ──────────────────────────────────────


class TestIndentiaGraphCollectionCount:
    def test_count_returns_integer(self):
        es = _make_mock_es()
        # count() now uses search(size=0) — mock the total from hits.total.value
        es.search.return_value = {"hits": {"hits": [], "total": {"value": 42, "relation": "eq"}}}
        col = IndentiaGraphCollection(es, "test_index")
        assert col.count() == 42

    def test_count_returns_zero_on_exception(self):
        es = _make_mock_es()
        es.search.side_effect = Exception("connection refused")
        col = IndentiaGraphCollection(es, "test_index")
        assert col.count() == 0


# ── IndentiaGraphBackend ─────────────────────────────────────────────────


class TestIndentiaGraphBackend:
    def test_get_collection_create_false_raises_when_index_missing(self):
        backend = IndentiaGraphBackend()
        mock_es = _make_mock_es()
        mock_es.indices.exists.return_value = False
        with patch.object(backend, "_client", return_value=mock_es):
            with pytest.raises(FileNotFoundError):
                backend.get_collection("/some/path", "mempalace_drawers", create=False)

    def test_get_collection_create_true_creates_index_when_missing(self, tmp_path):
        backend = IndentiaGraphBackend()
        mock_es = _make_mock_es()
        mock_es.indices.exists.return_value = False
        with patch.object(backend, "_client", return_value=mock_es):
            col = backend.get_collection(
                str(tmp_path / "palace"), "mempalace_drawers", create=True
            )

        mock_es.indices.create.assert_called_once()
        assert isinstance(col, IndentiaGraphCollection)

    def test_get_collection_create_true_skips_existing_index(self):
        backend = IndentiaGraphBackend()
        mock_es = _make_mock_es()
        mock_es.indices.exists.return_value = True
        with patch.object(backend, "_client", return_value=mock_es):
            col = backend.get_collection("/path", "mempalace_drawers", create=True)

        mock_es.indices.create.assert_not_called()
        assert isinstance(col, IndentiaGraphCollection)

    def test_index_name_is_lowercased_and_path_isolated(self):
        backend = IndentiaGraphBackend()
        mock_es = _make_mock_es()
        mock_es.indices.exists.return_value = True
        with patch.object(backend, "_client", return_value=mock_es):
            col = backend.get_collection("/path", "MemPalace_Drawers", create=False)

        # Index name must be lowercase and include a path-based hash suffix
        # so different palace directories never share the same ES index.
        assert col._index.startswith("mempalace_drawers_")
        assert col._index == col._index.lower()

    def test_index_mapping_uses_cosine_similarity(self):
        from mempalace.backends.indentiagraph import _INDEX_MAPPING

        props = _INDEX_MAPPING["mappings"]["properties"]
        assert props["embedding"]["similarity"] == "cosine"
        assert props["embedding"]["dims"] == 384
        assert props["embedding"]["index"] is True

    def test_get_or_create_delegates_to_get_collection(self):
        backend = IndentiaGraphBackend()
        with patch.object(backend, "get_collection") as mock_get:
            mock_get.return_value = MagicMock()
            backend.get_or_create_collection("/path", "mempalace_drawers")

        mock_get.assert_called_once_with("/path", "mempalace_drawers", create=True)
