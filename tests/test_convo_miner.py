import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from mempalace.convo_miner import mine_convos
from mempalace.palace import file_already_mined, NORMALIZE_VERSION


class _FakeCollection:
    """In-memory collection that implements the MemPalace collection interface.

    Used in place of a real backend so tests don't require a running server.
    """

    def __init__(self):
        self._docs: dict = {}  # id -> {content, **meta}

    def _matches(self, doc: dict, where: dict) -> bool:
        if "$and" in where:
            return all(self._matches(doc, c) for c in where["$and"])
        if "$or" in where:
            return any(self._matches(doc, c) for c in where["$or"])
        return all(doc.get(k) == v for k, v in where.items())

    def add(self, *, documents, ids, metadatas=None):
        self.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def upsert(self, *, documents, ids, metadatas=None):
        metas = metadatas or [{}] * len(documents)
        for doc, doc_id, meta in zip(documents, ids, metas):
            self._docs[doc_id] = {"content": doc, **meta}

    def update(self, *, ids, documents=None, metadatas=None):
        for i, doc_id in enumerate(ids):
            if doc_id not in self._docs:
                raise ValueError(f"ID not found: {doc_id}")
            if documents is not None:
                self._docs[doc_id]["content"] = documents[i]
            if metadatas is not None:
                self._docs[doc_id].update(metadatas[i])

    def get(self, **kwargs):
        where = kwargs.get("where")
        limit = kwargs.get("limit", 10000)
        offset = kwargs.get("offset", 0)
        ids_filter = kwargs.get("ids")

        items = list(self._docs.items())
        if ids_filter:
            items = [(k, v) for k, v in items if k in ids_filter]
        if where:
            items = [(k, v) for k, v in items if self._matches(v, where)]
        items = items[offset : offset + limit]

        result_ids = [k for k, _ in items]
        result_docs = [v["content"] for _, v in items]
        result_metas = [
            {fk: fv for fk, fv in v.items() if fk != "content"} for _, v in items
        ]
        return {"ids": result_ids, "documents": result_docs, "metadatas": result_metas}

    def query(self, **kwargs):
        n = kwargs.get("n_results", 5)
        items = list(self._docs.items())[:n]
        ids_row = [k for k, _ in items]
        docs_row = [v["content"] for _, v in items]
        metas_row = [
            {fk: fv for fk, fv in v.items() if fk != "content"} for _, v in items
        ]
        return {
            "ids": [ids_row],
            "documents": [docs_row],
            "metadatas": [metas_row],
            "distances": [[0.1] * len(items)],
        }

    def delete(self, **kwargs):
        ids = kwargs.get("ids")
        where = kwargs.get("where")
        if ids:
            for id_ in ids:
                self._docs.pop(id_, None)
        elif where:
            to_del = [k for k, v in self._docs.items() if self._matches(v, where)]
            for k in to_del:
                del self._docs[k]

    def count(self):
        return len(self._docs)


def test_convo_mining():
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "chat.txt"), "w") as f:
            f.write(
                "> What is memory?\nMemory is persistence.\n\n"
                "> Why does it matter?\nIt enables continuity.\n\n"
                "> How do we build it?\nWith structured storage.\n"
            )

        palace_path = os.path.join(tmpdir, "palace")
        fake_col = _FakeCollection()
        with patch("mempalace.convo_miner.get_collection", return_value=fake_col):
            mine_convos(tmpdir, palace_path, wing="test_convos")

        assert fake_col.count() >= 2

        # Verify documents were actually stored
        results = fake_col.query(query_texts=["memory persistence"], n_results=1)
        assert len(results["documents"][0]) > 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_does_not_reprocess_short_files(capsys):
    """Files below MIN_CHUNK_SIZE get a sentinel so they are skipped on re-run."""
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "tiny.txt"), "w") as f:
            f.write("hi")

        palace_path = os.path.join(tmpdir, "palace")
        fake_col = _FakeCollection()

        with patch("mempalace.convo_miner.get_collection", return_value=fake_col):
            # First run -- file is processed (sentinel written)
            mine_convos(tmpdir, palace_path, wing="test")
            capsys.readouterr()  # drain output

            # Verify sentinel was written (resolve path -- macOS /var -> /private/var)
            resolved_file = str(Path(tmpdir).resolve() / "tiny.txt")
            assert file_already_mined(fake_col, resolved_file)

            # Second run -- file should be skipped
            mine_convos(tmpdir, palace_path, wing="test")
            out2 = capsys.readouterr().out
            assert "Files skipped (already filed): 1" in out2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_does_not_reprocess_empty_chunk_files(capsys):
    """Files that normalize but produce 0 exchange chunks get a sentinel."""
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "no_exchanges.txt"), "w") as f:
            f.write("This is a plain paragraph without any exchange markers. " * 5)

        palace_path = os.path.join(tmpdir, "palace")
        fake_col = _FakeCollection()

        with patch("mempalace.convo_miner.get_collection", return_value=fake_col):
            mine_convos(tmpdir, palace_path, wing="test")
            mine_convos(tmpdir, palace_path, wing="test")
            out2 = capsys.readouterr().out
            assert "Files skipped (already filed): 1" in out2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_rebuilds_stale_drawers_after_schema_bump(capsys):
    """When stored drawers have an older normalize_version, the next mine
    silently purges them and refiles — no manual erase required.

    This is what makes the strip_noise upgrade apply to existing corpora:
    users just run `mempalace mine` again and old noise-filled drawers get
    replaced with clean ones."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "chat.txt"
        convo_path.write_text(
            "> What is memory?\nMemory is persistence.\n\n"
            "> Why does it matter?\nIt enables continuity.\n\n"
            "> How do we build it?\nWith structured storage.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")
        resolved = str(Path(tmpdir).resolve() / "chat.txt")
        fake_col = _FakeCollection()

        with patch("mempalace.convo_miner.get_collection", return_value=fake_col):
            # First mine — stamps drawers with NORMALIZE_VERSION
            mine_convos(tmpdir, palace_path, wing="test")
            capsys.readouterr()

            first_pass = fake_col.get(where={"source_file": resolved})
            first_ids = set(first_pass["ids"])
            assert first_ids, "first mine should produce drawers"
            for meta in first_pass["metadatas"]:
                assert meta.get("normalize_version") == NORMALIZE_VERSION

            # Simulate pre-v2 drawers: rewrite metadata to an older version,
            # and replace content with "noise" so we can see it get cleaned up.
            stale_metas = []
            for meta in first_pass["metadatas"]:
                stale = dict(meta)
                stale["normalize_version"] = 1
                stale_metas.append(stale)
            fake_col.update(
                ids=list(first_pass["ids"]),
                documents=["STALE NOISE"] * len(first_pass["ids"]),
                metadatas=stale_metas,
            )
            # Add an extra orphan drawer that should also be purged.
            fake_col.add(
                ids=["orphan_drawer"],
                documents=["OLD ORPHAN"],
                metadatas=[
                    {
                        "wing": "test",
                        "room": "default",
                        "source_file": resolved,
                        "chunk_index": 999,
                        "normalize_version": 1,
                    }
                ],
            )

            # Second mine — version gate should trigger rebuild
            mine_convos(tmpdir, palace_path, wing="test")
            out = capsys.readouterr().out
            assert (
                "Files skipped (already filed): 0" in out
            ), "stale drawers should force a rebuild, not a skip"

            rebuilt = fake_col.get(where={"source_file": resolved})
            # Orphan is gone
            assert "orphan_drawer" not in rebuilt["ids"]
            # No stale content survived
            assert all("STALE NOISE" not in d for d in rebuilt["documents"])
            assert all("OLD ORPHAN" not in d for d in rebuilt["documents"])
            # All rebuilt drawers carry the current version
            for meta in rebuilt["metadatas"]:
                assert meta.get("normalize_version") == NORMALIZE_VERSION
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
