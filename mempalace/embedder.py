"""Local embedding generator for MemPalace.

Uses sentence-transformers (all-MiniLM-L6-v2, 384 dims) to generate
embeddings locally — no API key required, no data leaves the machine.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIMS = 384

# Module-level singleton — loaded once on first use.
_embedder: "Embedder | None" = None


def get_embedder() -> "Embedder":
    """Return the module-level Embedder singleton, creating it if needed."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


class Embedder:
    """Thin wrapper around a sentence-transformers model.

    All embeddings are L2-normalised so cosine similarity equals dot product.
    """

    def __init__(self, model_name: str = _MODEL_NAME):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the IndentiaGraph backend. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model %s …", model_name)
        self._model = SentenceTransformer(model_name)
        self._dims = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model ready (%d dims)", self._dims)

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns a list of floats (length = dims)."""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings. Returns a list of float lists."""
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
