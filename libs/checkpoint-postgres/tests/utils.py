import math
import random
from collections import Counter
from typing import Any, Optional

from langchain_core.embeddings import Embeddings


class CharacterEmbeddings(Embeddings):
    """Simple character-frequency based embeddings using random projections."""

    def __init__(self, dims: int = 50, seed: int = 42):
        """Initialize with embedding dimensions and random seed."""
        self._rng = random.Random(seed)
        self._char_to_idx: dict[str, int] = {}
        self._projection: Optional[list[list[float]]] = None
        self.dims = dims

    def _ensure_projection_matrix(self, texts: list[str]) -> None:
        """Lazily initialize character mapping and projection matrix."""
        if self._projection is None:
            chars = sorted(set("".join(texts)))
            self._char_to_idx = {c: i for i, c in enumerate(chars)}
            self._projection = [
                [self._rng.gauss(0, 1 / math.sqrt(self.dims)) for _ in range(self.dims)]
                for _ in range(len(chars))
            ]

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        counts = Counter(text)
        char_vec = [0.0] * len(self._char_to_idx)

        for char, count in counts.items():
            if char in self._char_to_idx:
                char_vec[self._char_to_idx[char]] = count

        total = sum(char_vec)
        if total > 0:
            char_vec = [v / total for v in char_vec]
        embedding = [
            sum(a * b for a, b in zip(char_vec, proj))
            for proj in zip(*self._projection)
        ]

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        self._ensure_projection_matrix(texts)
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string."""
        self._ensure_projection_matrix([text])
        return self._embed_one(text)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CharacterEmbeddings) and self.dims == other.dims
