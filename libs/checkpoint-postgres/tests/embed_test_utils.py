"""Embedding utilities for testing."""

import math
import random
from collections import Counter, defaultdict
from typing import Any

from langchain_core.embeddings import Embeddings


class CharacterEmbeddings(Embeddings):
    """Simple character-frequency based embeddings using random projections."""

    def __init__(self, dims: int = 50, seed: int = 42):
        """Initialize with embedding dimensions and random seed."""
        self._rng = random.Random(seed)
        self.dims = dims
        # Create projection vector for each character lazily
        self._char_projections: defaultdict[str, list[float]] = defaultdict(
            lambda: [
                self._rng.gauss(0, 1 / math.sqrt(self.dims)) for _ in range(self.dims)
            ]
        )

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        counts = Counter(text)
        total = sum(counts.values())

        if total == 0:
            return [0.0] * self.dims

        embedding = [0.0] * self.dims
        for char, count in counts.items():
            weight = count / total
            char_proj = self._char_projections[char]
            for i, proj in enumerate(char_proj):
                embedding[i] += weight * proj

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string."""
        return self._embed_one(text)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CharacterEmbeddings) and self.dims == other.dims
