"""Utilities for working with embedding functions and LangChain's Embeddings interface.

This module provides tools to wrap arbitrary embedding functions (both sync and async)
into LangChain's Embeddings interface. This enables using custom embedding functions
with LangChain-compatible tools while maintaining support for both synchronous and
asynchronous operations.
"""

import asyncio
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Union

from langchain_core.embeddings import Embeddings

EmbeddingsFunc = Callable[[Sequence[str]], list[list[float]]]
"""Type for synchronous embedding functions.

The function should take a sequence of strings and return a list of embeddings,
where each embedding is a list of floats. The dimensionality of the embeddings
should be consistent for all inputs.
"""

AEmbeddingsFunc = Callable[[Sequence[str]], Awaitable[list[list[float]]]]
"""Type for asynchronous embedding functions.

Similar to EmbeddingsFunc, but returns an awaitable that resolves to the embeddings.
"""


def ensure_embeddings(
    embed: Union[Embeddings, EmbeddingsFunc, AEmbeddingsFunc, None],
    *,
    aembed: Optional[AEmbeddingsFunc] = None,
) -> Embeddings:
    """Ensure that an embedding function conforms to LangChain's Embeddings interface.

    This function wraps arbitrary embedding functions to make them compatible with
    LangChain's Embeddings interface. It handles both synchronous and asynchronous
    functions.

    Args:
        embed: Either an existing Embeddings instance, or a function that converts
            text to embeddings. If the function is async, it will be used for both
            sync and async operations.
        aembed: Optional async function for embeddings. If provided, it will be used
            for async operations while the sync function is used for sync operations.
            Must be None if embed is async.

    Returns:
        An Embeddings instance that wraps the provided function(s).

    Example:
        >>> def my_embed_fn(texts): return [[0.1, 0.2] for _ in texts]
        >>> async def my_async_fn(texts): return [[0.1, 0.2] for _ in texts]
        >>> # Wrap a sync function
        >>> embeddings = ensure_embeddings(my_embed_fn)
        >>> # Wrap an async function
        >>> embeddings = ensure_embeddings(my_async_fn)
        >>> # Provide both sync and async implementations
        >>> embeddings = ensure_embeddings(my_embed_fn, aembed=my_async_fn)
    """
    if embed is None and aembed is None:
        raise ValueError("embed or aembed must be provided")
    if isinstance(embed, Embeddings):
        return embed
    return EmbeddingsLambda(embed, afunc=aembed)


class EmbeddingsLambda(Embeddings):
    """Wrapper to convert embedding functions into LangChain's Embeddings interface.

    This class allows arbitrary embedding functions to be used with LangChain-compatible
    tools. It supports both synchronous and asynchronous operations, and can be
    initialized with either:
    1. A synchronous function for both sync/async operations
    2. An async function for both sync/async operations
    3. Both sync and async functions for their respective operations

    The embedding functions should convert text into fixed-dimensional vectors that
    capture the semantic meaning of the text.

    Args:
        func: Function that converts text to embeddings. Can be sync or async.
            If async, it will be used for both sync and async operations.
        afunc: Optional async function for embeddings. If provided, it will be used
            for async operations while func is used for sync operations.
            Must be None if func is async.

    Example:
        >>> def my_embed_fn(texts):
        ...     # Return 2D embeddings for each text
        ...     return [[0.1, 0.2] for _ in texts]
        >>> embeddings = EmbeddingsLambda(my_embed_fn)
        >>> result = embeddings.embed_query("hello")  # Returns [0.1, 0.2]
    """

    def __init__(
        self,
        func: Union[EmbeddingsFunc, AEmbeddingsFunc, None],
        afunc: Optional[AEmbeddingsFunc] = None,
    ) -> None:
        if _is_async_callable(func):
            if afunc is not None:
                raise ValueError(
                    "afunc must be None if func is async. The async func will be used for both sync and async operations."
                )
            self.afunc = func
        else:
            self.func = func
            if afunc is not None:
                self.afunc = afunc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: List of texts to convert to embeddings.

        Returns:
            List of embeddings, one per input text. Each embedding is a list of floats.

        Raises:
            ValueError: If the instance was initialized with only an async function.
        """
        if not hasattr(self, "func"):
            raise ValueError(
                "EmbeddingsLambda was initialized with an async function but no sync function. "
                "Use aembed_documents for async operation or provide a sync function."
            )
        return self.func(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single piece of text.

        Args:
            text: Text to convert to an embedding.

        Returns:
            Embedding vector as a list of floats.

        Note:
            This is equivalent to calling embed_documents with a single text
            and taking the first result.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of texts into vectors.

        Args:
            texts: List of texts to convert to embeddings.

        Returns:
            List of embeddings, one per input text. Each embedding is a list of floats.

        Note:
            If no async function was provided, this falls back to the sync implementation.
        """
        if not hasattr(self, "afunc"):
            return await super().aembed_documents(texts)
        return await self.afunc(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single piece of text.

        Args:
            text: Text to convert to an embedding.

        Returns:
            Embedding vector as a list of floats.

        Note:
            This is equivalent to calling aembed_documents with a single text
            and taking the first result.
        """
        if not hasattr(self, "afunc"):
            return await super().aembed_query(text)
        return (await self.afunc([text]))[0]


def _is_async_callable(
    func: Any,
) -> bool:
    """Check if a function is async.

    This includes both async def functions and classes with async __call__ methods.

    Args:
        func: Function or callable object to check.

    Returns:
        True if the function is async, False otherwise.
    """
    return (
        asyncio.iscoroutinefunction(func)
        or hasattr(func, "__call__")  # noqa: B004
        and asyncio.iscoroutinefunction(func.__call__)
    )


__all__ = [
    "ensure_embeddings",
    "EmbeddingsFunc",
    "AEmbeddingsFunc",
]
