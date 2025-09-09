"""Utilities for working with embedding functions and LangChain's Embeddings interface.

This module provides tools to wrap arbitrary embedding functions (both sync and async)
into LangChain's Embeddings interface. This enables using custom embedding functions
with LangChain-compatible tools while maintaining support for both synchronous and
asynchronous operations.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Awaitable, Sequence
from typing import Any, Callable

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
    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str | None,
) -> Embeddings:
    """Ensure that an embedding function conforms to LangChain's Embeddings interface.

    This function wraps arbitrary embedding functions to make them compatible with
    LangChain's Embeddings interface. It handles both synchronous and asynchronous
    functions.

    Args:
        embed: Either an existing Embeddings instance, or a function that converts
            text to embeddings. If the function is async, it will be used for both
            sync and async operations.

    Returns:
        An Embeddings instance that wraps the provided function(s).

    ??? example "Examples"
        Wrap a synchronous embedding function:
        ```python
        def my_embed_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = ensure_embeddings(my_embed_fn)
        result = embeddings.embed_query("hello")  # Returns [0.1, 0.2]
        ```

        Wrap an asynchronous embedding function:
        ```python
        async def my_async_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = ensure_embeddings(my_async_fn)
        result = await embeddings.aembed_query("hello")  # Returns [0.1, 0.2]
        ```

        Initialize embeddings using a provider string:
        ```python
        # Requires langchain>=0.3.9 and langgraph-checkpoint>=2.0.11
        embeddings = ensure_embeddings("openai:text-embedding-3-small")
        result = embeddings.embed_query("hello")
        ```
    """
    if embed is None:
        raise ValueError("embed must be provided")
    if isinstance(embed, str):
        init_embeddings = _get_init_embeddings()
        if init_embeddings is None:
            from importlib.metadata import PackageNotFoundError, version

            try:
                lc_version = version("langchain")
                version_info = f"Found langchain version {lc_version}, but"
            except PackageNotFoundError:
                version_info = "langchain is not installed;"

            raise ValueError(
                f"Could not load embeddings from string '{embed}'. {version_info} "
                "loading embeddings by provider:identifier string requires langchain>=0.3.9 "
                "as well as the provider-specific package. "
                "Install LangChain with: pip install 'langchain>=0.3.9' "
                "and the provider-specific package (e.g., 'langchain-openai>=0.3.0'). "
                "Alternatively, specify 'embed' as a compatible Embeddings object or python function."
            )
        return init_embeddings(embed)

    if isinstance(embed, Embeddings):
        return embed
    return EmbeddingsLambda(embed)


class EmbeddingsLambda(Embeddings):
    """Wrapper to convert embedding functions into LangChain's Embeddings interface.

    This class allows arbitrary embedding functions to be used with LangChain-compatible
    tools. It supports both synchronous and asynchronous operations, and can handle:
    1. A synchronous function for sync operations (async operations will use sync function)
    2. An async function for both sync/async operations (sync operations will raise an error)

    The embedding functions should convert text into fixed-dimensional vectors that
    capture the semantic meaning of the text.

    Args:
        func: Function that converts text to embeddings. Can be sync or async.
            If async, it will be used for async operations, but sync operations
            will raise an error. If sync, it will be used for both sync and async operations.

    ??? example "Examples"
        With a sync function:
        ```python
        def my_embed_fn(texts):
            # Return 2D embeddings for each text
            return [[0.1, 0.2] for _ in texts]

        embeddings = EmbeddingsLambda(my_embed_fn)
        result = embeddings.embed_query("hello")  # Returns [0.1, 0.2]
        await embeddings.aembed_query("hello")  # Also returns [0.1, 0.2]
        ```

        With an async function:
        ```python
        async def my_async_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = EmbeddingsLambda(my_async_fn)
        await embeddings.aembed_query("hello")  # Returns [0.1, 0.2]
        # Note: embed_query() would raise an error
        ```
    """

    def __init__(
        self,
        func: EmbeddingsFunc | AEmbeddingsFunc,
    ) -> None:
        if func is None:
            raise ValueError("func must be provided")
        if _is_async_callable(func):
            self.afunc = func
        else:
            self.func = func

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: list of texts to convert to embeddings.

        Returns:
            list of embeddings, one per input text. Each embedding is a list of floats.

        Raises:
            ValueError: If the instance was initialized with only an async function.
        """
        func = getattr(self, "func", None)
        if func is None:
            raise ValueError(
                "EmbeddingsLambda was initialized with an async function but no sync function. "
                "Use aembed_documents for async operation or provide a sync function."
            )
        return func(texts)

    def embed_query(self, text: str) -> list[float]:
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

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed a list of texts into vectors.

        Args:
            texts: list of texts to convert to embeddings.

        Returns:
            list of embeddings, one per input text. Each embedding is a list of floats.

        Note:
            If no async function was provided, this falls back to the sync implementation.
        """
        afunc = getattr(self, "afunc", None)
        if afunc is None:
            return await super().aembed_documents(texts)
        return await afunc(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embed a single piece of text.

        Args:
            text: Text to convert to an embedding.

        Returns:
            Embedding vector as a list of floats.

        Note:
            This is equivalent to calling aembed_documents with a single text
            and taking the first result.
        """
        afunc = getattr(self, "afunc", None)
        if afunc is None:
            return await super().aembed_query(text)
        return (await afunc([text]))[0]


def get_text_at_path(obj: Any, path: str | list[str]) -> list[str]:
    """Extract text from an object using a path expression or pre-tokenized path.

    Args:
        obj: The object to extract text from
        path: Either a path string or pre-tokenized path list.

    !!! info "Path types handled"
        - Simple paths: "field1.field2"
        - Array indexing: "[0]", "[*]", "[-1]"
        - Wildcards: "*"
        - Multi-field selection: "{field1,field2}"
        - Nested paths in multi-field: "{field1,nested.field2}"
    """
    if not path or path == "$":
        return [json.dumps(obj, sort_keys=True, ensure_ascii=False)]

    tokens = tokenize_path(path) if isinstance(path, str) else path

    def _extract_from_obj(obj: Any, tokens: list[str], pos: int) -> list[str]:
        if pos >= len(tokens):
            if isinstance(obj, (str, int, float, bool)):
                return [str(obj)]
            elif obj is None:
                return []
            elif isinstance(obj, (list, dict)):
                return [json.dumps(obj, sort_keys=True, ensure_ascii=False)]
            return []

        token = tokens[pos]
        results = []

        if token.startswith("[") and token.endswith("]"):
            if not isinstance(obj, list):
                return []

            index = token[1:-1]
            if index == "*":
                for item in obj:
                    results.extend(_extract_from_obj(item, tokens, pos + 1))
            else:
                try:
                    idx = int(index)
                    if idx < 0:
                        idx = len(obj) + idx
                    if 0 <= idx < len(obj):
                        results.extend(_extract_from_obj(obj[idx], tokens, pos + 1))
                except (ValueError, IndexError):
                    return []

        elif token.startswith("{") and token.endswith("}"):
            if not isinstance(obj, dict):
                return []

            fields = [f.strip() for f in token[1:-1].split(",")]
            for field in fields:
                nested_tokens = tokenize_path(field)
                if nested_tokens:
                    current_obj: dict | None = obj
                    for nested_token in nested_tokens:
                        if (
                            isinstance(current_obj, dict)
                            and nested_token in current_obj
                        ):
                            current_obj = current_obj[nested_token]
                        else:
                            current_obj = None
                            break
                    if current_obj is not None:
                        if isinstance(current_obj, (str, int, float, bool)):
                            results.append(str(current_obj))
                        elif isinstance(current_obj, (list, dict)):
                            results.append(
                                json.dumps(
                                    current_obj, sort_keys=True, ensure_ascii=False
                                )
                            )

        # Handle wildcard
        elif token == "*":
            if isinstance(obj, dict):
                for value in obj.values():
                    results.extend(_extract_from_obj(value, tokens, pos + 1))
            elif isinstance(obj, list):
                for item in obj:
                    results.extend(_extract_from_obj(item, tokens, pos + 1))

        # Handle regular field
        else:
            if isinstance(obj, dict) and token in obj:
                results.extend(_extract_from_obj(obj[token], tokens, pos + 1))

        return results

    return _extract_from_obj(obj, tokens, 0)


# Private utility functions


def tokenize_path(path: str) -> list[str]:
    """Tokenize a path into components.

    !!! info "Types handled"
        - Simple paths: "field1.field2"
        - Array indexing: "[0]", "[*]", "[-1]"
        - Wildcards: "*"
        - Multi-field selection: "{field1,field2}"
    """
    if not path:
        return []

    tokens = []
    current: list[str] = []
    i = 0
    while i < len(path):
        char = path[i]

        if char == "[":  # Handle array index
            if current:
                tokens.append("".join(current))
                current = []
            bracket_count = 1
            index_chars = ["["]
            i += 1
            while i < len(path) and bracket_count > 0:
                if path[i] == "[":
                    bracket_count += 1
                elif path[i] == "]":
                    bracket_count -= 1
                index_chars.append(path[i])
                i += 1
            tokens.append("".join(index_chars))
            continue

        elif char == "{":  # Handle multi-field selection
            if current:
                tokens.append("".join(current))
                current = []
            brace_count = 1
            field_chars = ["{"]
            i += 1
            while i < len(path) and brace_count > 0:
                if path[i] == "{":
                    brace_count += 1
                elif path[i] == "}":
                    brace_count -= 1
                field_chars.append(path[i])
                i += 1
            tokens.append("".join(field_chars))
            continue

        elif char == ".":  # Handle regular field
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
        i += 1

    if current:
        tokens.append("".join(current))

    return tokens


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


@functools.lru_cache
def _get_init_embeddings() -> Callable[[str], Embeddings] | None:
    try:
        from langchain.embeddings import init_embeddings  # type: ignore

        return init_embeddings
    except ImportError:
        return None


__all__ = [
    "ensure_embeddings",
    "EmbeddingsFunc",
    "AEmbeddingsFunc",
]
