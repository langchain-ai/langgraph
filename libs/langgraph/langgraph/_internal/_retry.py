def default_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, (httpx.HTTPStatusError, requests.HTTPError)):
        # requests.Response.__bool__ is `.ok`, so status >= 400 is falsy.
        # Identity-check None so 4xx/5xx responses are inspected, not treated as missing.
        if exc.response is None:
            return True
        status = exc.response.status_code
        return status in (408, 429) or 500 <= status < 600
    if isinstance(
        exc,
        (
            ValueError,
            TypeError,
            ArithmeticError,
            ImportError,
            LookupError,
            NameError,
            SyntaxError,
            RuntimeError,
            ReferenceError,
            StopIteration,
            StopAsyncIteration,
            OSError,
        ),
    ):
        return False
    return True
