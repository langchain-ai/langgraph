def default_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    # `requests.RequestException` subclasses `OSError`, so requests' transient
    # network failures would otherwise be swallowed by the non-retryable
    # `OSError` branch below and never retried.
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        # `Response.__bool__` is an alias for `Response.ok`, so every error
        # response is falsy. A truthiness check here would send all of them --
        # 4xx included -- down the `else` branch and retry them.
        return (
            500 <= exc.response.status_code < 600 if exc.response is not None else True
        )
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
