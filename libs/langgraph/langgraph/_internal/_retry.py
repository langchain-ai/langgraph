def default_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        # `requests.Response.__bool__` is an alias for `Response.ok`, so every
        # error response is falsy. Compare against `None` explicitly to reach
        # the status-code check for real error responses.
        if exc.response is not None:
            return 500 <= exc.response.status_code < 600
        return True
    if isinstance(exc, requests.RequestException):
        # `requests.RequestException` subclasses `OSError`, so connection errors
        # and timeouts would otherwise fall into the non-retryable `OSError`
        # branch below. Treat these transient failures as retryable.
        return True
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
