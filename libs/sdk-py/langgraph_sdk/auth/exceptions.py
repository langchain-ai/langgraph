"""Exceptions used in the auth system."""

from __future__ import annotations

import http
from collections.abc import Mapping


class HTTPException(Exception):
    """HTTP exception that you can raise to return a specific HTTP error response.

    Since this is defined in the auth module, we default to a 401 status code.

    Args:
        status_code: HTTP status code for the error. Defaults to 401 "Unauthorized".
        detail: Detailed error message. If `None`, uses a default
            message based on the status code.
        headers: Additional HTTP headers to include in the error response.

    Example:
        Default:
        ```python
        raise HTTPException()
        # HTTPException(status_code=401, detail='Unauthorized')
        ```

        Add headers:
        ```python
        raise HTTPException(headers={"X-Custom-Header": "Custom Value"})
        # HTTPException(status_code=401, detail='Unauthorized', headers={"WWW-Authenticate": "Bearer"})
        ```

        Custom error:
        ```python
        raise HTTPException(status_code=404, detail="Not found")
        ```
    """

    def __init__(
        self,
        status_code: int = 401,
        detail: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if detail is None:
            detail = http.HTTPStatus(status_code).phrase
        self.status_code = status_code
        self.detail = detail
        self.headers = headers

    def __str__(self) -> str:
        return f"{self.status_code}: {self.detail}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(status_code={self.status_code!r}, detail={self.detail!r})"


__all__ = ["HTTPException"]
