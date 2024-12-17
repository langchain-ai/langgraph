"""Exceptions used in the auth system."""

import http
import typing


class HTTPException(Exception):
    """HTTP exception that you can raise to return a specific HTTP error response.

    Since this is defined in the auth module, we default to a 401 status code.

    Args:
        status_code (int, optional): HTTP status code for the error. Defaults to 401 "Unauthorized".
        detail (str | None, optional): Detailed error message. If None, uses a default
            message based on the status code.
        headers (typing.Mapping[str, str] | None, optional): Additional HTTP headers to
            include in the error response.

    Attributes:
        status_code (int): The HTTP status code of the error
        detail (str): The error message or description
        headers (typing.Mapping[str, str] | None): Additional HTTP headers

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
        detail: typing.Optional[str] = None,
        headers: typing.Optional[typing.Mapping[str, str]] = None,
    ) -> None:
        if detail is None:
            detail = http.HTTPStatus(status_code).phrase
        self.status_code = status_code
        self.detail = detail
        self.headers = headers

    def __str__(self) -> str:
        """Return a string representation of the HTTP exception.

        Returns:
            str: A string in the format 'status_code: detail'
        """
        return f"{self.status_code}: {self.detail}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the HTTP exception.

        Returns:
            str: A string representation showing the class name and all attributes
        """
        class_name = self.__class__.__name__
        return f"{class_name}(status_code={self.status_code!r}, detail={self.detail!r})"


__all__ = ["HTTPException"]
