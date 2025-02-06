import functools

from urllib3 import __version__ as urllib3version  # type: ignore[import-untyped]
from urllib3 import connection  # type: ignore[import-untyped]


def _ensure_str(s, encoding="utf-8", errors="strict") -> str:
    if isinstance(s, str):
        return s

    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    return str(s)


# Copied from https://github.com/urllib3/urllib3/blob/1c994dfc8c5d5ecaee8ed3eb585d4785f5febf6e/src/urllib3/connection.py#L231
def request(self, method, url, body=None, headers=None):
    """Make the request.

    This function is based on the urllib3 request method, with modifications
    to handle potential issues when using vcrpy in concurrent workloads.

    Args:
        self: The HTTPConnection instance.
        method (str): The HTTP method (e.g., 'GET', 'POST').
        url (str): The URL for the request.
        body (Optional[Any]): The body of the request.
        headers (Optional[dict]): Headers to send with the request.

    Returns:
        The result of calling the parent request method.
    """
    # Update the inner socket's timeout value to send the request.
    # This only triggers if the connection is reused.
    if getattr(self, "sock", None) is not None:
        self.sock.settimeout(self.timeout)

    if headers is None:
        headers = {}
    else:
        # Avoid modifying the headers passed into .request()
        headers = headers.copy()
    if "user-agent" not in (_ensure_str(k.lower()) for k in headers):
        headers["User-Agent"] = connection._get_default_user_agent()
    # The above is all the same ^^^
    # The following is different:
    return self._parent_request(method, url, body=body, headers=headers)


_PATCHED = False


def patch_urllib3():
    """Patch the request method of urllib3 to avoid type errors when using vcrpy.

    In concurrent workloads (such as the tracing background queue), the
    connection pool can get in a state where an HTTPConnection is created
    before vcrpy patches the HTTPConnection class. In urllib3 >= 2.0 this isn't
    a problem since they use the proper super().request(...) syntax, but in older
    versions, super(HTTPConnection, self).request is used, resulting in a TypeError
    since self is no longer a subclass of "HTTPConnection" (which at this point
    is vcr.stubs.VCRConnection).

    This method patches the class to fix the super() syntax to avoid mixed inheritance.
    In the case of the LangSmith tracing logic, it doesn't really matter since we always
    exclude cache checks for calls to LangSmith.

    The patch is only applied for urllib3 versions older than 2.0.
    """
    global _PATCHED
    if _PATCHED:
        return
    from packaging import version

    if version.parse(urllib3version) >= version.parse("2.0"):
        _PATCHED = True
        return

    # Lookup the parent class and its request method
    parent_class = connection.HTTPConnection.__bases__[0]
    parent_request = parent_class.request

    def new_request(self, *args, **kwargs):
        """Handle parent request.

        This method binds the parent's request method to self and then
        calls our modified request function.
        """
        self._parent_request = functools.partial(parent_request, self)
        return request(self, *args, **kwargs)

    connection.HTTPConnection.request = new_request
    _PATCHED = True
