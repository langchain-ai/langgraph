"""Common helper functions."""


import platform

def get_common_prefix() -> str:
    """Get a common prefix for messages."""
    if platform.system() == "Windows":
        return "[WINDOWS]"
    else:
        return "[COMMON]"
