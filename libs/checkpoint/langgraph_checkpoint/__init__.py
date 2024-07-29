try:
    from importlib import metadata

    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = "unknown"
