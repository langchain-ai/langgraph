import importlib
import re
import sys

import importlib.metadata as importlib_metadata


def test_version_is_semver_string() -> None:
    from langgraph.version import __version__

    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[abrc]\d+)?", __version__)


def test_version_import_does_not_call_importlib_metadata(monkeypatch) -> None:
    sys.modules.pop("langgraph.version", None)

    def _raise_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("importlib.metadata.version should not be called")

    monkeypatch.setattr(importlib_metadata, "version", _raise_if_called)

    module = importlib.import_module("langgraph.version")
    assert module.__version__
