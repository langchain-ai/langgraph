"""Ensure everything advertised in ``__all__`` is actually importable.

This is a drift-guard for the public API: if a symbol is renamed or removed
without updating ``__all__`` (or the re-export breaks), CI fails here instead
of the docs/tutorials breaking silently downstream.

Related issue: #5810
"""
import importlib

# Public packages we expect to be importable in a normal install. Packages
# that are not installed (e.g. optional extras) are skipped rather than failing.
_PUBLIC_PACKAGES = [
    "langgraph",
    "langgraph.graph",
    "langgraph.prebuilt",
    "langgraph.checkpoint",
    "langgraph.store",
    "langgraph.sdk",
]


def _iter_public_modules():
    for name in _PUBLIC_PACKAGES:
        try:
            yield importlib.import_module(name)
        except ImportError:
            continue


def test_public_symbols_importable():
    missing = []
    for mod in _iter_public_modules():
        for sym in getattr(mod, "__all__", []):
            try:
                getattr(mod, sym)
            except AttributeError:
                missing.append(f"{mod.__name__}.{sym}")
    assert not missing, "Non-importable public symbols: " + ", ".join(missing)
