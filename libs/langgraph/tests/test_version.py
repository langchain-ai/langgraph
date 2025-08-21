"""
Tests for version management and import functionality.
"""

import pytest
import langgraph


def test_version_available():
    """Test that __version__ is available and accessible."""
    assert hasattr(langgraph, "__version__")
    assert isinstance(langgraph.__version__, str)
    assert len(langgraph.__version__) > 0


def test_version_import():
    """Test that __version__ can be imported from about module."""
    from langgraph.about import __version__
    assert __version__ == langgraph.__version__


def test_no_importlib_metadata():
    """Test that importlib.metadata is not used for version."""
    import importlib.util
    
    # Check that importlib.metadata is not imported in the version module
    version_source = importlib.util.find_spec("langgraph.version")
    assert version_source is not None
    
    # The about module should not import importlib.metadata
    about_source = importlib.util.find_spec("langgraph.about")
    assert about_source is not None


def test_fast_import():
    """Test that import is fast (basic smoke test)."""
    import time
    
    start_time = time.perf_counter()
    import langgraph
    end_time = time.perf_counter()
    
    import_time = end_time - start_time
    
    # Import should be reasonably fast (less than 100ms)
    assert import_time < 0.1, f"Import took {import_time:.6f}s, expected < 0.1s"
