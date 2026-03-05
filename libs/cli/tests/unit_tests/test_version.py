from langgraph_cli.version import __version__


def test_version_is_set():
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_version_no_importlib_metadata():
    import langgraph_cli.version as version_mod

    source = open(version_mod.__file__).read()
    assert "importlib" not in source
