from langgraph_cli.cli import _inmem_python_version_note


def test_inmem_python_version_note_for_unsupported_old_python() -> None:
    note = _inmem_python_version_note((3, 10))
    assert "requires Python 3.11 or higher" in note
    assert "3.10" in note


def test_inmem_python_version_note_for_supported_python() -> None:
    assert _inmem_python_version_note((3, 11)) == ""
    assert _inmem_python_version_note((3, 13)) == ""


def test_inmem_python_version_note_for_unsupported_new_python() -> None:
    note = _inmem_python_version_note((3, 14))
    assert "supports Python 3.11-3.13" in note
    assert "3.14" in note
