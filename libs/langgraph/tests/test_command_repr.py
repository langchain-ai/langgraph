import threading

from langgraph.types import Command


def test_command_repr_with_unpicklable_update() -> None:
    """Command.__repr__ must not deep-copy via asdict (issue #9175)."""
    cmd = Command(update={"lock": threading.Lock(), "status": "active"})
    text = repr(cmd)
    assert text.startswith("Command(")
    assert "status" in text
    assert "lock" in text


def test_command_repr_omits_defaults() -> None:
    cmd = Command(update={"ok": True})
    assert repr(cmd) == "Command(update={'ok': True})"
