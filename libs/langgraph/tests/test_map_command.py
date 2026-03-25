"""Tests for map_command in langgraph.pregel._io."""

from langgraph._internal._constants import NULL_TASK_ID, RESUME
from langgraph.constants import START
from langgraph.pregel._io import map_command
from langgraph.types import Command


def test_map_command_update_with_dict() -> None:
    """Standard dict update should produce writes."""
    cmd = Command(update={"key": "value"})
    results = list(map_command(cmd))
    assert (NULL_TASK_ID, "key", "value") in results


def test_map_command_update_with_zero() -> None:
    """update=0 is a valid falsy value and must not be silently dropped."""
    cmd = Command(update=0)
    results = list(map_command(cmd))
    assert (NULL_TASK_ID, "__root__", 0) in results


def test_map_command_update_with_false() -> None:
    """update=False is a valid falsy value and must not be silently dropped."""
    cmd = Command(update=False)
    results = list(map_command(cmd))
    assert (NULL_TASK_ID, "__root__", False) in results


def test_map_command_update_with_empty_string() -> None:
    """update='' is a valid falsy value and must not be silently dropped."""
    cmd = Command(update="")
    results = list(map_command(cmd))
    assert (NULL_TASK_ID, "__root__", "") in results


def test_map_command_update_with_empty_list() -> None:
    """update=[] is falsy but not None, so map_command enters the update branch.

    _update_as_tuples interprets [] as an empty sequence of (key, value) tuples,
    so no writes are produced — but crucially the branch is still entered.
    """
    cmd = Command(update=[])
    results = list(map_command(cmd))
    # Empty tuple-list yields nothing, which is correct
    assert results == []


def test_map_command_update_with_empty_dict() -> None:
    """update={} is a valid falsy value and must not be silently dropped."""
    cmd = Command(update={})
    results = list(map_command(cmd))
    assert results == []  # empty dict → no items


def test_map_command_update_none_produces_no_writes() -> None:
    """update=None is the sentinel and must not produce any writes."""
    cmd = Command(update=None)
    results = list(map_command(cmd))
    # Should have no update-related writes
    assert all(r[1] != "__root__" for r in results)


def test_map_command_goto_and_update() -> None:
    """Both goto and update should produce their respective writes."""
    cmd = Command(goto="node_a", update={"counter": 42})
    results = list(map_command(cmd))
    assert (NULL_TASK_ID, "branch:to:node_a", START) in results
    assert (NULL_TASK_ID, "counter", 42) in results


def test_map_command_resume_and_update() -> None:
    """Both resume and update should produce their respective writes."""
    cmd = Command(resume="yes", update={"status": "done"})
    results = list(map_command(cmd))
    assert (NULL_TASK_ID, RESUME, "yes") in results
    assert (NULL_TASK_ID, "status", "done") in results


def test_map_command_update_with_tuples() -> None:
    """update as list of (str, value) tuples should produce writes."""
    cmd = Command(update=[("a", 1), ("b", 2)])
    results = list(map_command(cmd))
    assert (NULL_TASK_ID, "a", 1) in results
    assert (NULL_TASK_ID, "b", 2) in results
