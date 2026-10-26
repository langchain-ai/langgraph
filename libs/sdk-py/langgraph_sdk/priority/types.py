"""Priority types for LangGraph.

This module defines the context the server passes to the function registered
with `@priority.assign`, and the settings that function returns.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence
from dataclasses import dataclass

from langgraph_sdk.auth.types import BaseUser


@dataclass(frozen=True, slots=True)
class PriorityContext:
    """Context passed to the `@priority.assign` function at the first run on a thread with no settings."""

    user: BaseUser
    """The user from the deployment's auth, custom auth or LangSmith auth. Without auth, `user.identity` is empty."""

    permissions: Sequence[str]
    """The user's permissions from the same auth. Without auth, it's empty."""

    thread_id: str | None
    """The thread's ID, or `None` when the run creates its thread without being given an ID."""

    thread_metadata: dict[str, typing.Any]
    """The thread's metadata."""

    assistant_id: str
    """The ID of the assistant the run uses."""

    graph_id: str
    """The ID of the graph the assistant runs."""

    assistant_metadata: dict[str, typing.Any]
    """The assistant's metadata."""

    run_metadata: dict[str, typing.Any]
    """The run's metadata."""

    cron_id: str | None
    """The ID of the cron that started the run, or `None` when no cron started it."""


class PrioritySettings(typing.TypedDict, total=False):
    """Priority settings returned by the `@priority.assign` function.

    Every key is optional: a missing key or a key set to `None` takes its
    default, and returning `None` instead of a dict gives every key its default.
    A `priority` outside -5 to 5, or a `fairness_group` that is empty or longer
    than 256 characters, is an error.
    """

    priority: int | None
    """An integer from -5 to 5, where 5 runs first and -5 runs last. Defaults to 0."""

    fairness_group: str | None
    """The thread's fairness group, a non-empty string of at most 256 characters.

    Defaults to `"default"`.

    The server stores it unencrypted, even when the deployment uses custom
    encryption, so use an opaque ID rather than personal data.
    """
