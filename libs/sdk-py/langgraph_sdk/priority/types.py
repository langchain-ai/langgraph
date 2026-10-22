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
    """Context passed to the `@priority.assign` function at a thread's first run."""

    user: BaseUser
    """The authenticated user. Without custom auth, `identity` can be empty."""

    permissions: Sequence[str]
    """The permissions granted to the authenticated user."""

    thread_id: str | None
    """The thread's id, or `None` when the run creates its thread without being given an id."""

    thread_metadata: dict[str, typing.Any]
    """The thread's metadata."""

    assistant_id: str
    """The id of the assistant the run uses."""

    graph_id: str
    """The id of the graph the assistant runs."""

    assistant_metadata: dict[str, typing.Any]
    """The assistant's metadata."""

    run_metadata: dict[str, typing.Any]
    """The run's metadata."""

    cron_id: str | None
    """The id of the cron that started the run, or `None` when no cron started it."""


class PrioritySettings(typing.TypedDict, total=False):
    """Priority settings returned by the `@priority.assign` function.

    Every key is optional. A missing key, a key set to `None`, or a `None`
    return in place of the dict takes that key's default. A value outside its
    range, or an empty `fairness_group`, is an error.
    """

    priority: int | None
    """An integer from -5 to 5, where 5 runs first and -5 runs last. Defaults to 0.

    This is the opposite of Temporal's `priority_key`, where 1 runs first.
    """

    fairness_group: str | None
    """The thread's fairness group, a non-empty string. Defaults to `"default"`.

    The server stores it unencrypted, even when the deployment uses custom
    encryption, so use an opaque id, never personal data such as a name or an
    email address.
    """

    fairness_weight: int | None
    """The weight of the thread's fairness group, an integer from 1 to 100. Defaults to 1."""
