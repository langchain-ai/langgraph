"""Utility to convert a user provided function into a Runnable with a ChannelWrite."""

from __future__ import annotations

import concurrent.futures
import functools
import inspect
import sys
import types
from collections.abc import Awaitable, Callable, Generator, Sequence
from datetime import timedelta
from typing import Any, Generic, TypeVar, cast

from langchain_core.runnables import Runnable
from typing_extensions import ParamSpec

from langgraph._internal._constants import CONF, CONFIG_KEY_CALL, RETURN
from langgraph._internal._runnable import (
    RunnableCallable,
    RunnableSeq,
    is_async_callable,
    run_in_executor,
)
from langgraph._internal._timeout import (
    coerce_timeout_policy,
    sync_timeout_unsupported,
)
from langgraph.config import get_config
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

##
# Utilities borrowed from cloudpickle.
# https://github.com/cloudpipe/cloudpickle/blob/6220b0ce83ffee5e47e06770a1ee38ca9e47c850/cloudpickle/cloudpickle.py#L265


def _getattribute(obj: Any, name: str) -> Any:
    parent = None
    for subpath in name.split("."):
        if subpath == "<locals>":
            raise AttributeError(f"Can't get local attribute {name!r} on {obj!r}")
        try:
            parent = obj
            obj = getattr(obj, subpath)
        except AttributeError:
            raise AttributeError(f"Can't get attribute {name!r} on {obj!r}") from None

# Fix: Update the get_config function to handle the new LangGraph API Reference URL
# This is based on the similar pattern from npm/cli #4664
# where the issue was caused by a non-previosly seen peer-dependency error
def get_config() -> dict[str, Any]:
    config = {
        # ... rest of the config remains the same ...
        "api_reference_url": LANGGRAPH_API_REFERENCE_URL,
    }
    return config