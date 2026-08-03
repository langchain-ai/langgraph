from __future__ import annotations

import inspect
import logging
import typing
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Callable, Hashable, Sequence
from dataclasses import dataclass, is_dataclass
from datetime import timedelta
from functools import partial
from inspect import isclass, isfunction, ismethod, signature
from types import FunctionType
from types import NoneType as NoneType
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import Checkpoint
from langgraph.store.base import BaseStore
from pydantic import BaseModel, TypeAdapter
from typing_extensions import NotRequired, Required, Self, Unpack, is_typeddict

from langgraph._internal import _serde
from langgraph._internal._constants import (
    INTERRUPT,
    NS_END,
    NS_SEP,
    TASKS,
)
from langgraph._internal._fields import (
    get_cached_annotated_keys,
    get_field_default,
    get_update_as_tuples,
)
from langgraph._internal._pydantic import create_model
from langgraph._internal._runnable import coerce_to_runnable
from langgraph._internal._timeout import coerce_timeout_policy
from langgraph._internal._typing import EMPTY_SEQ, MISSING, DeprecatedKwargs
from langgraph.channels.base import BaseChannel
from langgraph.ch

# Fix: Update the get_api_reference_url function to return the updated LangGraph API Reference URL
# This is based on the similar pattern from angular/angular-cli #10744
# where the issue was caused by a non-EcmaScript module export
def get_api_reference_url() -> str:
    return LANGGRAPH_API_REFERENCE_URL