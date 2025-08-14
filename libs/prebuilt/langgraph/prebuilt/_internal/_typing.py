from __future__ import annotations

from typing import Awaitable, Callable, TypeVar, Union

from langgraph._internal._typing import StateLike

try:
    from typing import ParamSpec  # 3.10+
except ImportError:
    from typing_extensions import ParamSpec  # type: ignore

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

MaybeAwaitable = Union[T, Awaitable[T]]
SyncOrAsync = Callable[P, MaybeAwaitable[R]]

# PreConfiguredChatModel is used to support chat models that have beeen pre-configured
# using .bind().
# For example, chat_model.bind(api_key="...") will return a PreConfiguredChatModel
PreConfiguredChatModel = Runnable[LanguageModelInput, BaseMessage]
ContextT = TypeVar("ContextT", bound=StateLike)
