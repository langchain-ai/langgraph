"""Redefined messages as a work-around for pydantic issue with AnyStr.

The code below creates version of pydantic models
that will work in unit tests with AnyStr as id field
Please note that the `id` field is assigned AFTER the model is created
to workaround an issue with pydantic ignoring the __eq__ method on
subclassed strings.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from tests.any import AnyStr


def _AnyIdAIMessage(**kwargs: Any) -> AIMessage:
    """Create ai message with an any id field."""
    message = AIMessage(**kwargs)
    message.id = AnyStr()
    return message


def _AnyIdHumanMessage(**kwargs: Any) -> HumanMessage:
    """Create a human message with an any id field."""
    message = HumanMessage(**kwargs)
    message.id = AnyStr()
    return message
