import pytest

from langgraph_sdk import Auth


def _handler():
    async def handler(ctx, value):
        return ctx is not None and value is not None

    return handler


def test_resource_decorator_registers_specific_actions():
    auth = Auth()

    handler = auth.on.threads(actions=["read", "search"])(_handler())

    assert auth._handlers == {
        ("threads", "read"): [handler],
        ("threads", "search"): [handler],
    }


def test_resource_decorator_registers_single_action():
    auth = Auth()

    handler = auth.on.threads(actions="read")(_handler())

    assert auth._handlers == {("threads", "read"): [handler]}


def test_resource_decorator_without_actions_registers_resource_wildcard():
    auth = Auth()

    handler = auth.on.threads(_handler())

    assert auth._handlers == {("threads", "*"): [handler]}


def test_resource_decorator_rejects_mismatched_resources():
    auth = Auth()

    with pytest.raises(ValueError, match=r"Use @auth\.on"):
        auth.on.threads(resources="assistants", actions="read")(_handler())
