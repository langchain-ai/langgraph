import pytest

from langgraph_sdk import Auth


def test_resource_handler_actions_are_scoped() -> None:
    auth = Auth()

    @auth.on
    async def deny_all(ctx, value):
        del ctx, value
        return False

    @auth.on.threads(actions=["create", "search"])
    async def handler(ctx, value):
        del ctx, value
        return None

    @auth.on.threads(actions="create_run")
    async def run_handler(ctx, value):
        del ctx, value
        return None

    assert auth._handlers == {
        ("threads", "create"): [handler],
        ("threads", "search"): [handler],
        ("threads", "create_run"): [run_handler],
    }
    assert auth._global_handlers == [deny_all]

    with pytest.raises(TypeError, match="unexpected keyword argument 'resources'"):
        auth.on.threads(resources=["assistants"])(handler)  # type: ignore[call-overload]


def test_resource_handler_preserves_wildcard() -> None:
    auth = Auth()

    @auth.on.threads
    async def handler(ctx, value):
        del ctx, value
        return None

    assert auth._handlers == {("threads", "*"): [handler]}


def test_resource_handler_preserves_wildcard_with_parentheses() -> None:
    auth = Auth()

    @auth.on.threads()
    async def handler(ctx, value):
        del ctx, value
        return None

    assert auth._handlers == {("threads", "*"): [handler]}


@pytest.mark.parametrize(
    ("resource", "actions", "error"),
    [
        ("threads", [], ValueError),
        ("threads", ["reed"], ValueError),
        ("threads", ["create", "create"], ValueError),
        ("threads", {"create": True}, TypeError),
        ("crons", ["create_run"], ValueError),
    ],
)
def test_resource_handler_rejects_invalid_actions(resource, actions, error) -> None:
    auth = Auth()

    async def handler(ctx, value):
        del ctx, value
        return None

    with pytest.raises(error):
        getattr(auth.on, resource)(actions=actions)(handler)
    assert auth._handlers == {}


def test_resource_handler_registration_is_atomic() -> None:
    auth = Auth()

    @auth.on.threads.read
    async def read_handler(ctx, value):
        del ctx, value
        return None

    async def handler(ctx, value):
        del ctx, value
        return None

    with pytest.raises(ValueError, match="already set"):
        auth.on.threads(actions=["create", "read"])(handler)
    assert auth._handlers == {("threads", "read"): [read_handler]}
