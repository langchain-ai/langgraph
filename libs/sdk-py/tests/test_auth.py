import pytest

from langgraph_sdk import Auth


def test_resource_handler_actions_are_scoped() -> None:
    auth = Auth()

    @auth.on.threads(actions=["create", "search"])
    async def handler(ctx, value):
        del ctx, value
        return None

    assert auth._handlers == {
        ("threads", "create"): [handler],
        ("threads", "search"): [handler],
    }

    with pytest.raises(TypeError, match="unexpected keyword argument 'resources'"):
        auth.on.threads(resources=["assistants"])(handler)  # type: ignore[call-overload]
