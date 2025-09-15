import pytest
from langgraph.warnings import LangGraphDeprecatedSinceV10
from typing_extensions import TypedDict

from langgraph.prebuilt import create_react_agent
from tests.model import FakeToolCallingModel


class Config(TypedDict):
    model: str


@pytest.mark.filterwarnings("ignore:`config_schema` is deprecated")
@pytest.mark.filterwarnings("ignore:`get_config_jsonschema` is deprecated")
def test_config_schema_deprecation() -> None:
    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
    ):
        agent = create_react_agent(FakeToolCallingModel(), [], config_schema=Config)
        assert agent.context_schema == Config

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`config_schema` is deprecated. Use `get_context_jsonschema` for the relevant schema instead.",
    ):
        assert agent.config_schema() is not None

    with pytest.warns(
        LangGraphDeprecatedSinceV10,
        match="`get_config_jsonschema` is deprecated. Use `get_context_jsonschema` instead.",
    ):
        assert agent.get_config_jsonschema() is not None


def test_extra_kwargs_deprecation() -> None:
    with pytest.raises(
        TypeError,
        match="create_react_agent\(\) got unexpected keyword arguments: \{'extra': 'extra'\}",
    ):
        create_react_agent(FakeToolCallingModel(), [], extra="extra")
