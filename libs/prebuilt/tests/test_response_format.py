"""Test suite for create_react_agent with structured output response_format permutations."""

from dataclasses import dataclass
from typing import TypedDict, Union

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langgraph.prebuilt import create_agent
from langgraph.prebuilt.responses import NativeOutput, ToolOutput
from tests.model import FakeToolCallingModel


# Test data models
class WeatherBaseModel(BaseModel):
    temperature: float = Field(description="The temperature in fahrenheit")
    condition: str = Field(description="Weather condition")


@dataclass
class WeatherDataclass:
    temperature: float
    condition: str


class WeatherTypedDict(TypedDict):
    temperature: float
    condition: str


weather_json_schema = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number", "description": "Temperature in fahrenheit"},
        "condition": {"type": "string", "description": "Weather condition"},
    },
    "required": ["temperature", "condition"],
}


class LocationResponse(BaseModel):
    city: str = Field(description="The city name")
    country: str = Field(description="The country name")


def get_weather() -> str:
    """Get the weather."""

    return "The weather is sunny and 75°F."


def get_location() -> str:
    """Get the current location."""

    return "You are in New York, USA."


class TestResponseFormatAsModel:
    def test_pydantic_model(self) -> None:
        """Test response_format as Pydantic model."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = WeatherBaseModel(temperature=75.0, condition="sunny")
        model = FakeToolCallingModel[WeatherBaseModel](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(model, [get_weather], response_format=WeatherBaseModel)
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_dataclass(self) -> None:
        """Test response_format as dataclass."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherDataclass",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = WeatherDataclass(temperature=75.0, condition="sunny")
        model = FakeToolCallingModel[WeatherDataclass](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(model, [get_weather], response_format=WeatherDataclass)
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_typed_dict(self) -> None:
        """Test response_format as TypedDict."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherTypedDict",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = {"temperature": 75.0, "condition": "sunny"}
        model = FakeToolCallingModel[dict](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(model, [get_weather], response_format=WeatherTypedDict)
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_json_schema(self) -> None:
        """Test response_format as JSON schema."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "weather_schema",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = {"temperature": 75.0, "condition": "sunny"}
        model = FakeToolCallingModel[dict](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(model, [get_weather], response_format=weather_json_schema)
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5


class TestResponseFormatAsToolOutput:
    def test_pydantic_model(self) -> None:
        """Test response_format as ToolOutput with Pydantic model."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = WeatherBaseModel(temperature=75.0, condition="sunny")
        model = FakeToolCallingModel[WeatherBaseModel](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model, [get_weather], response_format=ToolOutput(WeatherBaseModel)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_dataclass(self) -> None:
        """Test response_format as ToolOutput with dataclass."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherDataclass",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = WeatherDataclass(temperature=75.0, condition="sunny")
        model = FakeToolCallingModel[WeatherDataclass](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model, [get_weather], response_format=ToolOutput(WeatherDataclass)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_typed_dict(self) -> None:
        """Test response_format as ToolOutput with TypedDict."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherTypedDict",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = {"temperature": 75.0, "condition": "sunny"}
        model = FakeToolCallingModel[dict](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model, [get_weather], response_format=ToolOutput(WeatherTypedDict)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_json_schema(self) -> None:
        """Test response_format as ToolOutput with JSON schema."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "weather_schema",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = {"temperature": 75.0, "condition": "sunny"}
        model = FakeToolCallingModel[dict](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model, [get_weather], response_format=ToolOutput(weather_json_schema)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_union_of_types(self) -> None:
        """Test response_format as ToolOutput with Union of various types."""
        # Test with WeatherBaseModel
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = WeatherBaseModel(temperature=75.0, condition="sunny")
        model = FakeToolCallingModel[Union[WeatherBaseModel, LocationResponse]](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model,
            [get_weather, get_location],
            response_format=ToolOutput(Union[WeatherBaseModel, LocationResponse]),
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

        # Test with LocationResponse
        tool_calls_location = [
            [{"args": {}, "id": "1", "name": "get_location"}],
            [
                {
                    "name": "LocationResponse",
                    "id": "2",
                    "args": {"city": "New York", "country": "USA"},
                }
            ],
        ]

        expected_location_response = LocationResponse(city="New York", country="USA")
        model_location = FakeToolCallingModel[
            Union[WeatherBaseModel, LocationResponse]
        ](
            tool_calls=tool_calls_location,
            structured_response=expected_location_response,
        )

        agent_location = create_agent(
            model_location,
            [get_weather, get_location],
            response_format=ToolOutput(Union[WeatherBaseModel, LocationResponse]),
        )
        response_location = agent_location.invoke(
            {"messages": [HumanMessage("Where am I?")]}
        )

        assert response_location["structured_response"] == expected_location_response
        assert len(response_location["messages"]) == 5


# 3. response_format as a native output
class TestResponseFormatAsNativeOutput:
    def test_pydantic_model(self) -> None:
        """Test response_format as NativeOutput with Pydantic model."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherBaseModel",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = WeatherBaseModel(temperature=75.0, condition="sunny")
        model = FakeToolCallingModel[WeatherBaseModel](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model, [get_weather], response_format=NativeOutput(WeatherBaseModel)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_dataclass(self) -> None:
        """Test response_format as NativeOutput with dataclass."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherDataclass",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = WeatherDataclass(temperature=75.0, condition="sunny")
        model = FakeToolCallingModel[WeatherDataclass](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model, [get_weather], response_format=NativeOutput(WeatherDataclass)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_typed_dict(self) -> None:
        """Test response_format as NativeOutput with TypedDict."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "WeatherTypedDict",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = {"temperature": 75.0, "condition": "sunny"}
        model = FakeToolCallingModel[dict](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model, [get_weather], response_format=NativeOutput(WeatherTypedDict)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5

    def test_json_schema(self) -> None:
        """Test response_format as NativeOutput with JSON schema."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [
                {
                    "name": "weather_schema",
                    "id": "2",
                    "args": {"temperature": 75.0, "condition": "sunny"},
                }
            ],
        ]

        expected_response = {"temperature": 75.0, "condition": "sunny"}
        model = FakeToolCallingModel[dict](
            tool_calls=tool_calls, structured_response=expected_response
        )

        agent = create_agent(
            model, [get_weather], response_format=NativeOutput(weather_json_schema)
        )
        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == expected_response
        assert len(response["messages"]) == 5


def test_union_of_types(self) -> None:
    """Test response_format as NativeOutput with Union (if supported)."""
    tool_calls = [
        [{"args": {}, "id": "1", "name": "get_weather"}],
        [
            {
                "name": "WeatherBaseModel",
                "id": "2",
                "args": {"temperature": 75.0, "condition": "sunny"},
            }
        ],
    ]

    expected_response = WeatherBaseModel(temperature=75.0, condition="sunny")
    model = FakeToolCallingModel[Union[WeatherBaseModel, LocationResponse]](
        tool_calls=tool_calls, structured_response=expected_response
    )

    agent = create_agent(
        model,
        [get_weather, get_location],
        response_format=NativeOutput(Union[WeatherBaseModel, LocationResponse]),
    )
    response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

    assert response["structured_response"] == expected_response
    assert len(response["messages"]) == 5
