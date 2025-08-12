#!/usr/bin/env python3
"""
Simple test script to verify structured output functionality works correctly.
This script tests the implementation without requiring pytest or poetry.
"""

import sys
import os
import json
from typing import Literal
from pydantic import BaseModel, Field

# Add the langgraph library to the path
sys.path.insert(0, '/home/daytona/langgraph/libs/langgraph')

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
    from langchain_core.tools import tool as dec_tool
    from langgraph.prebuilt import create_react_agent
    from tests.test_prebuilt import FakeToolCallingModel
    
    print("✓ Successfully imported required modules")
    
    # Define the WeatherResponse Pydantic model
    class WeatherResponse(BaseModel):
        temperature: float = Field(description="The temperature in fahrenheit")
        wind_direction: str = Field(description="The direction of the wind in abbreviated form")
        wind_speed: float = Field(description="The speed of the wind in mph")
    
    print("✓ Successfully defined WeatherResponse model")
    
    # Create a get_weather tool
    @dec_tool
    def get_weather(city: Literal["nyc", "sf"]) -> str:
        """Use this to get weather information."""
        if city == "nyc":
            return "It is cloudy in NYC, with 5 mph winds in the North-East direction and a temperature of 70 degrees"
        elif city == "sf":
            return "It is 75 degrees and sunny in SF, with 3 mph winds in the South-East direction"
        else:
            raise AssertionError("Unknown city")
    
    print("✓ Successfully created get_weather tool")
    
    # Create a model that will first call the tool, then return structured JSON
    model = FakeToolCallingModel(
        tool_calls=[
            # First call: call the get_weather tool
            [ToolCall(name="get_weather", args={"city": "nyc"}, id="call_1")],
            # Second call: return structured response (no tool calls)
            []
        ]
    )
    
    # Override the _generate method to return structured JSON content on the final call
    original_generate = model._generate
    def custom_generate(messages, stop=None, run_manager=None, **kwargs):
        result = original_generate(messages, stop, run_manager, **kwargs)
        # On the final call (index 1), return structured JSON content
        if model.index == 2:  # After incrementing in original_generate
            result.generations[0].message.content = '{"temperature": 70.0, "wind_direction": "NE", "wind_speed": 5.0}'
        return result
    
    model._generate = custom_generate
    
    print("✓ Successfully configured test model")
    
    # Create agent with structured response format
    agent = create_react_agent(model, [get_weather], response_format=WeatherResponse)
    
    print("✓ Successfully created agent with structured response format")
    
    # Run the agent
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in NYC?")]})
    
    print("✓ Successfully invoked agent")
    
    # Verify the result contains both messages and structured_response
    assert "messages" in result, "Result should contain 'messages' field"
    assert "structured_response" in result, "Result should contain 'structured_response' field"
    
    print("✓ Result contains both messages and structured_response fields")
    
    # Verify the structured_response is a WeatherResponse instance with correct data
    structured_response = result["structured_response"]
    assert isinstance(structured_response, WeatherResponse), f"Expected WeatherResponse, got {type(structured_response)}"
    assert structured_response.temperature == 70.0, f"Expected temperature 70.0, got {structured_response.temperature}"
    assert structured_response.wind_direction == "NE", f"Expected wind_direction 'NE', got {structured_response.wind_direction}"
    assert structured_response.wind_speed == 5.0, f"Expected wind_speed 5.0, got {structured_response.wind_speed}"
    
    print("✓ Structured response contains correct data:")
    print(f"  - Temperature: {structured_response.temperature}°F")
    print(f"  - Wind Direction: {structured_response.wind_direction}")
    print(f"  - Wind Speed: {structured_response.wind_speed} mph")
    
    # Verify we have the expected messages
    messages = result["messages"]
    assert len(messages) >= 4, f"Expected at least 4 messages, got {len(messages)}"
    assert isinstance(messages[0], HumanMessage), f"First message should be HumanMessage, got {type(messages[0])}"
    assert isinstance(messages[-1], AIMessage), f"Last message should be AIMessage, got {type(messages[-1])}"
    assert messages[-1].content == '{"temperature": 70.0, "wind_direction": "NE", "wind_speed": 5.0}', f"Last message content incorrect: {messages[-1].content}"
    
    print("✓ Message flow is correct")
    print(f"  - Total messages: {len(messages)}")
    print(f"  - First message type: {type(messages[0]).__name__}")
    print(f"  - Last message type: {type(messages[-1]).__name__}")
    
    print("\n🎉 ALL TESTS PASSED! Structured output functionality is working correctly.")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
