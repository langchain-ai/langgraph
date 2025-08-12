#!/usr/bin/env python3
"""Simple test script to validate structured output functionality."""

import sys
import os
sys.path.insert(0, '/home/daytona/langgraph/libs/langgraph')

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import ToolCall
from langgraph.prebuilt import create_react_agent
from tests.test_prebuilt import FakeToolCallingModel

def test_structured_output():
    """Test that create_react_agent works correctly with structured output."""
    
    # Define the WeatherResponse model matching the documentation example
    class WeatherResponse(BaseModel):
        """Respond to the user with weather information"""
        temperature: float = Field(description="The temperature in fahrenheit")
        wind_direction: str = Field(description="The direction of the wind in abbreviated form")
        wind_speed: float = Field(description="The speed of the wind in km/h")
    
    # Create a tool call that simulates the response format tool being called
    weather_tool_call = ToolCall(
        name="WeatherResponse",
        args={
            "temperature": 75.0,
            "wind_direction": "SE", 
            "wind_speed": 4.83
        },
        id="weather_response_1",
    )
    
    # Create a fake model that will call the WeatherResponse tool
    model = FakeToolCallingModel(tool_calls=[[weather_tool_call]])
    
    # Create the agent with structured output
    agent = create_react_agent(model, [], response_format=WeatherResponse)
    
    # Test the agent
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})
    
    # Verify the result contains both messages and structured_response
    assert "messages" in result, "Result should contain 'messages' field"
    assert "structured_response" in result, "Result should contain 'structured_response' field"
    
    # Verify the structured_response is of the correct type and has correct values
    structured_response = result["structured_response"]
    assert isinstance(structured_response, WeatherResponse), f"Expected WeatherResponse, got {type(structured_response)}"
    assert structured_response.temperature == 75.0, f"Expected temperature 75.0, got {structured_response.temperature}"
    assert structured_response.wind_direction == "SE", f"Expected wind_direction 'SE', got {structured_response.wind_direction}"
    assert structured_response.wind_speed == 4.83, f"Expected wind_speed 4.83, got {structured_response.wind_speed}"
    
    # Verify the messages contain the expected structure
    messages = result["messages"]
    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"  # Human message, AI message with tool call, Tool message
    
    # Check the human message
    assert isinstance(messages[0], HumanMessage), f"Expected HumanMessage, got {type(messages[0])}"
    assert messages[0].content == "What's the weather in SF?", f"Expected specific content, got {messages[0].content}"
    
    # Check the AI message with tool call
    assert isinstance(messages[1], AIMessage), f"Expected AIMessage, got {type(messages[1])}"
    assert len(messages[1].tool_calls) == 1, f"Expected 1 tool call, got {len(messages[1].tool_calls)}"
    assert messages[1].tool_calls[0]["name"] == "WeatherResponse", f"Expected WeatherResponse tool call, got {messages[1].tool_calls[0]['name']}"
    
    # Check the tool message response
    assert isinstance(messages[2], ToolMessage), f"Expected ToolMessage, got {type(messages[2])}"
    assert messages[2].name == "WeatherResponse", f"Expected WeatherResponse name, got {messages[2].name}"
    assert messages[2].tool_call_id == "weather_response_1", f"Expected specific tool_call_id, got {messages[2].tool_call_id}"
    
    print("✅ All structured output tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_structured_output()
        print("🎉 Structured output functionality is working correctly!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
