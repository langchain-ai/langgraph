#!/usr/bin/env python3
"""Simple test script to validate structured output functionality."""

import sys
import os
sys.path.insert(0, '/home/daytona/langgraph/libs/langgraph')

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import ToolCall
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, List, Any
from langgraph.prebuilt import create_react_agent

class SimpleFakeModel(BaseChatModel):
    """Simple fake model for testing structured output."""
    
    def __init__(self, tool_calls_to_make=None, **kwargs):
        super().__init__(**kwargs)
        self._tool_calls_to_make = tool_calls_to_make or []
        self._call_count = 0
    
    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response with tool calls."""
        content = f"Response {self._call_count}"
        tool_calls = self._tool_calls_to_make if self._call_count == 0 else []
        
        message = AIMessage(
            content=content,
            tool_calls=tool_calls
        )
        self._call_count += 1
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    @property
    def _llm_type(self) -> str:
        return "simple-fake-model"
    
    def bind_tools(self, tools, **kwargs):
        """Bind tools to the model - required for structured output functionality."""
        # Return self to maintain the same interface
        return self

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
    model = SimpleFakeModel(tool_calls_to_make=[weather_tool_call])
    
    # Create the agent with structured output
    agent = create_react_agent(model, [], response_format=WeatherResponse)
    
    # Test the agent
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})
    
    # Debug: Print the actual result to see what we get
    print(f"Actual result keys: {list(result.keys())}")
    print(f"Actual result: {result}")
    
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
    assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}"
    
    # Check the human message
    assert isinstance(messages[0], HumanMessage), f"Expected HumanMessage, got {type(messages[0])}"
    assert messages[0].content == "What's the weather in SF?", f"Expected specific content, got {messages[0].content}"
    
    # Find the AI message with tool call
    ai_message = None
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            ai_message = msg
            break
    
    assert ai_message is not None, "Expected to find an AI message with tool calls"
    assert len(ai_message.tool_calls) == 1, f"Expected 1 tool call, got {len(ai_message.tool_calls)}"
    assert ai_message.tool_calls[0]["name"] == "WeatherResponse", f"Expected WeatherResponse tool call, got {ai_message.tool_calls[0]['name']}"
    
    # Find the tool message response
    tool_message = None
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "WeatherResponse":
            tool_message = msg
            break
    
    assert tool_message is not None, "Expected to find a ToolMessage with name WeatherResponse"
    assert tool_message.tool_call_id == "weather_response_1", f"Expected specific tool_call_id, got {tool_message.tool_call_id}"
    
    print("✅ All structured output tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_structured_output()
        print("🎉 Structured output functionality is working correctly!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




