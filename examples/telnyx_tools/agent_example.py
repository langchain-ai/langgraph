"""Example LangGraph agent with Telnyx communication tools.

This example demonstrates how to create a LangGraph agent that can send SMS
messages and make phone calls using the Telnyx API.

Prerequisites:
    1. Set environment variables:
        export TELNYX_API_KEY='your-api-key'
        export TELNYX_FROM_NUMBER='+15551234567'
        export TELNYX_MESSAGING_PROFILE_ID='your-profile-id'  # optional

    2. Install dependencies:
        pip install langgraph langchain-openai telnyx

Usage:
    python agent_example.py
"""

import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import Telnyx tools
from telnyx_tools import TelnyxSendSMS, TelnyxMakeCall, TelnyxHangupCall


def create_telnyx_agent(model_name: str = "gpt-4o"):
    """Create a LangGraph agent with Telnyx communication tools.

    Args:
        model_name: The OpenAI model to use for the agent.

    Returns:
        A compiled LangGraph agent with Telnyx tools.
    """
    # Initialize the model
    model = ChatOpenAI(model=model_name, temperature=0)

    # Initialize Telnyx tools
    tools = [
        TelnyxSendSMS(),
        TelnyxMakeCall(),
        TelnyxHangupCall(),
    ]

    # Create the react agent
    agent = create_react_agent(model, tools)

    return agent


def main():
    """Run an example interaction with the Telnyx agent."""
    # Verify environment variables
    if not os.environ.get("TELNYX_API_KEY"):
        print("Warning: TELNYX_API_KEY not set. Agent will return error messages.")
        print("Set it with: export TELNYX_API_KEY='your-api-key'")

    if not os.environ.get("TELNYX_FROM_NUMBER"):
        print("Warning: TELNYX_FROM_NUMBER not set. Agent will ask for sender number.")
        print("Set it with: export TELNYX_FROM_NUMBER='+15551234567'")

    # Create the agent
    agent = create_telnyx_agent()

    # Example 1: Send an SMS
    print("\n" + "=" * 60)
    print("Example 1: Sending an SMS")
    print("=" * 60)

    response = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Send an SMS to +15551234567 with the message 'Hello from LangGraph!'",
                )
            ]
        }
    )

    for message in response["messages"]:
        print(f"{message.__class__.__name__}: {message.content}")

    # Example 2: Make a call
    print("\n" + "=" * 60)
    print("Example 2: Making a phone call")
    print("=" * 60)

    response = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Call +15551234567",
                )
            ]
        }
    )

    for message in response["messages"]:
        print(f"{message.__class__.__name__}: {message.content}")


if __name__ == "__main__":
    main()
