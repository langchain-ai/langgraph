# Telnyx Tools for LangGraph

This example demonstrates how to integrate Telnyx telephony capabilities into LangGraph agents using LangChain-compatible tools.

## Overview

Telnyx is a global communications platform providing SMS, voice, and other telephony services. This integration allows LangGraph agents to:

- Send SMS text messages
- Make voice phone calls
- Hang up active calls

## Prerequisites

1. **Telnyx Account**: Sign up at [telnyx.com](https://telnyx.com)
2. **API Key**: Get your API key from the [Telnyx Portal](https://portal.telnyx.com)
3. **Phone Number**: Purchase or port a phone number in the Telnyx Portal
4. **Messaging Profile** (optional): Create a messaging profile for SMS

## Installation

```bash
pip install langgraph langchain-openai telnyx
```

## Configuration

Set the following environment variables:

```bash
export TELNYX_API_KEY='your-api-key'
export TELNYX_FROM_NUMBER='+15551234567'  # Your Telnyx phone number
export TELNYX_MESSAGING_PROFILE_ID='your-profile-id'  # Optional, for SMS
```

## Usage

### Basic Usage with ToolNode

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, create_react_agent
from telnyx_tools import TelnyxSendSMS, TelnyxMakeCall

# Initialize tools
tools = [
    TelnyxSendSMS(),
    TelnyxMakeCall(),
]

# Create agent
model = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(model, tools)

# Send an SMS
response = agent.invoke({
    "messages": [("user", "Send SMS to +15551234567 saying 'Hello!'")]
})
```

### Custom Configuration

You can also configure tools with explicit credentials instead of environment variables:

```python
from telnyx_tools import TelnyxSendSMS

sms_tool = TelnyxSendSMS(
    api_key="your-api-key",
    from_number="+15551234567",
    messaging_profile_id="your-profile-id"
)
```

## Available Tools

### TelnyxSendSMS

Send SMS text messages to any phone number.

**Parameters:**
- `to` (required): Destination phone number in E.164 format (e.g., `+15551234567`)
- `body` (required): Text message content
- `from_` (optional): Sender phone number (uses `TELNYX_FROM_NUMBER` if not provided)

### TelnyxMakeCall

Initiate a voice phone call using Telnyx Call Control.

**Parameters:**
- `to` (required): Destination phone number in E.164 format
- `from_` (optional): Caller phone number (uses `TELNYX_FROM_NUMBER` if not provided)
- `webhook_url` (optional): URL for receiving call status events

### TelnyxHangupCall

Hang up an active voice call.

**Parameters:**
- `call_control_id` (required): The call control ID from a previous call

## Example Agent

See `agent_example.py` for a complete working example:

```bash
python agent_example.py
```

## Phone Number Format

All phone numbers must be in E.164 format:
- Start with `+`
- Include country code
- No spaces or dashes
- Example: `+15551234567`

## Error Handling

All tools return descriptive error messages for:
- Missing or invalid API keys
- Missing phone numbers
- Invalid phone number formats
- API communication errors

## Telnyx Documentation

- [Telnyx API Reference](https://developers.telnyx.com/docs/api/v2/overview)
- [Telnyx Python SDK](https://github.com/team-telnyx/telnyx-python)
- [Call Control Documentation](https://developers.telnyx.com/docs/api/v2/call-control)

## License

This example is provided under the same license as LangGraph.
