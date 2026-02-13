"""Telnyx SMS tool for LangGraph agents.

This module provides a LangChain-compatible tool for sending SMS messages
via the Telnyx API.
"""

from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class TelnyxSMSInput(BaseModel):
    """Input schema for Telnyx SMS tool."""

    to: str = Field(
        ...,
        description="The destination phone number in E.164 format (e.g., '+15551234567')",
    )
    body: str = Field(
        ...,
        description="The text message content to send",
    )
    from_: str = Field(
        default="",
        description="The source phone number in E.164 format. If not provided, uses TELNYX_FROM_NUMBER environment variable.",
    )


class TelnyxSendSMS(BaseTool):
    """Tool for sending SMS messages via Telnyx API.

    This tool allows LangGraph agents to send text messages to phone numbers
    using the Telnyx telecommunications platform.

    Setup:
        1. Install the telnyx package: `pip install telnyx`
        2. Set your Telnyx API key: `export TELNYX_API_KEY='your-api-key'`
        3. Set your Telnyx phone number: `export TELNYX_FROM_NUMBER='+15551234567'`
        4. Optionally set a messaging profile ID: `export TELNYX_MESSAGING_PROFILE_ID='your-profile-id'`

    Example:
        ```python
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from telnyx_tools import TelnyxSendSMS

        model = ChatOpenAI(model="gpt-4")
        tools = [TelnyxSendSMS()]
        agent = create_react_agent(model, tools)

        # Agent can now send SMS messages
        response = agent.invoke(
            {"messages": [("user", "Send an SMS to +15551234567 saying 'Hello from LangGraph!'")]}
        )
        ```

    Attributes:
        name: The name of the tool for agent use.
        description: Description of what the tool does.
    """

    name: str = "telnyx_send_sms"
    description: str = (
        "Send an SMS text message to a phone number. "
        "Use this when you need to send a text message to someone. "
        "The 'to' parameter should be a phone number in E.164 format (e.g., '+15551234567'). "
        "The 'body' parameter is the text content of the message."
    )
    args_schema: type[BaseModel] = TelnyxSMSInput

    # Configuration attributes
    api_key: Optional[str] = Field(default=None, exclude=True)
    from_number: Optional[str] = Field(default=None, exclude=True)
    messaging_profile_id: Optional[str] = Field(default=None, exclude=True)

    def __init__(
        self,
        api_key: Optional[str] = None,
        from_number: Optional[str] = None,
        messaging_profile_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Telnyx SMS tool.

        Args:
            api_key: Telnyx API key. If not provided, reads from TELNYX_API_KEY env var.
            from_number: Default sender phone number in E.164 format. If not provided,
                reads from TELNYX_FROM_NUMBER env var.
            messaging_profile_id: Optional messaging profile ID. If not provided,
                reads from TELNYX_MESSAGING_PROFILE_ID env var.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.from_number = from_number
        self.messaging_profile_id = messaging_profile_id

    def _run(
        self,
        to: str,
        body: str,
        from_: str = "",
        run_manager=None,
    ) -> str:
        """Send an SMS message.

        Args:
            to: Destination phone number in E.164 format.
            body: Text message content.
            from_: Source phone number (optional, uses config if not provided).

        Returns:
            A success message with the message ID or an error message.
        """
        import os

        try:
            import telnyx
        except ImportError:
            return (
                "Error: telnyx package not installed. "
                "Please install it with: pip install telnyx"
            )

        # Get API key from config or environment
        api_key = self.api_key or os.environ.get("TELNYX_API_KEY")
        if not api_key:
            return "Error: TELNYX_API_KEY not set. Please set the TELNYX_API_KEY environment variable."

        telnyx.api_key = api_key

        # Determine the sender number
        sender = from_ or self.from_number or os.environ.get("TELNYX_FROM_NUMBER")
        if not sender:
            return (
                "Error: No sender phone number provided. "
                "Either pass the 'from_' parameter or set TELNYX_FROM_NUMBER environment variable."
            )

        # Get messaging profile ID if available
        profile_id = (
            self.messaging_profile_id
            or os.environ.get("TELNYX_MESSAGING_PROFILE_ID")
            or None
        )

        try:
            # Build message parameters
            message_params = {
                "from_": sender,
                "to": to,
                "text": body,
            }

            if profile_id:
                message_params["messaging_profile_id"] = profile_id

            # Send the message
            message = telnyx.Message.create(**message_params)

            return (
                f"Successfully sent SMS to {to}. "
                f"Message ID: {message.id}. "
                f"Status: {message.status}."
            )

        except telnyx.error.AuthenticationError:
            return "Error: Invalid Telnyx API key. Please check your TELNYX_API_KEY."
        except telnyx.error.InvalidRequestError as e:
            return f"Error: Invalid request - {str(e)}"
        except Exception as e:
            return f"Error sending SMS: {str(e)}"

    async def _arun(
        self,
        to: str,
        body: str,
        from_: str = "",
        run_manager=None,
    ) -> str:
        """Async implementation - calls the sync version.

        Note: Telnyx SDK is sync-only, so this wraps the sync call.
        For production use with async LangGraph, consider using
        asyncio.to_thread() or running in an executor.
        """
        return self._run(to, body, from_, run_manager)
