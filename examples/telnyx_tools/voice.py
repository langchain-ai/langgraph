"""Telnyx Voice tool for LangGraph agents.

This module provides LangChain-compatible tools for making and managing
voice calls via the Telnyx API.
"""

from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class TelnyxCallInput(BaseModel):
    """Input schema for Telnyx voice call tool."""

    to: str = Field(
        ...,
        description="The destination phone number in E.164 format (e.g., '+15551234567')",
    )
    from_: str = Field(
        default="",
        description="The source phone number in E.164 format. If not provided, uses TELNYX_FROM_NUMBER environment variable.",
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Optional webhook URL for call status events",
    )


class TelnyxHangupInput(BaseModel):
    """Input schema for Telnyx hangup tool."""

    call_control_id: str = Field(
        ...,
        description="The call control ID of the call to hang up",
    )


class TelnyxMakeCall(BaseTool):
    """Tool for making voice calls via Telnyx API.

    This tool allows LangGraph agents to initiate phone calls using the
    Telnyx Voice API with Call Control.

    Setup:
        1. Install the telnyx package: `pip install telnyx`
        2. Set your Telnyx API key: `export TELNYX_API_KEY='your-api-key'`
        3. Set your Telnyx phone number: `export TELNYX_FROM_NUMBER='+15551234567'`
        4. Configure your Telnyx Call Control application in the dashboard

    Example:
        ```python
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from telnyx_tools import TelnyxMakeCall

        model = ChatOpenAI(model="gpt-4")
        tools = [TelnyxMakeCall()]
        agent = create_react_agent(model, tools)

        # Agent can now make phone calls
        response = agent.invoke(
            {"messages": [("user", "Call +15551234567 and play a greeting")]}
        )
        ```

    Attributes:
        name: The name of the tool for agent use.
        description: Description of what the tool does.
    """

    name: str = "telnyx_make_call"
    description: str = (
        "Make a phone call to a specified number using Telnyx. "
        "Use this when you need to call someone. "
        "The 'to' parameter should be a phone number in E.164 format (e.g., '+15551234567'). "
        "Returns a call_control_id that can be used to manage the call."
    )
    args_schema: type[BaseModel] = TelnyxCallInput

    # Configuration attributes
    api_key: Optional[str] = Field(default=None, exclude=True)
    from_number: Optional[str] = Field(default=None, exclude=True)

    def __init__(
        self,
        api_key: Optional[str] = None,
        from_number: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Telnyx voice call tool.

        Args:
            api_key: Telnyx API key. If not provided, reads from TELNYX_API_KEY env var.
            from_number: Default caller phone number in E.164 format. If not provided,
                reads from TELNYX_FROM_NUMBER env var.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.from_number = from_number

    def _run(
        self,
        to: str,
        from_: str = "",
        webhook_url: Optional[str] = None,
        run_manager=None,
    ) -> str:
        """Make a voice call.

        Args:
            to: Destination phone number in E.164 format.
            from_: Source phone number (optional, uses config if not provided).
            webhook_url: Optional webhook URL for call status events.

        Returns:
            A success message with the call control ID or an error message.
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

        # Determine the caller number
        caller = from_ or self.from_number or os.environ.get("TELNYX_FROM_NUMBER")
        if not caller:
            return (
                "Error: No caller phone number provided. "
                "Either pass the 'from_' parameter or set TELNYX_FROM_NUMBER environment variable."
            )

        try:
            # Create the call
            call_params = {
                "from_": caller,
                "to": to,
            }

            if webhook_url:
                call_params["webhook_url"] = webhook_url

            call = telnyx.Call.create(**call_params)

            return (
                f"Successfully initiated call to {to}. "
                f"Call Control ID: {call.call_control_id}. "
                f"Call will be connected shortly."
            )

        except telnyx.error.AuthenticationError:
            return "Error: Invalid Telnyx API key. Please check your TELNYX_API_KEY."
        except telnyx.error.InvalidRequestError as e:
            return f"Error: Invalid request - {str(e)}"
        except Exception as e:
            return f"Error making call: {str(e)}"

    async def _arun(
        self,
        to: str,
        from_: str = "",
        webhook_url: Optional[str] = None,
        run_manager=None,
    ) -> str:
        """Async implementation - calls the sync version."""
        return self._run(to, from_, webhook_url, run_manager)


class TelnyxHangupCall(BaseTool):
    """Tool for hanging up voice calls via Telnyx API.

    This tool allows LangGraph agents to hang up active phone calls
    using the Telnyx Call Control API.

    Example:
        ```python
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from telnyx_tools import TelnyxMakeCall, TelnyxHangupCall

        model = ChatOpenAI(model="gpt-4")
        tools = [TelnyxMakeCall(), TelnyxHangupCall()]
        agent = create_react_agent(model, tools)
        ```

    Attributes:
        name: The name of the tool for agent use.
        description: Description of what the tool does.
    """

    name: str = "telnyx_hangup_call"
    description: str = (
        "Hang up an active phone call. "
        "Use this when you need to end a call. "
        "Provide the call_control_id returned from a previous telnyx_make_call."
    )
    args_schema: type[BaseModel] = TelnyxHangupInput

    # Configuration attributes
    api_key: Optional[str] = Field(default=None, exclude=True)

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Telnyx hangup tool.

        Args:
            api_key: Telnyx API key. If not provided, reads from TELNYX_API_KEY env var.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self.api_key = api_key

    def _run(
        self,
        call_control_id: str,
        run_manager=None,
    ) -> str:
        """Hang up an active call.

        Args:
            call_control_id: The call control ID of the call to hang up.

        Returns:
            A success message or an error message.
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

        try:
            # Retrieve and hang up the call
            call = telnyx.Call.retrieve(call_control_id)
            call.hangup()

            return f"Successfully hung up call {call_control_id}."

        except telnyx.error.AuthenticationError:
            return "Error: Invalid Telnyx API key. Please check your TELNYX_API_KEY."
        except telnyx.error.InvalidRequestError as e:
            return f"Error: Invalid request - {str(e)}"
        except Exception as e:
            return f"Error hanging up call: {str(e)}"

    async def _arun(
        self,
        call_control_id: str,
        run_manager=None,
    ) -> str:
        """Async implementation - calls the sync version."""
        return self._run(call_control_id, run_manager)
