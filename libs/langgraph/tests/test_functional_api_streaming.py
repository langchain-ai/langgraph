"""
Test case for GitHub Issue #6373: Functional API stream_mode='messages' fix

This test verifies that token-level streaming works correctly in the Functional API
when using stream_mode='messages'. Previously, callbacks from inside @task were not
propagating correctly, causing only a single complete message to be returned instead
of streaming tokens progressively.

The fix ensures that StreamMessagesHandler is properly merged with task callbacks
instead of being skipped when call.callbacks is truthy.
"""

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langgraph.func import entrypoint, task
from tests.fake_chat import FakeChatModel


def test_functional_api_stream_mode_messages() -> None:
    """Test that stream_mode='messages' works with @task decorator.
    
    This tests the fix for GitHub Issue #6373 where the Functional API
    did not stream responses token-by-token. The issue was that
    StreamMessagesHandler was being skipped due to incorrect callback
    merging logic in prepare_push_task_functional().
    """
    # Create a fake chat model that streams tokens
    model = FakeChatModel(
        messages=[AIMessage(content="Hello world this is a streaming test response")]
    )
    
    @task
    def call_llm(messages: list[BaseMessage]) -> AIMessage:
        """LLM call inside a @task decorator."""
        return model.invoke(messages)
    
    @entrypoint()
    def functional_agent(messages: list[BaseMessage]) -> list[BaseMessage]:
        """Functional API agent that uses @task for LLM calls."""
        response = call_llm(messages).result()
        return [response]
    
    messages = [HumanMessage(content="Hi there!")]
    
    # Stream with stream_mode="messages" and subgraphs=True
    # subgraphs=True is required to receive messages from nested tasks
    collected_chunks = []
    for ns, (msg, metadata) in functional_agent.stream(
        messages, 
        stream_mode="messages",
        subgraphs=True
    ):
        if hasattr(msg, 'content') and msg.content:
            collected_chunks.append(msg.content)
    
    # With the fix, we should get multiple chunks (token-level streaming)
    # Without the fix, we would only get 1 chunk (complete message)
    assert len(collected_chunks) > 1, (
        f"Expected multiple streaming chunks, but got {len(collected_chunks)}. "
        "This indicates StreamMessagesHandler is not propagating to @task callbacks."
    )
    
    # Verify the complete response is reconstructed from chunks
    full_response = "".join(collected_chunks)
    assert "Hello" in full_response
    assert "streaming" in full_response


def test_functional_api_stream_mode_messages_with_user_callbacks() -> None:
    """Test that user-provided callbacks don't break message streaming.
    
    This specifically tests the scenario where users pass their own callbacks
    via config. Previously, if call.callbacks was truthy (even an empty 
    CallbackManager), the StreamMessagesHandler from the manager was skipped.
    """
    from langchain_core.callbacks import BaseCallbackHandler
    
    # Create a fake chat model that streams tokens
    model = FakeChatModel(
        messages=[AIMessage(content="Token by token streaming test")]
    )
    
    # Track if user callback receives events
    user_callback_events = []
    
    class UserCallback(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            user_callback_events.append(token)
    
    @task
    def call_llm(messages: list[BaseMessage]) -> AIMessage:
        return model.invoke(messages)
    
    @entrypoint()
    def functional_agent(messages: list[BaseMessage]) -> list[BaseMessage]:
        response = call_llm(messages).result()
        return [response]
    
    messages = [HumanMessage(content="Test")]
    
    # Stream with user-provided callback
    collected_chunks = []
    for ns, (msg, metadata) in functional_agent.stream(
        messages, 
        stream_mode="messages",
        subgraphs=True,
        config={"callbacks": [UserCallback()]}
    ):
        if hasattr(msg, 'content') and msg.content:
            collected_chunks.append(msg.content)
    
    # Both user callbacks and StreamMessagesHandler should work
    assert len(collected_chunks) > 1, "Message streaming failed with user callbacks"
    assert len(user_callback_events) > 0, "User callback did not receive any events"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
