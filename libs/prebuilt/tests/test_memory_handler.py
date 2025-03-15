from typing import List

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langgraph.prebuilt.memory_handler import (
    SUMMARIES_NS,
    SummarizationMemoryHandler,
    count_tokens_approximately,
)
from langgraph.store.memory import InMemoryStore


class MockChatModel:
    """Mock chat model for testing the summarizer."""

    def __init__(self, responses=None):
        """Initialize with predefined responses."""
        self.responses = responses or ["This is a mock summary."]
        self.response_index = 0
        self.invoke_calls = []

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Mock invoke method that returns predefined responses."""
        self.invoke_calls.append(messages)
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return AIMessage(content=response)

    def bind(self, **kwargs):
        """Mock bind method that returns self."""
        return self


def test_count_tokens_approximately_empty_list():
    """Test token counting with empty message list."""
    messages = []
    assert count_tokens_approximately(messages) == 0


def test_count_tokens_approximately_string_content():
    """Test token counting with string message content."""
    messages = [
        # 5 chars + 4 role chars -> 3 + 3 = 6 tokens
        HumanMessage(content="Hello"),
        # 8 chars + 9 role chars -> 5 + 3 = 8 tokens
        AIMessage(content="Hi there"),
        # 12 chars + 4 role chars -> 4 + 3 = 7 tokens
        HumanMessage(content="How are you?"),
    ]
    assert count_tokens_approximately(messages) == 21


def test_count_tokens_approximately_dict_content():
    """Test token counting with dictionary content."""
    messages = [
        # '[{"foo": "bar"}]' -> 16 chars + 4 role chars -> 5 + 3 = 8 tokens
        HumanMessage(content=[{"foo": "bar"}]),
        # '[{"test": 123}]' -> 15 chars + 9 role chars -> 6 + 3 = 9 tokens
        AIMessage(content=[{"test": 123}]),
    ]
    assert count_tokens_approximately(messages) == 17


def test_count_tokens_approximately_tool_calls():
    """Test token counting with tool calls."""
    tool_calls = [{"name": "test_tool", "args": {"foo": "bar"}, "id": "1"}]
    messages = [
        # tool calls json -> 79 chars + 9 role chars -> 22 + 3 = 25 tokens
        AIMessage(content="", tool_calls=tool_calls),
        # 15 chars + 4 role chars -> 5 + 3 = 8 tokens
        HumanMessage(content="Regular message"),
    ]
    assert count_tokens_approximately(messages) == 33


def test_count_tokens_approximately_custom_token_length():
    """Test token counting with custom token length."""
    messages = [
        # 11 chars + 4 role chars -> (4 tokens of length 4 / 8 tokens of length 2) + 3
        HumanMessage(content="Hello world"),
        # 7 chars + 9 role chars -> (4 tokens of length 4 / 8 tokens of length 2) + 3
        AIMessage(content="Testing"),
    ]
    assert count_tokens_approximately(messages, token_length=4) == 14
    assert count_tokens_approximately(messages, token_length=2) == 22


def test_not_enough_messages():
    """Test that summarizer doesn't summarize when there aren't enough messages."""
    model = MockChatModel()
    store = InMemoryStore()
    thread_id = "test_thread_id"
    config = {"configurable": {"thread_id": thread_id}}

    summarizer = SummarizationMemoryHandler(
        model=model, store=store, token_counter=len, max_tokens=4, max_summary_tokens=0
    )

    # Create a few messages (not enough to trigger summarization)
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        HumanMessage(content="How are you?"),
    ]

    # Call the summarizer
    result = summarizer.invoke(messages, config)

    # Should return the original messages unchanged
    assert result == messages
    # Model should not have been called
    assert len(model.invoke_calls) == 0


def test_summarize_first_time():
    """Test summarization when it happens for the first time."""
    store = InMemoryStore()
    thread_id = "test_thread_id"
    config = {"configurable": {"thread_id": thread_id}}

    model = MockChatModel(responses=["This is a summary of the conversation."])
    summarizer = SummarizationMemoryHandler(
        model=model, store=store, token_counter=len, max_tokens=6, max_summary_tokens=0
    )

    # Create enough messages to trigger summarization
    messages = [
        # these messages will be summarized
        HumanMessage(content="Message 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Message 2"),
        AIMessage(content="Response 2"),
        HumanMessage(content="Message 3"),
        AIMessage(content="Response 3"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Message 4"),
        AIMessage(content="Response 4"),
        HumanMessage(content="Latest message"),
    ]

    # Call the summarizer
    result = summarizer.invoke(messages, config)

    # Check that model was called
    assert len(model.invoke_calls) == 1

    # Check that the result has the expected structure:
    # - First message should be a summary
    # - Last 3 messages should be the last 3 original messages
    assert len(result) == 4
    assert result[0].type == "system"
    assert "summary" in result[0].content.lower()
    assert result[1:] == messages[-3:]

    # Check that summary was stored in the store
    summary_value = store.get(SUMMARIES_NS, thread_id).value
    assert summary_value is not None
    assert "summary" in summary_value
    assert summary_value["summary"] == "This is a summary of the conversation."
    assert (
        summary_value["summarized_messages"] == messages[:6]
    )  # All messages except the latest

    # Test subsequent invocation
    result = summarizer.invoke(messages, config)
    assert len(result) == 4
    assert result[0].type == "system"
    assert (
        result[0].content
        == "Summary of conversation earlier: This is a summary of the conversation."
    )
    assert result[1:] == messages[-3:]


def test_with_system_message():
    """Test summarization with a system message present."""
    store = InMemoryStore()
    thread_id = "test_thread_id"
    config = {"configurable": {"thread_id": thread_id}}

    model = MockChatModel(responses=["Summary with system message present."])
    summarizer = SummarizationMemoryHandler(
        model=model, store=store, token_counter=len, max_tokens=5, max_summary_tokens=0
    )

    # Create messages with a system message
    messages = [
        # this is counted towards the max_tokens
        SystemMessage(content="You are a helpful assistant."),
        # these messages will be summarized
        HumanMessage(content="Message 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Message 2"),
        AIMessage(content="Response 2"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Message 3"),
        AIMessage(content="Response 3"),
        HumanMessage(content="Message 4"),
        AIMessage(content="Response 4"),
        HumanMessage(content="Latest message"),
    ]

    # Call the summarizer
    result = summarizer.invoke(messages, config)

    # Check that model was called
    assert len(model.invoke_calls) == 1
    assert model.invoke_calls[0] == messages[1:5] + [
        HumanMessage(content="Create a summary of the conversation above:")
    ]

    # Check that the result has the expected structure:
    # - System message should be preserved
    # - Second message should be a summary of messages 2-5
    # - Last 5 messages should be the last 5 original messages
    assert len(result) == 7
    assert result[0].type == "system"
    assert result[1].type == "system"  # Summary message
    assert "summary" in result[1].content.lower()
    assert result[2:] == messages[-5:]


def test_subsequent_summarization():
    """Test that subsequent summarizations build on previous summaries."""
    store = InMemoryStore()
    thread_id = "test_thread_id"
    config = {"configurable": {"thread_id": thread_id}}

    model = MockChatModel(
        responses=[
            "First summary of the conversation.",
            "Updated summary including new messages.",
        ]
    )
    summarizer = SummarizationMemoryHandler(
        model=model, store=store, token_counter=len, max_tokens=6, max_summary_tokens=0
    )

    # First batch of messages
    messages1 = [
        HumanMessage(content="Message 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Message 2"),
        AIMessage(content="Response 2"),
        HumanMessage(content="Message 3"),
        AIMessage(content="Response 3"),
        HumanMessage(content="Latest message 1"),
    ]

    # First summarization
    result1 = summarizer.invoke(messages1, config)

    # Add more messages to trigger another summarization
    # We need to add at least max_messages (4) new messages
    messages2 = messages1 + [
        AIMessage(content="Response to latest 1"),
        HumanMessage(content="Message 4"),
        AIMessage(content="Response 4"),
        HumanMessage(content="Message 5"),
        AIMessage(content="Response 5"),
        HumanMessage(content="Message 6"),
        AIMessage(content="Response 6"),
        HumanMessage(content="Latest message 2"),
    ]

    summary_value = store.get(SUMMARIES_NS, thread_id).value
    assert summary_value["summary"] == "First summary of the conversation."

    # Second summarization
    result2 = summarizer.invoke(messages2, config)

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Check that the second call includes the previous summary
    second_call_messages = model.invoke_calls[1]
    assert any(
        "First summary" in str(msg.content)
        if hasattr(msg, "content")
        else "First summary" in str(msg)
        for msg in second_call_messages
    )

    # Check that the final result has the updated summary
    assert "summary" in result2[0].content.lower()
    assert result2[-3:] == messages2[-3:]

    # Check that the updated summary was stored
    summary_value = store.get(SUMMARIES_NS, thread_id).value
    assert summary_value["summary"] == "Updated summary including new messages."


def test_no_thread_id():
    """Test that an error is raised when no thread ID is provided."""
    model = MockChatModel()
    store = InMemoryStore()

    summarizer = SummarizationMemoryHandler(
        model=model, store=store, max_tokens=10, max_summary_tokens=0
    )
    config_without_thread_id = {"configurable": {}}

    with pytest.raises(ValueError, match="requires a thread ID"):
        summarizer.invoke([], config_without_thread_id)


def test_no_store():
    """Test that an error is raised when no store is provided."""
    model = MockChatModel()
    with pytest.raises(ValueError, match="must be compiled with a store"):
        SummarizationMemoryHandler(
            model=model, store=None, max_tokens=10, max_summary_tokens=0
        )


def test_with_empty_messages():
    """Test summarization with empty message content."""
    model = MockChatModel(responses=["Summary with empty messages."])
    store = InMemoryStore()
    thread_id = "test_thread_id"
    config = {"configurable": {"thread_id": thread_id}}

    def count_non_empty_messages(messages: list[BaseMessage]) -> int:
        return sum(1 for msg in messages if msg.content)

    summarizer = SummarizationMemoryHandler(
        model=model,
        store=store,
        token_counter=count_non_empty_messages,
        max_tokens=6,
        max_summary_tokens=0,
    )

    # Create messages with some empty content
    messages = [
        HumanMessage(content=""),
        AIMessage(content="Response 1"),
        HumanMessage(content="Message 2"),
        AIMessage(content=""),
        HumanMessage(content="Message 3"),
        AIMessage(content="Response 3"),
        HumanMessage(content="Message 4"),
        AIMessage(content="Response 4"),
        HumanMessage(content="Latest message"),
    ]

    # Call the summarizer
    result = summarizer.invoke(messages, config)

    # Check that summarization still works with empty messages
    assert len(result) == 2
    assert "summary" in result[0].content.lower()
    assert result[1:] == messages[-1:]


def test_large_number_of_messages():
    """Test summarization with a large number of messages."""
    model = MockChatModel(responses=["Summary of many messages."])
    store = InMemoryStore()
    thread_id = "test_thread_id"
    config = {"configurable": {"thread_id": thread_id}}

    summarizer = SummarizationMemoryHandler(
        model=model, store=store, token_counter=len, max_tokens=22, max_summary_tokens=0
    )

    # Create a large number of messages
    messages = []
    for i in range(20):  # 20 pairs of messages = 40 messages total
        messages.append(HumanMessage(content=f"Human message {i}"))
        messages.append(AIMessage(content=f"AI response {i}"))

    # Add one final message
    messages.append(HumanMessage(content="Final message"))

    # Call the summarizer
    result = summarizer.invoke(messages, config)

    # Check that summarization works with many messages
    assert (
        len(result) == 20
    )  # summary (for the first 22 messages) + 19 remaining original messages
    assert "summary" in result[0].content.lower()
    assert result[1:] == messages[22:]  # last 19 original messages

    # Check that the model was called with a subset of messages
    # The implementation might limit how many messages are sent to the model
    assert len(model.invoke_calls) == 1


def test_only_summarize_new_messages():
    """Test that only new messages are summarized, not all messages."""
    model = MockChatModel(
        responses=[
            "First summary of the conversation.",
            "Updated summary including only new messages.",
        ]
    )
    store = InMemoryStore()
    thread_id = "test_thread_id"
    config = {"configurable": {"thread_id": thread_id}}

    summarizer = SummarizationMemoryHandler(
        model=model, store=store, token_counter=len, max_tokens=6, max_summary_tokens=0
    )

    # First batch of messages
    messages1 = [
        # first 18 tokens will be summarized
        HumanMessage(content="Message 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Message 2"),
        AIMessage(content="Response 2"),
        HumanMessage(content="Message 3"),
        AIMessage(content="Response 3"),
        # this will be propagated to the next summarization
        HumanMessage(content="Latest message 1"),
    ]

    # First summarization
    result1 = summarizer.invoke(messages1, config)

    # Verify the first summarization happened
    assert len(model.invoke_calls) == 1

    # Check that the summary was stored correctly
    summary_value = store.get(SUMMARIES_NS, thread_id).value
    assert summary_value["summary"] == "First summary of the conversation."
    assert summary_value["total_summarized_messages"] == 6  # first 6 messages

    # Add more messages to trigger another summarization
    # We need to add at least max_messages (4) new messages
    # The result1 already contains a summary message + last 3 messages from messages1
    messages2 = messages1.copy()

    # Add enough new messages to trigger summarization (at least max_messages)
    new_messages = [
        # these will be summarized
        AIMessage(content="Response to latest 1"),
        HumanMessage(content="Message 4"),
        AIMessage(content="Response 4"),
        HumanMessage(content="Message 5"),
        AIMessage(content="Response 5"),
        # these will be propagated to the next summarization
        HumanMessage(content="Latest message 2"),
    ]

    # Add all but the last message to messages_to_summarize
    messages2.extend(new_messages)

    # Second summarization
    result2 = summarizer.invoke(messages2, config)

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Get the messages sent to the model in the second call
    second_call_messages = model.invoke_calls[1]

    # The last message in second_call_messages should be the prompt with the previous summary
    prompt_message = second_call_messages[-1]
    assert "First summary of the conversation" in prompt_message.content
    assert "Extend the summary" in prompt_message.content

    # Check that the messages sent to the model are only the new ones
    assert [msg.content for msg in second_call_messages[:-1]] == [
        "Latest message 1",
        "Response to latest 1",
        "Message 4",
        "Response 4",
        "Message 5",
        "Response 5",
    ]

    # Check that the updated summary was stored
    updated_summary_value = store.get(SUMMARIES_NS, thread_id).value
    assert (
        updated_summary_value["summary"]
        == "Updated summary including only new messages."
    )
