import json
import math
from typing import Callable, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import (
    _get_message_openai_role,
)
from langchain_core.messages.utils import (
    trim_messages as trim_messages_core,
)
from langchain_core.prompts.chat import ChatPromptTemplate, ChatPromptValue
from langchain_core.runnables import RunnableConfig

from langgraph.store.base import BaseStore

SUMMARIES_NS = ("summaries",)

TokenCounter = Callable[[list[BaseMessage]], int]


DEFAULT_INITIAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
        ("user", "Create a summary of the conversation above:"),
    ]
)


DEFAULT_EXISTING_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
        (
            "user",
            "This is summary of the conversation to date: {existing_summary}\n\n"
            "Extend the summary by taking into account the new messages above:",
        ),
    ]
)

DEFAULT_FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        # if exists
        ("placeholder", "{system_message}"),
        ("system", "Summary of conversation earlier: {summary}"),
        ("placeholder", "{messages}"),
    ]
)

DEFAULT_APPROXIMATE_TOKEN_LENGTH = 4
DEFAULT_EXTRA_TOKENS_PER_MESSAGE = 3


def count_tokens_approximately(
    messages: list[BaseMessage],
    token_length: int = DEFAULT_APPROXIMATE_TOKEN_LENGTH,
    extra_tokens_per_message: int = DEFAULT_EXTRA_TOKENS_PER_MESSAGE,
    include_name: bool = True,
) -> int:
    token_count = 0
    for message in messages:
        message_chars = 0
        if isinstance(message.content, str):
            message_chars += len(message.content)

        # TODO: handle image content blocks properly
        else:
            content = json.dumps(message.content)
            message_chars += len(content)

        if (
            isinstance(message, AIMessage)
            # exclude Anthropic format as tool calls are already included in the content
            and not isinstance(message.content, list)
            and message.tool_calls
        ):
            tool_calls_content = json.dumps(message.tool_calls)
            message_chars += len(tool_calls_content)

        role = _get_message_openai_role(message)
        message_chars += len(role)

        if message.name and include_name:
            message_chars += len(message.name)

        # NOTE: we're rounding up per message to ensure that the token counts
        # are always consistent
        token_count += math.ceil(message_chars / token_length)

        # add extra tokens per message
        # see this https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        token_count += extra_tokens_per_message

    return token_count


def trim_messages(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    token_counter: TokenCounter = count_tokens_approximately,
) -> list[BaseMessage]:
    return trim_messages_core(
        messages,
        max_tokens=max_tokens,
        token_counter=token_counter,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )


def summarize_messages(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    model: BaseChatModel,
    store: BaseStore,
    config: RunnableConfig,
    max_summary_tokens: int = 256,
    token_counter: TokenCounter = count_tokens_approximately,
    initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
    final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
) -> list[BaseMessage]:
    """A memory handler that summarizes messages when they exceed a token limit and replaces summarized messages with a single summary message.

    Args:
        messages: The list of messages to process.
        max_tokens: Maximum number of tokens to return.
        model: The language model to use for generating summaries.
        store: Storage backend for persisting summaries between runs.
        config: Configuration that should contain a "configurable" key with a "thread_id" value.
        max_summary_tokens: Maximum number of tokens to return from the summarization LLM.
        token_counter: Function to count tokens in a message. Defaults to approximate counting.
        initial_summary_prompt: Prompt template for generating the first summary.
        existing_summary_prompt: Prompt template for updating an existing summary.
        final_prompt: Prompt template that combines summary with the remaining messages before returning.
    """
    if store is None:
        raise ValueError(
            "Cannot initialize SummarizationMemoryHandler with empty store. "
            "If you're using this inside a graph, it must be compiled with a store, e.g. `graph = builder.compile(store=store)`"
        )

    model = model.bind(max_tokens=max_summary_tokens)

    if max_summary_tokens >= max_tokens:
        raise ValueError("`max_summary_tokens` must be less than `max_tokens`.")

    thread_id = config.get("configurable", {}).get("thread_id")

    # it's possible for someone to need summarization in a long-running loop
    # instead of multi-turn conversation (i.e., in a single-turn conversation, without need for thread IDs / checkpointer).
    # however, there is an issue of using `store` in this case:
    # summaries will persist across invocations, which is definitely not desirable.
    # for now raising an error here, but could need a more elegant solution for this
    # (or a different one altogether)
    if not thread_id:
        raise ValueError(
            "SummarizationMemoryHandler requires a thread ID / checkpointer."
        )

    # First handle system message if present
    if messages and isinstance(messages[0], SystemMessage):
        existing_system_message = messages[0]
        # remove the system message from the list of messages to summarize
        messages = messages[1:]
        # adjust the token budget to account for the system message to be added
        max_tokens -= token_counter([existing_system_message])
    else:
        existing_system_message = None

    if not messages:
        return (
            messages
            if existing_system_message is None
            else [existing_system_message] + messages
        )

    # Check if we have a stored summary for this thread
    summary_item = store.get(SUMMARIES_NS, thread_id)
    summary_value = summary_item.value if summary_item else None
    total_summarized_messages = (
        summary_value["total_summarized_messages"] if summary_value else 0
    )

    # Single pass through messages to count tokens and find cutoff point
    n_tokens = 0
    idx = max(0, total_summarized_messages - 1)
    # we need to output messages that fit within max_tokens.
    # assuming that the summarization LLM also needs at most max_tokens
    # that will be turned into at most max_summary_tokens, you can try
    # to process at most max_tokens * 2 - max_summary_tokens
    max_total_tokens = max_tokens * 2 - max_summary_tokens
    for i in range(total_summarized_messages, len(messages)):
        n_tokens += token_counter([messages[i]])

        # If we're still under max_tokens, update the potential cutoff point
        if n_tokens <= max_tokens:
            idx = i

        # Check if we've exceeded the absolute maximum
        if n_tokens >= max_total_tokens:
            raise ValueError(
                f"SummarizationMemoryHandler cannot handle more than {max_total_tokens} tokens. "
                "Please increase the `max_tokens` or decrease the input size."
            )

    # If we haven't exceeded max_tokens, return original messages
    if n_tokens <= max_tokens:
        # we don't need to summarize, but we might still need to include the existing summary
        messages_to_summarize = None
    else:
        messages_to_summarize = messages[total_summarized_messages : idx + 1]

    # If the last message is:
    # (1) an AI message with tool calls - remove it
    #   to avoid issues w/ the LLM provider (as it will lack a corresponding tool message)
    # (2) a human message - remove it,
    #   since it is a user input and it doesn't make sense to summarize it without a corresponding AI message
    while messages_to_summarize and (
        (
            isinstance(messages_to_summarize[-1], AIMessage)
            and messages_to_summarize[-1].tool_calls
        )
        or isinstance(messages_to_summarize[-1], HumanMessage)
    ):
        messages_to_summarize.pop()

    if messages_to_summarize:
        if summary_value:
            summary_messages = cast(
                ChatPromptValue,
                existing_summary_prompt.invoke(
                    {
                        "messages": messages_to_summarize,
                        "existing_summary": summary_value["summary"],
                    }
                ),
            )
        else:
            summary_messages = cast(
                ChatPromptValue,
                initial_summary_prompt.invoke({"messages": messages_to_summarize}),
            )

        summary_message_response = model.invoke(summary_messages.messages)
        total_summarized_messages += len(messages_to_summarize)
        summary_value = {
            "summary": summary_message_response.content,
            "summarized_messages": messages_to_summarize,
            "total_summarized_messages": total_summarized_messages,
        }
        # Store the summary
        store.put(SUMMARIES_NS, thread_id, summary_value)

    if summary_value:
        updated_messages = cast(
            ChatPromptValue,
            final_prompt.invoke(
                {
                    "system_message": [existing_system_message]
                    if existing_system_message
                    else [],
                    "summary": summary_value["summary"],
                    "messages": messages[total_summarized_messages:],
                }
            ),
        )
        return updated_messages.messages
    else:
        # no changes are needed
        return (
            messages
            if existing_system_message is None
            else [existing_system_message] + messages
        )
