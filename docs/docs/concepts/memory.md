# Memory

## What is Memory?

Memory in the context of LLMs and AI applications refers to the ability to process, retain, and utilize information from past interactions or data sources. Examples include:

- Managing what messages (e.g., from a long message history) are sent to a chat model to limit token usage
- Summarizing past conversations to give a chat model context from prior interactions
- Selecting few shot examples (e.g., from a dataset) to guide model responses
- Maintaining persistent data (e.g., user preferences) across multiple chat sessions
- Allowing an LLM to update its own prompt using past information (e.g., meta-prompting)

Below, we'll discuss each of these examples in some detail. 

## Managing Messages

Chat models accept instructions through [messages](https://python.langchain.com/docs/concepts/#messages), which can serve as general instructions (e.g., a system message) or user-provided instructions (e.g., human messages). In chat applications, messages often alternate between human inputs and model responses, accumulating in a list over time. Because context windows are limited and token-rich message lists can be costly, many applications can benefit from approaches to actively manage messages.    

One approach is simply to remove messages (e.g., based upon some criteria such as recency) from a message list. To do this based upon a particular number of tokens, we can use [`trim_messages`](https://python.langchain.com/v0.2/docs/how_to/trim_messages/#getting-the-last-max_tokens-tokens). As an example, the `trim_messages` utility can keep the last `max_tokens` from a list of Messages. 

```python
from langchain_core.messages import trim_messages
trim_messages(
            messages,
            max_tokens=100,
            strategy="last", 
            token_counter=ChatOpenAI(model="gpt-4o"),
        )
```

In some cases, we want to remove specific messages from a list. This can be done using [RemoveMessage](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/#manually-deleting-messages) based upon `id`, a unique identifier for each message. In the below example, we keep only the last two messages in the list using `RemoveMessage`.

```python
from langchain_core.messages import RemoveMessage

# Message list
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Isolate messages to delete
delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
print(delete_messages)
[RemoveMessage(content='', id='1'), RemoveMessage(content='', id='2')]
```

When building agents in LangGraph, we commonly want to manage messages in state. [MessagesState](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state) is a built-in LangGraph state schema that includes a `messages` key, which is a list of messages, and an `add_messages` reducer for updating the messages list with new messages as the application runs. The `add_messages` reducer allows us to [append](https://langchain-ai.github.io/langgraph/concepts/low_level/#serialization) new messages to the `messages` state key. This is how we would achieve this as a state update in a graph node.

```
{"messages": [HumanMessage(content="message")]}
```

Additionally, the `add_messages` reducer [works with the `RemoveMessage` utility to remove selected messages from the `messages` state key](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/). Using `messages` and `delete_messages` as defined above, we can remove the first two messages from the list; we simply pass the list of `delete_messages` (`[RemoveMessage(content='', id='1'), RemoveMessage(content='', id='2')]`) to the `add_messages` reducer.

```python
from langgraph.graph.message import add_messages
add_messages(messages , delete_messages)
```

See this how-to [guide](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/) and module 2 from our [LangChain Academy](https://github.com/langchain-ai/langchain-academy/tree/main/module-2) course for example usage.

## Summarizing Past Conversations

The problem with trimming or removing messages, as shown above, is that we may loose information from culling of the message queue. Because of this, some applications benefit from a more sophisticated approach of summarizing the message history using a chat model. 

Simple prompting and orchestration logic can be used to achieve this. As an example, in LangGraph we can extend the [MessagesState](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state) to include a `summary` key. 

```python
from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str
```

Then, we can generate a summary of the chat history, using any existing summary as context for the next summary. This `summarize_conversation` node can be called after some number of messages have accumulated in the `messages` state key.

```python   
def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
```

See this how-to [here](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/) and module 2 from our [LangChain Academy](https://github.com/langchain-ai/langchain-academy/tree/main/module-2) course for example usage.

## Few Shot Examples

Few-shot learning is a powerful technique where LLMs can be ["programmed"](https://x.com/karpathy/status/1627366413840322562) inside the prompt with input-output examples to perform diverse tasks. While various [best-practices](https://python.langchain.com/docs/concepts/#1-generating-examples) can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input. 

LangChain [`ExampleSelectors`](https://python.langchain.com/docs/how_to/#example-selectors) can be used to customize few-shot example selection from a collection of examples using criteria such as length, semantic similarity, semantic ngram overlap, or maximal marginal relevance.

If few-shot examples are stored in a [LangSmith Dataset](https://docs.smith.langchain.com/how_to_guides/datasets), then dynamic few-shot example selectors can be used out-of-the box to achieve this same goal. LangSmith will index the dataset for you and enable retrieval of few shot examples that are most relevant to the user input based upon keyword similarity ([using a BM25-like algorithm](https://docs.smith.langchain.com/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) for keyword based similarity). 

See this how-to [video](https://www.youtube.com/watch?v=37VaU7e7t5o) for example usage of dynamic few-shot example selection in LangSmith. Also, see this [blog post](https://blog.langchain.dev/few-shot-prompting-to-improve-tool-calling-performance/) showcasing few-shot prompting to improve tool calling performance and this [blog post](https://blog.langchain.dev/aligning-llm-as-a-judge-with-human-preferences/) using few-shot example to align an LLMs to human preferences.

## Maintaining Data Across Chat Sessions

LangGraph's [persistence layer](https://langchain-ai.github.io/langgraph/concepts/persistence/#persistence) has checkpointers that utilize various storage systems, including an in-memory key-value store or different databases. These checkpoints capture the graph state at each execution step and accumulate in a thread, which can be accessed at a later time using a thread ID to resume a previous graph execution. We add persistence to our graph by passing a checkpointer to the `compile` method, as shown here.

```python
# Compile the graph with a checkpointer
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# Invoke the graph with a thread ID
config = {"configurable": {"thread_id": "1"}}
graph.invoke(input_state, config)

# get the latest state snapshot at a later time
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)
```

Persistence is critical sustaining a long-running chat sessions. For example, a chat between a user and an AI assistant may have interruptions. Persistence ensures that a user can continue that particular chat session at any later point in time. However, what happens if a user initiates a new chat session with an assistant? This spawns a new thread, and the information from the previous session (thread) is not retained. 

But, sometimes we want to maintain data across chat sessions (threads). For example, certain user preferences are relevant across all chat sessions with that particular user. The LangGraph memory API enables this, allowing specific keys in a state schema to be accessible across all threads. For example, we can define a state schema with a `user_preferences` key that is associated with a `user_id`. Any thread can access the value of the `user_preferences` key as long as the `user_id` is supplied.

```python
class State(MessagesState):
    user_preferences: Annotated[dict, SharedValue.on("user_id")]
```

For a specific example, we built a writing assistant using the memory API that learns and remembers the user's writing preferences. It writes content (e.g., blog posts or tweets), allows a user to make edits, reflects on the edits, and uses reflection to update a set of writing style heuristics. These heuristics are saved by the memory API to a particular key in the assistant's state schema and are accessible across all user sessions with the assistant.

See this video for more context on the memory API and this video for an overview of the writing assistant.

## Meta-prompting

Meta-prompting uses an LLM to generate or refine its own prompts or instructions. This approach allows the system to dynamically update and improve its own behavior, potentially leading to better performance on various tasks. This is particularly useful for tasks where the instructions are challenging to specify a priori. 

As with the writing assistant discussed above, human feedback is often a useful component of meta-prompting. With the writing assistant, feedback was used to create a set of rules that the memory API made accessible across all sessions with the user. With meta-prompting, human feedback is used slightly differently: it is used to re-write a task-specific prompt.

As an example, we created this [tweet generator](https://www.youtube.com/watch?v=Vn8A3BxfplE) that used meta-prompting to iteratively improve the tweet generation prompt. In this case, we used a LangSmith dataset to house several summarization test cases, captured human feedback on these test cases using the LangSmith Annotation Queue, and used the feedback to refine the summarization prompt directly. The process was repeated in a loop until the summaries met our criteria in human review.