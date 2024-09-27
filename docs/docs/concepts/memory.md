# Memory

## What is Memory?

Memory refers to the processing of data from the past to make it more useful for an application. Examples include:

- Managing what messages (e.g., from a long message history) are sent to a chat model to limit token usage;
- Summarizing past conversations to give a chat model context from prior interaction;
- Selecting few shot examples (e.g, from a dataset) to guide model responses;
- Personalizing applications with specific information (e.g., user attributes) gleaned from previous conversations;
- Allowing an LLM to update its own prompt using past information (e.g., meta-prompting);

Below, we'll discuss each of these examples in some detail. 

## Managing Messages

Chat models accept instructions through [messages](https://python.langchain.com/docs/concepts/#messages), which can serve as general instructions (e.g., a system message) or specific user-provided information (e.g., human messages). In chat applications, messages often alternate between human input and model responses, accumulating in a list over time. Because context windows are limited and token-rich message lists are costly, many applications can benefit from approaches to actively manage messages.    

One approach is simply to remove messages (e.g., based upon recency) from a message list. To do this based upon a particular number of tokens, we can use [`trim_messages`](https://python.langchain.com/v0.2/docs/how_to/trim_messages/#getting-the-last-max_tokens-tokens). As an example, we can use this utility to keep the last `max_tokens` from the total list of Messages 

```python
from langchain_core.messages import trim_messages
trim_messages(
            messages,
            max_tokens=100,
            strategy="last", 
            token_counter=ChatOpenAI(model="gpt-4o"),
        )
```

In some cases, we want to remove specific messages from a list. This can be done using [RemoveMessage](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/#manually-deleting-messages) based upon `id`, a unique identifier for each message. In the below example, we keep only the last two messages in the list but we can also select specific messages to remove based upon their `id`.

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

When building agents in LangGraph, we commonly want to manage message in state. [MessagesState](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state) is a built-in LangGraph state schema that includes a `messages` key, which is a list of messages, and an `add_messages` reducer for updating the messages list with new messages as the application runs. The `add_messages` reducer allows us to [append](https://langchain-ai.github.io/langgraph/concepts/low_level/#serialization) new messages to `messages` state key as shown here as a state update in a graph node.

```
{"messages": [HumanMessage(content="message")]}
```

Additionally, the `add_messages` reducer [works with the `RemoveMessage` utility to remove messages from the `messages` state key](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/). Using `messages` and `delete_messages` defined above, we can remove the first two messages from the list as shown below. We pass the list of `delete_messages` (`[RemoveMessage(content='', id='1'), RemoveMessage(content='', id='2')]`) to the `add_messages` reducer.

```python
from langgraph.graph.message import add_messages
add_messages(messages , delete_messages)
```

See this how-to [guide](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/) and module 2 from our [LangChain Academy](https://github.com/langchain-ai/langchain-academy/tree/main/module-2) course for example usage.

## Summarizing Past Conversations

The problem with trimming or removing messages, as shown above, is that we may loose information from culling of the message queue. Because of this, some applications benefit from a more sophisticated approach of summarizing the message history using a chat model. 

Fairly simple prompting and orchestration logic can be used to achieve this. As an example, in LangGraph we can extend the [MessagesState](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state) to include a `summary` key. 

```python
from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str
```

Then, we can simply generate a summary of the chat history, using any existing summary as context for the next summary. This summarize_conversation node can be called after some number of messages have accumulated in the `messages` state key.

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

## Selecting Few Shot Examples

Few-shot learning is a powerful technique where LLMs can be ["programmed"](https://x.com/karpathy/status/1627366413840322562) inside the prompt with input-output examples to perform diverse tasks. While various [best-practices](https://python.langchain.com/docs/concepts/#1-generating-examples) can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input. 

LangChain [`ExampleSelectors`](https://python.langchain.com/docs/how_to/#example-selectors) can be used to customize few-shot example selection from a collection of examples using criteria such as length, semantic similarity, semantic ngram overlap, or maximal marginal relevance.

If few-shot examples are stored in a [LangSmith Dataset](https://docs.smith.langchain.com/how_to_guides/datasets), then dynamic few-shot example selectors can be used out-of-the box to achieve this same goal. LangSmith will index the dataset for you and enable retrieval of few shot examples that are most relevant to the user input based upon keyword similarity ([using a BM25-like algorithm](https://docs.smith.langchain.com/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) for keyword based similarity scores). 

See this how-to [video](https://www.youtube.com/watch?v=37VaU7e7t5o) example usage of dynamic few-shot example selection in LangSmith.

## Personalizing Applications 

LangGraph's [persistence layer](https://langchain-ai.github.io/langgraph/concepts/persistence/#persistence) has checkpointers that utilize various storage systems, including an in-memory key-value store or different databases. These checkpoints write the graph state at each execution step into a thread, which can be accessed at a later time using a thread ID to resume a previous graph execution. 

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

While persistance is critical for long-running user sessions that may have interruptions, some applications benefit from sharing specific information across *many* graph executions. 

< TODO: Here we can show an example of a personalization use-case using new memory API. >

## Meta-prompting

Meta-prompting is an advanced technique where an AI system, typically a large language model (LLM), is used to generate or refine its own prompts or instructions. This approach allows the system to dynamically update and improve its own behavior, potentially leading to better performance on various tasks. A central consideration in meta-prompting is the source of past information used to improve the prompt. 

One source of past information for prompt improvement is human feedback. As an example, this [how-to guide](https://www.youtube.com/watch?v=Vn8A3BxfplE) shows how to improve a prompt for summarization using human feedback. This particular uses a LangSmith dataset to house the test cases and captures human feedback to grade summaries using the LangSmith Annotation Queue. The process is repeated in a loop until the prompts perform well based upon the human review of the summaries. 

< TODO: Add update examples of meta-prompting with LangSmith. >
