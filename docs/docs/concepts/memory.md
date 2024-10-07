# Memory

## What is Memory?

Memory in the context of LLMs and AI applications refers to the ability to process, store, and effectively recall information from past interactions or data sources. Memory is how your agents' get better over time.
We break this guide into two parts based on the scope of how memories are recalled: short-term memory and long-term memory.

**Short-term memory** persists **within** a single conversation (or session) with a user. In LangGraph, we manage short-term memory with the concept of [threads](persistence.md#threads) and [checkpointers](persistence.md#checkpoints). While this memory can still last forever, its scope is saved and recalled within a single thread.

**long-term memory** persists **across** conversations (or session) with a user. In LangGraph, we manage long-term memory with [stores](persistence.md#memory-store).

Both are important to understand and implement for your application.

![](img/memory/short-vs-long.png)

## Short-term memory

Short-term memory refers to the ability of an application to accurately remember previous interactions from the same conversation or session.
For a simple simple chat bot, you would persist a list of messages for each conversational turn. If a user uploads files, if the bot generates code artifacts, or if your agent causes other side-effects that should be re-used throughout the session, you would also save these objects (or references to the objects) in the graph's state. LangGraph checkpoints the state after each step of the graph. This [persistence layer](persistence.md#persistence) enables [thread](persistence.md#threads)-level memory.

As the conversation grows in length, you will need to think about how to **manage** that list of messages.
Too many messages will either (a) not fit inside an LLMs context window and will throw an error, or (b) will "distract" the LLM and cause it to perform poorly.
Therefore, it becomes crucial to think about what parts (or representations) of the conversation to pass to the LLM.

For the more general case, you will also need to think of how to represent the list of previous events within a given invocation.

We cover a few common techniques for managing message lists:

- [Editing message lists](#editing-message-lists): How to think about trimming and filtering a list of messages before passing to language model.
- [Managing messages within LangGraph](#managing-messages-within-langgraph): Some concepts that are helpful to know for managing messages within LangGraph.
- [Summarizing past conversations](#summarizing-past-conversations): A common technique to use when you don't just want to filter the list of messages.

### Editing message lists

Chat models accept instructions through [messages](https://python.langchain.com/docs/concepts/#messages), which can serve as general instructions (e.g., a system message) or user-provided instructions (e.g., human messages). In chat applications, messages often alternate between human inputs and model responses, accumulating in a list over time. Because context windows are limited and token-rich message lists can be costly, many applications can benefit from approaches to actively manage messages.    


![](img/memory/filter.png)

The most directed approach is to remove specific messages from a list. This can be done using [RemoveMessage](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/#manually-deleting-messages) based upon the message `id`, a unique identifier for each message. In the below example, we keep only the last two messages in the list using `RemoveMessage` to remove older messages based  upon their `id`.

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

Because the context window for chat model is denominated in tokens, it can be useful to trim message lists based upon some number of tokens that we want to retain. To do this, we can use [`trim_messages`](https://python.langchain.com/docs/how_to/trim_messages/#trimming-based-on-token-count) and specify number of token to keep from the list, as well as the `strategy` (e.g., keep the last `max_tokens`). 

```python
from langchain_core.messages import trim_messages
trim_messages(
    messages,
    # Keep the last <= n_count tokens of the messages.
    strategy="last",
    # Remember to adjust based on your model
    # or else pass a custom token_encoder
    token_counter=ChatOpenAI(model="gpt-4o"),
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    # Remember to adjust based on the desired conversation
    # length
    max_tokens=45,
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    start_on="human",
    # Most chat models expect that chat history ends with either:
    # (1) a HumanMessage or
    # (2) a ToolMessage
    end_on=("human", "tool"),
    # Usually, we want to keep the SystemMessage
    # if it's present in the original history.
    # The SystemMessage has special instructions for the model.
    include_system=True,
)
```
### Managing messages within LangGraph

When building agents in LangGraph, we commonly want to manage a list of messages in the graph state. Because this is such a common use case, [MessagesState](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state) is a built-in LangGraph state schema that includes a `messages` key, which is a list of messages. `MessagesState` also includes an `add_messages` reducer for updating the messages list with new messages as the application runs. The `add_messages` reducer allows us to [append](https://langchain-ai.github.io/langgraph/concepts/low_level/#serialization) new messages to the `messages` state key as shown below. When we perform a state update with `{"messages": new_message}` returned from `my_node`, the `add_messages` reducer appends `new_message` to the existing list of messages.

```python
def my_node(state: State):
    # Add a new message to the state
    new_message = HumanMessage(content="message")
    return {"messages": new_message}
```

The `add_messages` reducer built into `MessagesState` [also works with the `RemoveMessage` utility that we discussed above](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/). In this case, we can perform a state update with a list of `delete_messages` to remove specific messages from the `messages` list.

```python
def my_node(state: State):
    # Delete messages from state
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    return {"messages": delete_messages}
```

See this how-to [guide](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/) and module 2 from our [LangChain Academy](https://github.com/langchain-ai/langchain-academy/tree/main/module-2) course for example usage.

### Summarizing Past Conversations

The problem with trimming or removing messages, as shown above, is that we may lose information from culling of the message queue. Because of this, some applications benefit from a more sophisticated approach of summarizing the message history using a chat model. 

![](img/memory/summary.png)

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

## Long term memory

Long-term memory refers to the ability of a system to remember information across different conversations (or sessions).
There are many tradeoffs between long-term-memory techniques, and the right way to do so depends largely on your application's needs.
LangGraph aims to give you the low-level primitives to directly control the long-term memory of your application.
You can use LangGraph's [Store](persistence.md#memory-store) to accomplish this.

Long-term memory is far from a solved problem. While it is hard to provide generic advice, we have provided a few reliable patterns below that you should be consider when implementing long-term memory.

**Do you want to update memory "in the hot path" or "in the background"**

Memory can be updated either as part of your primary application logic (e.g. "in the hot path" of the application) or as a background task (as a separate function that generates memories based on the primary application's state).
There are pros and cons to each approach, we document them in [this section](#how-to-update-memory)

**Update own instructions**

Oftentimes, part of the system prompt (or instructions) to an LLM can be updated based on previous interactions.
This can be viewed as analyzing interactions and trying to determine what could have been done better, and then putting those learnings back into the system prompt for future interactions.
We dive into this more in [this section](#update-own-instructions)

**Learn a single profile**

This technique is useful when there is specific information you may want to remember about a user/organization/group.
You can define the schema of the profile ahead of time, and then use an LLM to update this based on interactions.
We dive into this more in [this section](#remember-a-profile)

**Learn multiple memories**
This technique is useful when you want repeatedly extract & items and remember those.
Similar to remembering a profile, you still define a schema to remember.
The difference is that rather than remembering ONE schema per user/organization/group, you remember a list.
We still use an LLM to update this list.
We dive into this more in [this section](#remember-a-list)

**Few shot examples**
Sometimes you don't need to use an LLM to update memory, but rather can just raw interactions.
You can then pull these raw interactions into the prompt as few-shot examples in future interactions.
We dive into this more in [this section](#few-shot-examples)

### How to update memory

There are two main ways to update memory: "in the hot path" and "in the background".

![](img/memory/hot_path_vs_background.png)

#### Updating memory in the hot path

This involves updating memory while the application is running. A concrete example of this is the way that ChatGPT does memory. ChatGPT can call tools to update or save a new memory. It decides when to use these tools and does so before responding to the user. 

This has a few benefits. First of all, it happens realtime, so if the user starts a new thread right away that memory will be present. The user also transparently sees when memories are stored.

This also has several downsides. It adds one more decision for the agent (what to commit to memory). This can can degrade its tool-calling performance. It may slow down the final response since it needs to decide what to commit to memory. It also typically leads to fewer things being saved to memory (since the assistant is multi-tasking), which will cause lower recall in later conversations.

#### Updating memory in the background

This involves updating memory in the background, typically as a completely separate graph or function. This can either be done as some part of background job that you write, or by using a separate memory service. Whenever a conversation completes (or on some schedule), long-term memory is "triggered" to extract and synthesize memories.

This has some benefits. Since it happens in the background, it incurs no latency. It also splits up the application logic from the memory logic, making it more modular and easy to manage.

This also has several downsides. It may not happen in real time, so users may not immediately see memory updated. You also have to think more about when to trigger this job - how do you know a conversation is finished?

## Update own instructions

This is an example of long term memory.

"Reflection" or "Meta-prompting" steps can use an LLM to generate or refine its own prompts or instructions. This approach allows the system to dynamically update and improve its own behavior, potentially leading to better performance on various tasks. This is particularly useful for tasks where the instructions are challenging to specify a priori. 

Meta-prompting can use past information to update the prompt. As an example, this [Tweet generator](https://www.youtube.com/watch?v=Vn8A3BxfplE) uses meta-prompting to iteratively improve the summarization prompt used to generate high quality paper summaries for Twitter. In this case, we used a LangSmith dataset to house several papers that we wanted to summarize, generated summaries using a naive summarization prompt, manually reviewed the summaries, captured feedback from human review using the LangSmith Annotation Queue, and passed this feedback to a chat model to re-generate the summarization prompt. The process was repeated in a loop until the summaries met our criteria in human review.

This will utilize the memory store concept above to store the updated instructions in a shared namespace. This namespace will have only a single item (unless you want to update instructions specific for each user, but that's a separate issue). This will look something like:

```python
# Node that *uses* the instructions
def call_model(state: State, store: BaseStore):
    instructions = store.search(("instructions",))[0]
    # Application logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"])
    ...


# Node that updates instructions
def update_instructions(state: State, store: BaseStore):
    namespace = ("instructions",)
    current_instructions = store.search(namespace)[0]
    # Memory logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"], conversation=state["messages"])
    output = llm.invoke(prompt)
    new_instructions = output['new_instructions']
    store.put(("instructions",), current_instructions.key, {"instructions": new_instructions})
    ...
```

![](img/memory/update-instructions.png)
### Learn a profile

The profile is generally just a JSON blob with various key-value pairs. When remembering a profile, you will want to make sure that you are **updating** the profile each time. As a result, you will want to pass in the previous profile and ask the LLM to generate a new profile (or some JSON patch to apply to the old profile).

If the profile is large, this can get tricky. You may need to split the profile into subsections and update each one individually. You also may need to fix errors if the LLM generates incorrect JSON.

![](img/memory/update-profile.png)
### Learn multiple memories

Remembering lists of information is easier in some ways, as the individual structures of each item is generally simpler and easier to generate.

It is more complex overall, as you now have to enable the LLM to *delete* or *update* existing items in the list. This can be tricky to prompt the LLM to do.

You can choose to circumvent this problem entirely by just making this list of items append only, and not allowing updates.

Another thing you will have take into account when working with lists is how to choose the relevant items to use. Right now we support filtering by metadata. We will be adding semantic search shortly.

![](img/memory/update-list.png)

## Few Shot Examples

LLMs can learn from examples. Few-shot learning lets you ["program"](https://x.com/karpathy/status/1627366413840322562) your LLM by updating the prompt with input-output examples to perform diverse tasks. While various [best-practices](https://python.langchain.com/docs/concepts/#1-generating-examples) can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input. 

LangChain [`ExampleSelectors`](https://python.langchain.com/docs/how_to/#example-selectors) can be used to customize few-shot example selection from a collection of examples using criteria such as length, semantic similarity, semantic ngram overlap, or maximal marginal relevance.

If few-shot examples are stored in a [LangSmith Dataset](https://docs.smith.langchain.com/how_to_guides/datasets), then dynamic few-shot example selectors can be used out-of-the box to achieve this same goal. LangSmith will index the dataset for you and enable retrieval of few shot examples that are most relevant to the user input based upon keyword similarity ([using a BM25-like algorithm](https://docs.smith.langchain.com/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) for keyword based similarity). 

See this how-to [video](https://www.youtube.com/watch?v=37VaU7e7t5o) for example usage of dynamic few-shot example selection in LangSmith. Also, see this [blog post](https://blog.langchain.dev/few-shot-prompting-to-improve-tool-calling-performance/) showcasing few-shot prompting to improve tool calling performance and this [blog post](https://blog.langchain.dev/aligning-llm-as-a-judge-with-human-preferences/) using few-shot example to align an LLMs to human preferences.