# Memory

## What is Memory?

Memory in the context of LLMs and AI applications refers to the ability to process, store, and effectively recall information from past interactions. With memory, your agents can learn from feedback and provide more relevant outputs to users.
This guide is divided into two sections based on the scope of memory recall: short-term memory and long-term memory.

**Short-term memory**, or thread-scoped memory, can be recalled _at any time_ **from within** a single conversational thread with a user. LangGraph manages short-term memory as a part of your agent's [state][#state]. State is persisted to a database using a [checkpointer](persistence.md#checkpoints), so the thread can be resumed at any time. Updates to short-term memory are triggered any time you invoke the graph or any time a step completes.

**Long-term memory** is shared **across** conversational threads. It can be recalled _at any time_ and in any node. Memories are scoped to any custom namespace, not just within a single thread. LangGraph provides [stores](persistence.md#memory-store) to let you save and recall long-term memories.

Both are important to understand and implement for your application.

![](img/memory/short-vs-long.png)

## Short-term memory

Short-term memory refers to the ability to accurately remember previous turns within a single thread. A thread organizes multiple turns from a single conversation or session, similar to the way an email or slack thread groups messages in a single conversation. Reading from and writing updates to the memory is scoped to within a single thread.

LangGraph agents access short-term memory using checkpointed state. Using chatbots as a common example, the state would contain conversation history as a list of chat messages. Every user request and every assistant response is appended as a message to the state. LangGraph saves the state updates in checkpoints scoped by the conversation's distinct [thread](persistence.md#threads) ID. For every subsequent user request, LangGraph can load the state from the appropriate checkpoint so your chatbot can see the entire conversation history.

If a user uploads files, if the bot generates code artifacts, or if the agent performs other side-effects, all of these objects could be stored and checkpointed as a part of your graph's state. That way, your bot can access the entire shared context for each conversation while keeping the context for each conversation separate from others.

Since conversation history is the most common form of representing short-term memory, in the next section, we will cover techniques for managing conversation history when interactions become **long**.

Long conversations pose a challenge to today's LLMs. The full history may (a) not even fit inside an LLM's context window, resulting in an irrecoverable error. Even _if_ your LLM technically supports the full context length, most LLMs (b) still perform poorly over long contexts. They get "distracted" by stale or off-topic content, all while suffering from slower response times and higher costs.

Managing short-term memory is really an exercise of balancing [precision & recall](https://en.wikipedia.org/wiki/Precision_and_recall#:~:text=Precision%20can%20be%20seen%20as,irrelevant%20ones%20are%20also%20returned) with your application's other performance requirements (latency & cost). As always, it's important to think critically about how you represent information for your LLM and to look at your data. We cover a few common techniques for managing message lists below and hope to provide sufficient context for you to pick the best tradeoffs for your application:

- [Editing message lists](#editing-message-lists): How to think about trimming and filtering a list of messages before passing to language model.
- [Managing messages within LangGraph](#managing-messages-within-langgraph): Some concepts that are helpful to know for managing messages within LangGraph.
- [Summarizing past conversations](#summarizing-past-conversations): A common technique to use when you don't just want to filter the list of messages.

### Editing message lists

Chat models accept context using [messages](https://python.langchain.com/docs/concepts/#messages). Messages transmit both developer or application-provided instructions (e.g., a system message) and user-provided instructions (e.g., human messages) all while recording the trajectory of the conversation or interaction. In chat applications, messages often alternate between human inputs and model responses, accumulating as a list over time. Because context windows are limited and token-rich message lists can be costly, many applications can benefit from using techniques to manually remove or forget stale information.

![](img/memory/filter.png)

The most direct approach is to remove old messages from a list (similar to a [least-recently used cache](https://en.wikipedia.org/wiki/Page_replacement_algorithm#Least_recently_used)).

The typical technique for deleting content from a list in LangGraph is to return an update from a node telling it to delete some portion of the list. You get to define what this update looks like, but a common approach would be to let you return an object or dictionary specifying which values to retain.

```python
def manage_list(existing: list, updates: Union[list, dict]):
    if isinstance(updates, list):
        # Normal case, add to the history
        return existing + updates
    elif isinstance(updates, dict) and updates["type"] == "keep":
        # You get to decide what this looks like.
        # For example, you could simplify and just accept a string "DELETE"
        # and clear the entire list.
        return existing[updates["from"]:updates["to"]]
    # etc. We define how to interpret updates

class State(TypedDict):
    my_list: Annotated[list, manage_list]

def my_node(state: State):
    return {
        # We return an update for the field "my_list" saying to
        # keep only values from index -5 to the end (deleting the rest)
        "my_list": {"type": "keep", "from": -5, "to": None}
    }
```

LangGraph will call the `manage_list` "[reducer][reducer]" function any time an update is returned under the key "my_list". Within that function, we define what types of updates to accept. Typically, messages will be added to the existing list (the conversation will grow); however, we've also added support to accept a dictionary that lets you "keep" certain parts of the state. This lets you programmatically drop old message context.

Another common approach is to let you return a list of "remove" objects that specify the IDs of all messages to delete. If you're using the LangChain messages and the [`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages) reducer (or `MessagesState`, which uses the same underlying functionality) in LangGraph, you can do this using a `RemoveMessage`.

```python
from langchain_core.messages import RemoveMessage, AIMessage
from langgraph.graph import add_messages
# ... other imports

class State(TypedDict):
    # add_messages will default to upserting messages by ID to the existing list
    # if a RemoveMessage is returned, it will delete the message in the list by ID
    messages: Annotated[list, add_messages]

def my_node_1(state: State):
    # Add an AI message to the `messages` list in the state
    return {"messages": [AIMessage(content="Hi")]}

def my_node_2(state: State):
    # Delete all but the last 2 messages from the `messages` list in the state
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    return {"messages": delete_messages}

```

In the example above, the `add_messages` reducer allows us to [append](https://langchain-ai.github.io/langgraph/concepts/low_level/#serialization) new messages to the `messages` state key as shown in `my_node_1`. When it sees a `RemoveMessage`, it will delete the message with that ID from the list (and the RemoveMessage will then be discarded). For more information on LangChain-specific message handling, check out [this how-to on using `RemoveMessage` ](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/).

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
            f"This is a summary of the conversation to date: {summary}\n\n"
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

### Knowing **when** to remove messages

Most LLMs have a maximum supported context window (denominated in tokens). A simple way to decide when to truncate messages is to count the tokens in the message history and truncate whenever it approaches that limit. Naive truncation is straightforward to implement on your own, though there are a few "gotchas". Some model APIs further restrict the sequence of message types (must start with human message, cannot have consecutive messages of the same type, etc.). If you're using LangChain, you can use the [`trim_messages`](https://python.langchain.com/docs/how_to/trim_messages/#trimming-based-on-token-count) utility and specify the number of tokens to keep from the list, as well as the `strategy` (e.g., keep the last `max_tokens`) to use for handling the boundary.

Below is an example.

```python
from langchain_core.messages import trim_messages
trim_messages(
    messages,
    # Keep the last <= n_count tokens of the messages.
    strategy="last",
    # Remember to adjust based on your model
    # or else pass a custom token_encoder
    token_counter=ChatOpenAI(model="gpt-4"),
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

## Long-term memory

Long-term memory refers to the ability of a system to remember information across different conversations (or sessions). While short-term memory is always scoped to a "thread", long-term memory is saved within custom scopes, or "namespaces." 

Long-term memories are saved in a [store](persistence.md#memory-store). Each memory is a JSON document stored in a custom `namespace` under a distinct `key` in that namespace. You can think of namespaces like "folders" or "directories" on your computer. They're one way of organizing information into arbitrary collections. Common things to include in a namespace would be a user or organiation ID, a schema type, or other contextual information that makes it easier to manage. To take the analogy further, the `key` would be akin to the memory's "filename", and the value would contain the contents. This permits arbitrary hierarchical organization of memories while still letting you organize and search memories using content filders to support cross-cutting searches.

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_contest)
store.put(namespace, key="a-memory", {"rules": ["User likes short, direct language", "User only speaks English & python"], "my-key": "my-value"})
# get the "memory" by ID
item = store.get(namespace)
# list "memories" within this namespace, filtering on content equivalence
items = store.search(namespace, filter={"my-key": "my-value"})
```

When adding long-term memory to your agent, it's important to think about how & how often to **write memories**, how to **stores and manage memory updates**, and how to **recall & represent memories** for the LLM in your application. These questions are all interdependent; each technique has tradeoffs, and the right way to do so depends largely on your application's needs.
LangGraph aims to give you the low-level primitives to directly control the long-term memory of your application, based on memory [Store](persistence.md#memory-store)'s.

Long-term memory is far from a solved problem. While it is hard to provide generic advice, we have provided a few reliable patterns below for your consideration as you implement long-term memory.

**Do you want to write memories "in the hot path" or "in the background"**

Memory can be updated either as part of your primary application logic (e.g. "in the hot path" of the application) or as a background task (as a separate function that generates memories based on the primary application's state). We document some tradeoffs for each appproach in [how to update memory](#how-to-update-memory)

**Do you want to manage memories as a single profile or as a list of events?**

Managing memories as a single, continuously updated "profile" or "schema" is useful when there is well-scoped, specific information you want to remember about a user, organization, or other entity (including the agent itself). You can define the schema of the profile ahead of time, and then use an LLM to update this based on interactions. Querying the "memory" is easy since it's a simple GET operation on a JSON document.We explain this in more detail in [remember a profile](#remember-a-profile). This technique can provide higher precision (on known information use cases) at the expense of lower recall (since you have to anticipate and model your domain, and updates to the doc tend to delete or rewrite away old information at a greater frequency).

Managing long-term memory as a collection of documents, on the other hand, lets you store an unbounded amount of information. This technique is useful when you want to repeatedly extract & remember items over a long time horizon but can be more complicated to query and manage over time.
Similar to the "profile" memory, you still define schema(s) for each memory. Rather than overwriting a single document, you instead will insert new ones (and potentially update or re-contextualize existing ones in the process). We explain this approach in more detail in [remember a list](#remember-a-list)

**Do you want to present memories to your agent as updated instructions or as few-shot examples?**

Memories are typically provided to the LLM as a part of the system prompt. Some common ways to "frame" memories for the LLM include providing raw information as "memories from previous interactions with user A", as system instructions or rules, or as few-shot examples.

Framing memories as "learning rules or instructions" typically means dedicating a portion of the system prompt to instructions the LLM can manage itself. After each conversation, you can prompt the LLM to evaluate its performance and update the instructions to better handle this type of task in the future. We explain this approach in more detail in [this section](#update-own-instructions).

Storing memories as few-shot examples lets you store and manage instructions as cause and effect. Each memory stores an input or context and expected response. Including a reasoning trajectory (a chain-of-thought) can also help provide sufficient context so that the memory is less likely to be mis-used in the future. We elaborate on this concept more in [this section](#few-shot-examples)

We will expand on techniques for writing, managing, and recalling & formatting memories in the following section.

### Writing memories

Humans form long-term memories when we sleep, but when and how should our agents create new memories? The two most common ways we see agents write memories are "in the hot path" and "in the background".

![](img/memory/hot_path_vs_background.png)

#### Writing memories in the hot path

This involves creating memories while the application is running. To provide a popular production example, ChatGPT manages memories uses a "save_memories" tool to upsert memories as content strings. It decides whether (and how) to use this tool every time it receives a user message and multi-tasks memory management with the rest of the user instructions.

This has a few benefits. First of all, it happens "in real time". If the user starts a new thread right away that memory will be present. The user also transparently sees when memories are stored, since the bot has to explicitly decide to store information and can relate that to the user.

This also has several downsides. It complicates the decisions the agent must make (what to commit to memory). This complication can degrade its tool-calling performance and reduce task completion rates. It will slow down the final response since it needs to decide what to commit to memory. It also typically leads to fewer things being saved to memory (since the assistant is multi-tasking), which will cause **lower recall** in later conversations.

#### Writing memories in the background

This involves updating memory as a conceptually separate task, typically as a completely separate graph or function. Since it happens in the background, it incurs no latency. It also splits up the application logic from the memory logic, making it more modular and easy to manage. It also lets you separate the timing of memory creation, letting you avoid redundant work. Your agent can focus on accomplishing its immediate task without having to consciously think about what it needs to remember.

This approach is not without its downsides, however. You have to think about how often to write memories. If it doesn't run in realtime, the user's interactions on other threads won't benefit from the new context. You also have to think about when to trigger this job. We typically recommend scheduling memories after some point of time, cancelling and re-scheduling for the future if new events occur on a given thread. Other popular choices are to form memories on some cron schedule or to let the user or application logic manually trigger memory formation.

### Managing memories

Once you've sorted out memory scheduling, it's important to think about **how to update memory with new information**.

There are two ends of the spectrum: on one hand, you could continuously update a single document (memory profile). On the other, you could only ever insert new documents every time you receive new information.

We will outline some tradeoffs between these two approaches below, understanding that most people will find it most appropriate to combine approaches and to settle on somewhere in the middle.

#### Manage individual profiles

A profile is generally just a JSON document with various key-value pairs you've selected to represent your domain. When remembering a profile, you will want to make sure that you are **updating** the profile each time. As a result, you will want to pass in the previous profile and ask the LLM to generate a new profile (or some JSON patch to apply to the old profile).

The larger the document, the more error-prone this can become. If your document becomes **too** large, you may want to consider splitting up the profiles into separate sections. You will likely need to use generation with retries and/or **strict** decoding when generating documents to ensure the memory schemas remains valid.

![](img/memory/update-profile.png)

#### Manage a collection of memories

Saving memories as an collection documents simplifies some things. Each individual memory can be more narrowly scoped and easier to generate. It also means you're less likely to **lose** information over time, since it's easier for an LLM to generate _new_ objects for new information than it is for it to reconcile that new information with information in a dense profile. This tends to lead to higher recall downstream.

This approach shifts some complexity to how you prompt the LLM to apply memory updates. You now have to enable the LLM to _delete_ or _update_ existing items in the list. This can be tricky to prompt the LLM to do. Some LLMs may default to over-inserting; others may default to over-updating. Tuning the behavior here is best done through evals, somethign you can do with a tool like [LangSmith](https://docs.smith.langchain.com/tutorials/Developers/evaluation).

This also shifts complexity to memory **search** (recall). You have to think about what relevant items to use. Right now we support filtering by metadata. We will be adding semantic search shortly.

Finally, this shifts some complexity to how you represent the memories for the LLM (and by extension, the schemas you use to save each memories). It's very easy to write memories that can easily be mistaken out-of-context. It's important to prompt the LLM to include all necessary contextual information in the given memory so that when you use it in later conversations it doesn't mistakenly mis-apply that information.

![](img/memory/update-list.png)

### Representing memories

Once you have saved memories, the way you then retrieve and present the memory content for the LLM can play a large role in how well your LLM incorporates that information in its responses.
The following sections a couple of common approaches. Note that these sections also will largely inform how you write and manage memories. Everything in memory is connected!

#### Update own instructions

While instructions are often static text written by the developer, many AI applications benefit from letting the users personalize the rules and instructions the agent should follow whenever it interacts with that user. This ideally can be inferred by its interactions with the user (so the user doesn't have to explicitly change settings in yoru app). In this sense, instructions are a form of long-form memory!

One way to apply this is using "reflection" or "Meta-prompting" steps. Prompt the LLM with the current instructions set (from the system prompt) and a conversation with the user and instruct the LLM to to refine its instructions. This approach allows the system to dynamically update and improve its own behavior, potentially leading to better performance on various tasks. This is particularly useful for tasks where the instructions are challenging to specify a priori.

Meta-prompting uses past information to refine prompts. For instance, a [Tweet generator](https://www.youtube.com/watch?v=Vn8A3BxfplE) employs meta-prompting to enhance its paper summarization prompt for Twitter. You could implement this using LangGraph's memory store to save updated instructions in a shared namespace. In this case we will namespace the memoreis

```python
# Node that *uses* the instructions
def call_model(state: State, store: BaseStore):
    namespace = ("agent_instructions", )
    instructions = store.get(namespace, key="agent_a")[0]
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
    store.put(("agent_instructions",), "agent_a", {"instructions": new_instructions})
    ...
```

![](img/memory/update-instructions.png)

#### Few Shot Examples

Sometimes it's easier to "show" than "tell." LLMs learn well from examples. Few-shot learning lets you ["program"](https://x.com/karpathy/status/1627366413840322562) your LLM by updating the prompt with input-output examples to illustrate the intended behavior. While various [best-practices](https://python.langchain.com/docs/concepts/#1-generating-examples) can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input.

Note that the memory store is just one way to store data as few-shot examples. If you want to have more developer involvement, or tie few-shots more closely to your evaluation harness, you can also use a [LangSmith Dataset](https://docs.smith.langchain.com/how_to_guides/datasets) to store your data. Then dynamic few-shot example selectors can be used out-of-the box to achieve this same goal. LangSmith will index the dataset for you and enable retrieval of few shot examples that are most relevant to the user input based upon keyword similarity ([using a BM25-like algorithm](https://docs.smith.langchain.com/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) for keyword based similarity). See this how-to [video](https://www.youtube.com/watch?v=37VaU7e7t5o) for example usage of dynamic few-shot example selection in LangSmith. Also, see this [blog post](https://blog.langchain.dev/few-shot-prompting-to-improve-tool-calling-performance/) showcasing few-shot prompting to improve tool calling performance and this [blog post](https://blog.langchain.dev/aligning-llm-as-a-judge-with-human-preferences/) using few-shot example to align an LLMs to human preferences.

## Fin

Thank you for reading! This doc just scratches the surface of how to think about memory in LangGraph. Please comment below with more challenges you've run into using memory in production.
