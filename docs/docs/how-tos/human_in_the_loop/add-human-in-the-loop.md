---
search:
  boost: 2
tags:
  - human-in-the-loop
  - hil
  - interrupt
hide:
  - tags
---

# Add human-in-the-loop

## `interrupt`

The [`interrupt` function][langgraph.types.interrupt] in LangGraph enables human-in-the-loop workflows by pausing the graph at a specific node, presenting information to a human, and resuming the graph with their input. It's useful for tasks like approvals, edits, or gathering additional context.

The graph is resumed using a [`Command`](../reference/types.md#langgraph.types.Command) object that provides the human's response.

```python
# highlight-next-line
from langgraph.types import interrupt, Command

def human_node(state: State):
   # highlight-next-line
   value = interrupt( # (1)!
      {
         "text_to_revise": state["some_text"] # (2)!
      }
   )
   return {
      "some_text": value # (3)!
   }


graph = graph_builder.compile(checkpointer=checkpointer) # (4)!

# Run the graph until the interrupt is hit.
config = {"configurable": {"thread_id": "some_id"}}
result = graph.invoke({"some_text": "original text"}, config=config) # (5)!
print(result['__interrupt__']) # (6)!
# > [
# >    Interrupt(
# >       value={'text_to_revise': 'original text'}, 
# >       resumable=True,
# >       ns=['human_node:6ce9e64f-edef-fe5d-f7dc-511fa9526960']
# >    )
# > ] 

# highlight-next-line
print(graph.invoke(Command(resume="Edited text"), config=config)) # (7)!
# > {'some_text': 'Edited text'}
```

1. `interrupt(...)` pauses execution at `human_node`, surfacing the given payload to a human.
2. Any JSON serializable value can be passed to the `interrupt` function. Here, a dict containing the text to revise.
3. Once resumed, the return value of `interrupt(...)` is the human-provided input, which is used to update the state.
4. A checkpointer is required to persist graph state. In production, this should be durable (e.g., backed by a database).
5. The graph is invoked with some initial state.
6. When the graph hits the interrupt, it returns an `Interrupt` object with the payload and metadata.
7. The graph is resumed with a `Command(resume=...)`, injecting the human's input and continuing execution.

??? example "Extended example: using `interrupt`"

      ```python
      from typing import TypedDict
      import uuid

      from langgraph.checkpoint.memory import InMemorySaver
      from langgraph.constants import START
      from langgraph.graph import StateGraph
      # highlight-next-line
      from langgraph.types import interrupt, Command

      class State(TypedDict):
         some_text: str

      def human_node(state: State):
         # highlight-next-line
         value = interrupt( # (1)!
            {
               "text_to_revise": state["some_text"] # (2)!
            }
         )
         return {
            "some_text": value # (3)!
         }


      # Build the graph
      graph_builder = StateGraph(State)
      graph_builder.add_node("human_node", human_node)
      graph_builder.add_edge(START, "human_node")

      checkpointer = InMemorySaver() # (4)!

      graph = graph_builder.compile(checkpointer=checkpointer)

      # Pass a thread ID to the graph to run it.
      config = {"configurable": {"thread_id": uuid.uuid4()}}

      # Run the graph until the interrupt is hit.
      result = graph.invoke({"some_text": "original text"}, config=config) # (5)!

      print(result['__interrupt__']) # (6)!
      # > [
      # >    Interrupt(
      # >       value={'text_to_revise': 'original text'}, 
      # >       resumable=True,
      # >       ns=['human_node:6ce9e64f-edef-fe5d-f7dc-511fa9526960']
      # >    )
      # > ] 

      # highlight-next-line
      print(graph.invoke(Command(resume="Edited text"), config=config)) # (7)!
      # > {'some_text': 'Edited text'}
      ```

      1. `interrupt(...)` pauses execution at `human_node`, surfacing the given payload to a human.
      2. Any JSON serializable value can be passed to the `interrupt` function. Here, a dict containing the text to revise.
      3. Once resumed, the return value of `interrupt(...)` is the human-provided input, which is used to update the state.
      4. A checkpointer is required to persist graph state. In production, this should be durable (e.g., backed by a database).
      5. The graph is invoked with some initial state.
      6. When the graph hits the interrupt, it returns an `Interrupt` object with the payload and metadata.
      7. The graph is resumed with a `Command(resume=...)`, injecting the human's input and continuing execution.



!!! tip "New in 0.4.0"

      `__interrupt__` is a special key that will be returned when running the graph if the graph is interrupted. Support for `__interrupt__` in `invoke` and `ainvoke` has been added in version 0.4.0. If you're on an older version, you will only see `__interrupt__` in the result if you use `stream` or `astream`. You can also use `graph.get_state(thread_id)` to get the interrupt value.

!!! warning

      Interrupts are both powerful and ergonomic. However, while they may resemble Python's input() function in terms of developer experience, it's important to note that they do not automatically resume execution from the interruption point. Instead, they rerun the entire node where the interrupt was used.
      For this reason, interrupts are typically best placed at the start of a node or in a dedicated node. Please read the [resuming from an interrupt](#how-does-resuming-from-an-interrupt-work) section for more details.

## Requirements

To use `interrupt` in your graph, you need to:

1. [**Specify a checkpointer**](../../concepts/persistence.md#checkpoints) to save the graph state after each step.
2. **Call `interrupt()`** in the appropriate place. See the [Design Patterns](#design-patterns) section for examples.
3. **Run the graph** with a [**thread ID**](../../concepts/persistence.md#threads) until the `interrupt` is hit.
4. **Resume execution** using `invoke`/`ainvoke`/`stream`/`astream` (see [**The `Command` primitive**](#resume-using-the-command-primitive)).

## Design patterns

There are typically three different **actions** that you can do with a human-in-the-loop workflow:

1. **Approve or Reject**: Pause the graph before a critical step, such as an API call, to review and approve the action. If the action is rejected, you can prevent the graph from executing the step, and potentially take an alternative action. This pattern often involve **routing** the graph based on the human's input.
2. **Edit Graph State**: Pause the graph to review and edit the graph state. This is useful for correcting mistakes or updating the state with additional information. This pattern often involves **updating** the state with the human's input.
3. **Get Input**: Explicitly request human input at a particular step in the graph. This is useful for collecting additional information or context to inform the agent's decision-making process.

Below we show different design patterns that can be implemented using these **actions**.

### Approve or reject

<figure markdown="1">
![image](../../concepts/img/human_in_the_loop/approve-or-reject.png){: style="max-height:400px"}
<figcaption>Depending on the human's approval or rejection, the graph can proceed with the action or take an alternative path.</figcaption>
</figure>

Pause the graph before a critical step, such as an API call, to review and approve the action. If the action is rejected, you can prevent the graph from executing the step, and potentially take an alternative action.

```python
from typing import Literal
from langgraph.types import interrupt, Command

def human_approval(state: State) -> Command[Literal["some_node", "another_node"]]:
    is_approved = interrupt(
        {
            "question": "Is this correct?",
            # Surface the output that should be
            # reviewed and approved by the human.
            "llm_output": state["llm_output"]
        }
    )

    if is_approved:
        return Command(goto="some_node")
    else:
        return Command(goto="another_node")

# Add the node to the graph in an appropriate location
# and connect it to the relevant nodes.
graph_builder.add_node("human_approval", human_approval)
graph = graph_builder.compile(checkpointer=checkpointer)

# After running the graph and hitting the interrupt, the graph will pause.
# Resume it with either an approval or rejection.
thread_config = {"configurable": {"thread_id": "some_id"}}
graph.invoke(Command(resume=True), config=thread_config)
```

??? example "Extended example: approve or reject with interrupt"

    ```python
    from typing import Literal, TypedDict
    import uuid

    from langgraph.constants import START, END
    from langgraph.graph import StateGraph
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.memory import MemorySaver

    # Define the shared graph state
    class State(TypedDict):
        llm_output: str
        decision: str

    # Simulate an LLM output node
    def generate_llm_output(state: State) -> State:
        return {"llm_output": "This is the generated output."}

    # Human approval node
    def human_approval(state: State) -> Command[Literal["approved_path", "rejected_path"]]:
        decision = interrupt({
            "question": "Do you approve the following output?",
            "llm_output": state["llm_output"]
        })

        if decision == "approve":
            return Command(goto="approved_path", update={"decision": "approved"})
        else:
            return Command(goto="rejected_path", update={"decision": "rejected"})

    # Next steps after approval
    def approved_node(state: State) -> State:
        print("✅ Approved path taken.")
        return state

    # Alternative path after rejection
    def rejected_node(state: State) -> State:
        print("❌ Rejected path taken.")
        return state

    # Build the graph
    builder = StateGraph(State)
    builder.add_node("generate_llm_output", generate_llm_output)
    builder.add_node("human_approval", human_approval)
    builder.add_node("approved_path", approved_node)
    builder.add_node("rejected_path", rejected_node)

    builder.set_entry_point("generate_llm_output")
    builder.add_edge("generate_llm_output", "human_approval")
    builder.add_edge("approved_path", END)
    builder.add_edge("rejected_path", END)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # Run until interrupt
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    result = graph.invoke({}, config=config)
    print(result["__interrupt__"])
    # Output:
    # Interrupt(value={'question': 'Do you approve the following output?', 'llm_output': 'This is the generated output.'}, ...)

    # Simulate resuming with human input
    # To test rejection, replace resume="approve" with resume="reject"
    final_result = graph.invoke(Command(resume="approve"), config=config)
    print(final_result)
    ```

See [how to review tool calls](./review-tool-calls.ipynb) for a more detailed example.

### Review & edit state

<figure markdown="1">
![image](../../concepts/img/human_in_the_loop/edit-graph-state-simple.png){: style="max-height:400px"}
<figcaption>A human can review and edit the state of the graph. This is useful for correcting mistakes or updating the state with additional information.
</figcaption>
</figure>

```python
from langgraph.types import interrupt

def human_editing(state: State):
    ...
    result = interrupt(
        # Interrupt information to surface to the client.
        # Can be any JSON serializable value.
        {
            "task": "Review the output from the LLM and make any necessary edits.",
            "llm_generated_summary": state["llm_generated_summary"]
        }
    )

    # Update the state with the edited text
    return {
        "llm_generated_summary": result["edited_text"] 
    }

# Add the node to the graph in an appropriate location
# and connect it to the relevant nodes.
graph_builder.add_node("human_editing", human_editing)
graph = graph_builder.compile(checkpointer=checkpointer)

...

# After running the graph and hitting the interrupt, the graph will pause.
# Resume it with the edited text.
thread_config = {"configurable": {"thread_id": "some_id"}}
graph.invoke(
    Command(resume={"edited_text": "The edited text"}), 
    config=thread_config
)
```

??? example "Extended example: edit state with interrupt"

    ```python
    from typing import TypedDict
    import uuid

    from langgraph.constants import START, END
    from langgraph.graph import StateGraph
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.memory import MemorySaver

    # Define the graph state
    class State(TypedDict):
        summary: str

    # Simulate an LLM summary generation
    def generate_summary(state: State) -> State:
        return {
            "summary": "The cat sat on the mat and looked at the stars."
        }

    # Human editing node
    def human_review_edit(state: State) -> State:
        result = interrupt({
            "task": "Please review and edit the generated summary if necessary.",
            "generated_summary": state["summary"]
        })
        return {
            "summary": result["edited_summary"]
        }

    # Simulate downstream use of the edited summary
    def downstream_use(state: State) -> State:
        print(f"✅ Using edited summary: {state['summary']}")
        return state

    # Build the graph
    builder = StateGraph(State)
    builder.add_node("generate_summary", generate_summary)
    builder.add_node("human_review_edit", human_review_edit)
    builder.add_node("downstream_use", downstream_use)

    builder.set_entry_point("generate_summary")
    builder.add_edge("generate_summary", "human_review_edit")
    builder.add_edge("human_review_edit", "downstream_use")
    builder.add_edge("downstream_use", END)

    # Set up in-memory checkpointing for interrupt support
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # Invoke the graph until it hits the interrupt
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    result = graph.invoke({}, config=config)

    # Output interrupt payload
    print(result["__interrupt__"])
    # Example output:
    # Interrupt(
    #   value={
    #     'task': 'Please review and edit the generated summary if necessary.',
    #     'generated_summary': 'The cat sat on the mat and looked at the stars.'
    #   },
    #   resumable=True,
    #   ...
    # )

    # Resume the graph with human-edited input
    edited_summary = "The cat lay on the rug, gazing peacefully at the night sky."
    resumed_result = graph.invoke(
        Command(resume={"edited_summary": edited_summary}),
        config=config
    )
    print(resumed_result)
    ```

### Review tool calls

<figure markdown="1">
![image](../../concepts/img/human_in_the_loop/tool-call-review.png){: style="max-height:400px"}
<figcaption>A human can review and edit the output from the LLM before proceeding. This is particularly
critical in applications where the tool calls requested by the LLM may be sensitive or require human oversight.
</figcaption>
</figure>

```python
def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
    # This is the value we'll be providing via Command(resume=<human_review>)
    human_review = interrupt(
        {
            "question": "Is this correct?",
            # Surface tool calls for review
            "tool_call": tool_call
        }
    )

    review_action, review_data = human_review

    # Approve the tool call and continue
    if review_action == "continue":
        return Command(goto="run_tool")

    # Modify the tool call manually and then continue
    elif review_action == "update":
        ...
        updated_msg = get_updated_msg(review_data)
        # Remember that to modify an existing message you will need
        # to pass the message with a matching ID.
        return Command(goto="run_tool", update={"messages": [updated_message]})

    # Give natural language feedback, and then pass that back to the agent
    elif review_action == "feedback":
        ...
        feedback_msg = get_feedback_msg(review_data)
        return Command(goto="call_llm", update={"messages": [feedback_msg]})
```

See [how to review tool calls](./review-tool-calls.ipynb) for a more detailed example.

### Validating human input

If you need to validate the input provided by the human within the graph itself (rather than on the client side), you can achieve this by using multiple interrupt calls within a single node.

```python
from langgraph.types import interrupt

def human_node(state: State):
    """Human node with validation."""
    question = "What is your age?"

    while True:
        answer = interrupt(question)

        # Validate answer, if the answer isn't valid ask for input again.
        if not isinstance(answer, int) or answer < 0:
            question = f"'{answer} is not a valid age. What is your age?"
            answer = None
            continue
        else:
            # If the answer is valid, we can proceed.
            break
            
    print(f"The human in the loop is {answer} years old.")
    return {
        "age": answer
    }
```

??? example "Extended example: validating user input"

    ```python
    from typing import TypedDict
    import uuid

    from langgraph.constants import START, END
    from langgraph.graph import StateGraph
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.memory import MemorySaver

    # Define graph state
    class State(TypedDict):
        age: int

    # Node that asks for human input and validates it
    def get_valid_age(state: State) -> State:
        prompt = "Please enter your age (must be a non-negative integer)."

        while True:
            user_input = interrupt(prompt)

            # Validate the input
            try:
                age = int(user_input)
                if age < 0:
                    raise ValueError("Age must be non-negative.")
                break  # Valid input received
            except (ValueError, TypeError):
                prompt = f"'{user_input}' is not valid. Please enter a non-negative integer for age."

        return {"age": age}

    # Node that uses the valid input
    def report_age(state: State) -> State:
        print(f"✅ Human is {state['age']} years old.")
        return state

    # Build the graph
    builder = StateGraph(State)
    builder.add_node("get_valid_age", get_valid_age)
    builder.add_node("report_age", report_age)

    builder.set_entry_point("get_valid_age")
    builder.add_edge("get_valid_age", "report_age")
    builder.add_edge("report_age", END)

    # Create the graph with a memory checkpointer
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # Run the graph until the first interrupt
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    result = graph.invoke({}, config=config)
    print(result["__interrupt__"])  # First prompt: "Please enter your age..."

    # Simulate an invalid input (e.g., string instead of integer)
    result = graph.invoke(Command(resume="not a number"), config=config)
    print(result["__interrupt__"])  # Follow-up prompt with validation message

    # Simulate a second invalid input (e.g., negative number)
    result = graph.invoke(Command(resume="-10"), config=config)
    print(result["__interrupt__"])  # Another retry

    # Provide valid input
    final_result = graph.invoke(Command(resume="25"), config=config)
    print(final_result)  # Should include the valid age
    ```


## Resume using the `Command` primitive

When the `interrupt` function is used within a graph, execution pauses at that point and awaits user input.

To resume execution, use the [`Command`](../reference/types.md#langgraph.types.Command) primitive, which can be supplied via the `invoke`, `ainvoke`, `stream`, or `astream` methods.

**Providing a response to the `interrupt`:**
To continue execution, pass the user's input using `Command(resume=value)`. The graph resumes execution from the beginning of the node where `interrupt(...)` was initially called. This time, the `interrupt` function will return the value provided in `Command(resume=value)` rather than pausing again.

```python
# Resume graph execution by providing the user's input.
graph.invoke(Command(resume={"age": "25"}), thread_config)
```

## How does resuming from an interrupt work?

!!! warning

    Resuming from an `interrupt` is **different** from Python's `input()` function, where execution resumes from the exact point where the `input()` function was called.

A critical aspect of using `interrupt` is understanding how resuming works. When you resume execution after an `interrupt`, graph execution starts from the **beginning** of the **graph node** where the last `interrupt` was triggered.

**All** code from the beginning of the node to the `interrupt` will be re-executed.

```python
counter = 0
def node(state: State):
    # All the code from the beginning of the node to the interrupt will be re-executed
    # when the graph resumes.
    global counter
    counter += 1
    print(f"> Entered the node: {counter} # of times")
    # Pause the graph and wait for user input.
    answer = interrupt()
    print("The value of counter is:", counter)
    ...
```

Upon **resuming** the graph, the counter will be incremented a second time, resulting in the following output:

```pycon
> Entered the node: 2 # of times
The value of counter is: 2
```

### Resuming multiple interrupts with one invocation

If you have multiple interrupts in the task queue, you can use `Command.resume` with a dictionary mapping
of interrupt ids to resume values to resume multiple interrupts with a single `invoke` / `stream` call.

For example, once your graph has been interrupted (multiple times, theoretically) and is stalled:

```python
resume_map = {
    i.interrupt_id: f"human input for prompt {i.value}"
    for i in parent.get_state(thread_config).interrupts
}

parent_graph.invoke(Command(resume=resume_map), config=thread_config)
```

## Common pitfalls

### Side-effects

Place code with side effects, such as API calls, **after** the `interrupt` to avoid duplication, as these are re-triggered every time the node is resumed.

=== "Side effects before interrupt (BAD)"

    This code will re-execute the API call another time when the node is resumed from
    the `interrupt`.

    This can be problematic if the API call is not idempotent or is just expensive.

    ```python
    from langgraph.types import interrupt

    def human_node(state: State):
        """Human node with validation."""
        api_call(...) # This code will be re-executed when the node is resumed.
        answer = interrupt(question)
    ```

=== "Side effects after interrupt (OK)"

    ```python
    from langgraph.types import interrupt

    def human_node(state: State):
        """Human node with validation."""
        
        answer = interrupt(question)
        
        api_call(answer) # OK as it's after the interrupt
    ```

=== "Side effects in a separate node (OK)"

    ```python
    from langgraph.types import interrupt

    def human_node(state: State):
        """Human node with validation."""
        
        answer = interrupt(question)
        
        return {
            "answer": answer
        }

    def api_call_node(state: State):
        api_call(...) # OK as it's in a separate node
    ```

### Subgraphs called as functions

When invoking a subgraph [as a function](low_level.md#as-a-function), the **parent graph** will resume execution from the **beginning of the node** where the subgraph was invoked (and where an `interrupt` was triggered). Similarly, the **subgraph**, will resume from the **beginning of the node** where the `interrupt()` function was called.

For example,

```python
def node_in_parent_graph(state: State):
    some_code()  # <-- This will re-execute when the subgraph is resumed.
    # Invoke a subgraph as a function.
    # The subgraph contains an `interrupt` call.
    subgraph_result = subgraph.invoke(some_input)
    ...
```

??? example "Extended example: parent and subgraph execution flow"

      Say we have a parent graph with 3 nodes:

      **Parent Graph**: `node_1` → `node_2` (subgraph call) → `node_3`

      And the subgraph has 3 nodes, where the second node contains an `interrupt`:

      **Subgraph**: `sub_node_1` → `sub_node_2` (`interrupt`) → `sub_node_3`

      When resuming the graph, the execution will proceed as follows:

      1. **Skip `node_1`** in the parent graph (already executed, graph state was saved in snapshot).
      2. **Re-execute `node_2`** in the parent graph from the start.
      3. **Skip `sub_node_1`** in the subgraph (already executed, graph state was saved in snapshot).
      4. **Re-execute `sub_node_2`** in the subgraph from the beginning.
      5. Continue with `sub_node_3` and subsequent nodes.

      Here is abbreviated example code that you can use to understand how subgraphs work with interrupts.
      It counts the number of times each node is entered and prints the count.

      ```python
      import uuid
      from typing import TypedDict

      from langgraph.graph import StateGraph
      from langgraph.constants import START
      from langgraph.types import interrupt, Command
      from langgraph.checkpoint.memory import MemorySaver


      class State(TypedDict):
         """The graph state."""
         state_counter: int


      counter_node_in_subgraph = 0

      def node_in_subgraph(state: State):
         """A node in the sub-graph."""
         global counter_node_in_subgraph
         counter_node_in_subgraph += 1  # This code will **NOT** run again!
         print(f"Entered `node_in_subgraph` a total of {counter_node_in_subgraph} times")

      counter_human_node = 0

      def human_node(state: State):
         global counter_human_node
         counter_human_node += 1 # This code will run again!
         print(f"Entered human_node in sub-graph a total of {counter_human_node} times")
         answer = interrupt("what is your name?")
         print(f"Got an answer of {answer}")


      checkpointer = MemorySaver()

      subgraph_builder = StateGraph(State)
      subgraph_builder.add_node("some_node", node_in_subgraph)
      subgraph_builder.add_node("human_node", human_node)
      subgraph_builder.add_edge(START, "some_node")
      subgraph_builder.add_edge("some_node", "human_node")
      subgraph = subgraph_builder.compile(checkpointer=checkpointer)


      counter_parent_node = 0

      def parent_node(state: State):
         """This parent node will invoke the subgraph."""
         global counter_parent_node

         counter_parent_node += 1 # This code will run again on resuming!
         print(f"Entered `parent_node` a total of {counter_parent_node} times")
  
         # Please note that we're intentionally incrementing the state counter
         # in the graph state as well to demonstrate that the subgraph update
         # of the same key will not conflict with the parent graph (until
         subgraph_state = subgraph.invoke(state)
         return subgraph_state


      builder = StateGraph(State)
      builder.add_node("parent_node", parent_node)
      builder.add_edge(START, "parent_node")

      # A checkpointer must be enabled for interrupts to work!
      checkpointer = MemorySaver()
      graph = builder.compile(checkpointer=checkpointer)

      config = {
         "configurable": {
            "thread_id": uuid.uuid4(),
         }
      }

      for chunk in graph.stream({"state_counter": 1}, config):
         print(chunk)

      print('--- Resuming ---')

      for chunk in graph.stream(Command(resume="35"), config):
         print(chunk)
      ```

      This will print out

      ```pycon
      Entered `parent_node` a total of 1 times
      Entered `node_in_subgraph` a total of 1 times
      Entered human_node in sub-graph a total of 1 times
      {'__interrupt__': (Interrupt(value='what is your name?', resumable=True, ns=['parent_node:4c3a0248-21f0-1287-eacf-3002bc304db4', 'human_node:2fe86d52-6f70-2a3f-6b2f-b1eededd6348'], when='during'),)}
      --- Resuming ---
      Entered `parent_node` a total of 2 times
      Entered human_node in sub-graph a total of 2 times
      Got an answer of 35
      {'parent_node': {'state_counter': 1}}
      ```



### Using multiple interrupts

Using multiple interrupts within a **single** node can be helpful for patterns like [validating human input](#validating-human-input). However, using multiple interrupts in the same node can lead to unexpected behavior if not handled carefully.

When a node contains multiple interrupt calls, LangGraph keeps a list of resume values specific to the task executing the node. Whenever execution resumes, it starts at the beginning of the node. For each interrupt encountered, LangGraph checks if a matching value exists in the task's resume list. Matching is **strictly index-based**, so the order of interrupt calls within the node is critical.

To avoid issues, refrain from dynamically changing the node's structure between executions. This includes adding, removing, or reordering interrupt calls, as such changes can result in mismatched indices. These problems often arise from unconventional patterns, such as mutating state via `Command(resume=..., update=SOME_STATE_MUTATION)` or relying on global variables to modify the node’s structure dynamically.

??? example "Extended example: incorrect code that introduces non-determinism"

    ```python
    import uuid
    from typing import TypedDict, Optional

    from langgraph.graph import StateGraph
    from langgraph.constants import START 
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.memory import MemorySaver


    class State(TypedDict):
        """The graph state."""

        age: Optional[str]
        name: Optional[str]


    def human_node(state: State):
        if not state.get('name'):
            name = interrupt("what is your name?")
        else:
            name = "N/A"

        if not state.get('age'):
            age = interrupt("what is your age?")
        else:
            age = "N/A"
            
        print(f"Name: {name}. Age: {age}")
        
        return {
            "age": age,
            "name": name,
        }


    builder = StateGraph(State)
    builder.add_node("human_node", human_node)
    builder.add_edge(START, "human_node")

    # A checkpointer must be enabled for interrupts to work!
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
        }
    }

    for chunk in graph.stream({"age": None, "name": None}, config):
        print(chunk)

    for chunk in graph.stream(Command(resume="John", update={"name": "foo"}), config):
        print(chunk)
    ```

    ```pycon
    {'__interrupt__': (Interrupt(value='what is your name?', resumable=True, ns=['human_node:3a007ef9-c30d-c357-1ec1-86a1a70d8fba'], when='during'),)}
    Name: N/A. Age: John
    {'human_node': {'age': 'John', 'name': 'N/A'}}
    ```