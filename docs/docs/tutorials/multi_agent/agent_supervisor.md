# Multi-agent supervisor

[**Supervisor**](../../concepts/multi_agent.md#supervisor) is a multi-agent architecture where **specialized** agents are coordinated by a central **supervisor agent**. The supervisor agent controls all communication flow and task delegation, making decisions about which agent to invoke based on the current context and task requirements.

In this tutorial, you will build a supervisor system with two agents — a research and a math expert. By the end of the tutorial you will:

1. Build specialized research and math agents
2. Build a supervisor for orchestrating them with the prebuilt [`langgraph-supervisor`](https://langchain-ai.github.io/langgraph/agents/multi-agent/#supervisor)
3. Build a supervisor from scratch
4. Implement advanced task delegation

![diagram](assets/diagram.png)

## Setup

First, let's install required packages and set our API keys

```python
%%capture --no-stderr
%pip install -U langgraph langgraph-supervisor langchain-tavily "langchain[openai]"
```

```python
import getpass
import os


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
```

!!! tip
    Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. [LangSmith](https://docs.smith.langchain.com) lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph.

## 1. Create worker agents

First, let's create our specialized worker agents — research agent and math agent:

* Research agent will have access to a web search tool using [Tavily API](https://tavily.com/)
* Math agent will have access to simple math tools (`add`, `multiply`, `divide`)

### Research agent

For web search, we will use `TavilySearch` tool from `langchain-tavily`:

```python
from langchain_tavily import TavilySearch

web_search = TavilySearch(max_results=3)
web_search_results = web_search.invoke("who is the mayor of NYC?")

print(web_search_results["results"][0]["content"])
```

**Output:**
```
Find events, attractions, deals, and more at nyctourism.com Skip Main Navigation Menu The Official Website of the City of New York Text Size Powered by Translate SearchSearch Primary Navigation The official website of NYC Home NYC Resources NYC311 Office of the Mayor Events Connect Jobs Search Office of the Mayor | Mayor's Bio | City of New York Secondary Navigation MayorBiographyNewsOfficials Eric L. Adams 110th Mayor of New York City Mayor Eric Adams has served the people of New York City as an NYPD officer, State Senator, Brooklyn Borough President, and now as the 110th Mayor of the City of New York. Mayor Eric Adams has served the people of New York City as an NYPD officer, State Senator, Brooklyn Borough President, and now as the 110th Mayor of the City of New York. He gave voice to a diverse coalition of working families in all five boroughs and is leading the fight to bring back New York City's economy, reduce inequality, improve public safety, and build a stronger, healthier city that delivers for all New Yorkers. As the representative of one of the nation's largest counties, Eric fought tirelessly to grow the local economy, invest in schools, reduce inequality, improve public safety, and advocate for smart policies and better government that delivers for all New Yorkers.
```

To create individual worker agents, we will use LangGraph's prebuilt [agent](../../agents/agents.md).

```python
from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)
```

Let's [run the agent](../../agents/run_agents.md) to verify that it behaves as expected. 

!!! note "We'll use `pretty_print_messages` helper to render the streamed agent outputs nicely"

  ```python
  from langchain_core.messages import convert_to_messages


  def pretty_print_message(message, indent=False):
      pretty_message = message.pretty_repr(html=True)
      if not indent:
          print(pretty_message)
          return

      indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
      print(indented)


  def pretty_print_messages(update, last_message=False):
      is_subgraph = False
      if isinstance(update, tuple):
          ns, update = update
          # skip parent graph updates in the printouts
          if len(ns) == 0:
              return

          graph_id = ns[-1].split(":")[0]
          print(f"Update from subgraph {graph_id}:")
          print("\n")
          is_subgraph = True

      for node_name, node_update in update.items():
          update_label = f"Update from node {node_name}:"
          if is_subgraph:
              update_label = "\t" + update_label

          print(update_label)
          print("\n")

          messages = convert_to_messages(node_update["messages"])
          if last_message:
              messages = messages[-1:]

          for m in messages:
              pretty_print_message(m, indent=is_subgraph)
          print("\n")
  ```

  ```python
  for chunk in research_agent.stream(
      {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
  ):
      pretty_print_messages(chunk)
  ```

  **Output:**
  ```
  Update from node agent:


  ================================== Ai Message ==================================
  Name: research_agent
  Tool Calls:
    tavily_search (call_U748rQhQXT36sjhbkYLSXQtJ)
   Call ID: call_U748rQhQXT36sjhbkYLSXQtJ
    Args:
      query: current mayor of New York City
      search_depth: basic


  Update from node tools:


  ================================= Tool Message ==================================
  Name: tavily_search

  {"query": "current mayor of New York City", "follow_up_questions": null, "answer": null, "images": [], "results": [{"title": "List of mayors of New York City - Wikipedia", "url": "https://en.wikipedia.org/wiki/List_of_mayors_of_New_York_City", "content": "The mayor of New York City is the chief executive of the Government of New York City, as stipulated by New York City's charter.The current officeholder, the 110th in the sequence of regular mayors, is Eric Adams, a member of the Democratic Party.. During the Dutch colonial period from 1624 to 1664, New Amsterdam was governed by the Director of Netherland.", "score": 0.9039154, "raw_content": null}, {"title": "Office of the Mayor | Mayor's Bio | City of New York - NYC.gov", "url": "https://www.nyc.gov/office-of-the-mayor/bio.page", "content": "Mayor Eric Adams has served the people of New York City as an NYPD officer, State Senator, Brooklyn Borough President, and now as the 110th Mayor of the City of New York. He gave voice to a diverse coalition of working families in all five boroughs and is leading the fight to bring back New York City's economy, reduce inequality, improve", "score": 0.8405867, "raw_content": null}, {"title": "Eric Adams - Wikipedia", "url": "https://en.wikipedia.org/wiki/Eric_Adams", "content": "Eric Leroy Adams (born September 1, 1960) is an American politician and former police officer who has served as the 110th mayor of New York City since 2022. Adams was an officer in the New York City Transit Police and then the New York City Police Department (```
  ```

### Math agent

For math agent tools we will use [vanilla Python functions](../../how-tos/tool-calling.md#define-a-tool):

```python
def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


math_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)
```

Let's run the math agent:

```python
for chunk in math_agent.stream(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 7"}]}
):
    pretty_print_messages(chunk)
```

**Output:**
```
Update from node agent:


================================== Ai Message ==================================
Name: math_agent
Tool Calls:
  add (call_p6OVLDHB4LyCNCxPOZzWR15v)
 Call ID: call_p6OVLDHB4LyCNCxPOZzWR15v
  Args:
    a: 3
    b: 5


Update from node tools:


================================= Tool Message ==================================
Name: add

8.0


Update from node agent:


================================== Ai Message ==================================
Name: math_agent
Tool Calls:
  multiply (call_EoaWHMLFZAX4AkajQCtZvbli)
 Call ID: call_EoaWHMLFZAX4AkajQCtZvbli
  Args:
    a: 8
    b: 7


Update from node tools:


================================= Tool Message ==================================
Name: multiply

56.0


Update from node agent:


================================== Ai Message ==================================
Name: math_agent

56


```

## 2. Create supervisor with `langgraph-supervisor`

To implement out multi-agent system, we will use @[`create_supervisor`][create_supervisor] from the prebuilt `langgraph-supervisor` library:

```python
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1"),
    agents=[research_agent, math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()
```

```python
from IPython.display import display, Image

display(Image(supervisor.get_graph().draw_mermaid_png()))
```

![Graph](assets/output.png)

**Note:** When you run this code, it will generate and display a visual representation of the supervisor graph showing the flow between the supervisor and worker agents.

Let's now run it with a query that requires both agents:

* research agent will look up the necessary GDP information
* math agent will perform division to find the percentage of NY state GDP, as requested

```python
for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]
```

**Output:**
```
Update from node supervisor:


================================= Tool Message ==================================
Name: transfer_to_research_agent

Successfully transferred to research_agent


Update from node research_agent:


================================= Tool Message ==================================
Name: transfer_back_to_supervisor

Successfully transferred back to supervisor


Update from node supervisor:


================================= Tool Message ==================================
Name: transfer_to_math_agent

Successfully transferred to math_agent


Update from node math_agent:


================================= Tool Message ==================================
Name: transfer_back_to_supervisor

Successfully transferred back to supervisor


Update from node supervisor:


================================== Ai Message ==================================
Name: supervisor

In 2024, the US GDP was $29.18 trillion and New York State's GDP was $2.297 trillion. New York State accounted for approximately 7.87% of the total US GDP in 2024.


```

## 3. Create supervisor from scratch

Let's now implement this same multi-agent system from scratch. We will need to:

1. [Set up how the supervisor communicates](#set-up-agent-communication) with individual agents
2. [Create the supervisor agent](#create-supervisor-agent)
3. Combine supervisor and worker agents into a [single multi-agent graph](#create-multi-agent-graph).

### Set up agent communication

We will need to define a way for the supervisor agent to communicate with the worker agents. A common way to implement this in multi-agent architectures is using **handoffs**, where one agent *hands off* control to another. Handoffs allow you to specify:

- **destination**: target agent to transfer to
- **payload**: information to pass to that agent

We will implement handoffs via **handoff tools** and give these tools to the supervisor agent: when the supervisor calls these tools, it will hand off control to a worker agent, passing the full message history to that agent.

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        # highlight-next-line
        return Command(
            # highlight-next-line
            goto=agent_name,  # (1)!
            # highlight-next-line
            update={**state, "messages": state["messages"] + [tool_message]},  # (2)!
            # highlight-next-line
            graph=Command.PARENT,  # (3)!
        )

    return handoff_tool


# Handoffs
assign_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent.",
)

assign_to_math_agent = create_handoff_tool(
    agent_name="math_agent",
    description="Assign task to a math agent.",
)
```

1. Name of the agent or node to hand off to.
2. Take the agent's messages and add them to the parent's state as part of the handoff. The next agent will see the parent state.
3. Indicate to LangGraph that we need to navigate to agent node in a **parent** multi-agent graph.

### Create supervisor agent

Then, let's create the supervisor agent with the handoff tools we just defined. We will use the prebuilt @[`create_react_agent`][create_react_agent]:

```python
supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[assign_to_research_agent, assign_to_math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
)
```

### Create multi-agent graph

Putting this all together, let's create a graph for our overall multi-agent system. We will add the supervisor and the individual agents as subgraph [nodes](../../concepts/low_level.md#nodes).

```python
from langgraph.graph import END

# Define the multi-agent supervisor graph
supervisor = (
    StateGraph(MessagesState)
    # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
    .add_node(supervisor_agent, destinations=("research_agent", "math_agent", END))
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START, "supervisor")
    # always return back to the supervisor
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()
)
```

Notice that we've added explicit [edges](../../concepts/low_level.md#edges) from worker agents back to the supervisor — this means that they are guaranteed to return control back to the supervisor. If you want the agents to respond directly to the user (i.e., turn the system into a router, you can remove these edges).

```python
from IPython.display import display, Image

display(Image(supervisor.get_graph().draw_mermaid_png()))
```

![Graph](assets/multi-output.png)

**Note:** When you run this code, it will generate and display a visual representation of the multi-agent supervisor graph showing the flow between the supervisor and worker agents.

With the multi-agent graph created, let's now run it!

```python
for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]
```

**Output:**
```
Update from node supervisor:


================================= Tool Message ==================================
Name: transfer_to_research_agent

Successfully transferred to research_agent


Update from node research_agent:


================================== Ai Message ==================================
Name: research_agent

- US GDP in 2024 is projected to be about $28.18 trillion USD (Statista; CBO projection).
- New York State's nominal GDP for 2024 is estimated at approximately $2.16 trillion USD (various economic reports).
- New York State's share of US GDP in 2024 is roughly 7.7%.

Sources:
- https://www.statista.com/statistics/216985/forecast-of-us-gross-domestic-product/
- https://nyassembly.gov/Reports/WAM/2025economic_revenue/2025_report.pdf?v=1740533306


Update from node supervisor:


================================= Tool Message ==================================
Name: transfer_to_math_agent

Successfully transferred to math_agent


Update from node math_agent:


================================== Ai Message ==================================
Name: math_agent

US GDP in 2024: $28.18 trillion
New York State GDP in 2024: $2.16 trillion
Percentage of US GDP from New York State: 7.67%


Update from node supervisor:


================================== Ai Message ==================================
Name: supervisor

Here are your results:

- 2024 US GDP (projected): $28.18 trillion USD
- 2024 New York State GDP (estimated): $2.16 trillion USD
- New York State's share of US GDP: approximately 7.7%

If you need the calculation steps or sources, let me know!


```

Let's examine the full resulting message history:

```python
for message in final_message_history:
    message.pretty_print()
```

**Output:**
```
================================ Human Message ==================================

find US and New York state GDP in 2024. what % of US GDP was New York state?
================================== Ai Message ===================================
Name: supervisor
Tool Calls:
  transfer_to_research_agent (call_KlGgvF5ahlAbjX8d2kHFjsC3)
 Call ID: call_KlGgvF5ahlAbjX8d2kHFjsC3
  Args:
================================= Tool Message ==================================
Name: transfer_to_research_agent

Successfully transferred to research_agent
================================== Ai Message ===================================
Name: research_agent
Tool Calls:
  tavily_search (call_ZOaTVUA6DKrOjWQldLhtrsO2)
 Call ID: call_ZOaTVUA6DKrOjWQldLhtrsO2
  Args:
    query: US GDP 2024 estimate or actual
    search_depth: advanced
  tavily_search (call_QsRAasxW9K03lTlqjuhNLFbZ)
 Call ID: call_QsRAasxW9K03lTlqjuhNLFbZ
  Args:
    query: New York state GDP 2024 estimate or actual
    search_depth: advanced
================================= Tool Message ==================================
Name: tavily_search

{"query": "US GDP 2024 estimate or actual", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://www.advisorperspectives.com/dshort/updates/2025/05/29/gdp-gross-domestic-product-q1-2025-second-estimate", "title": "Q1 GDP Second Estimate: Real GDP at -0.2%, Higher Than Expected", "content": "> Real gross domestic product (GDP) decreased at an annual rate of 0.2 percent in the first quarter of 2025 (January, February, and March), according to the second estimate released by the U.S. Bureau of Economic Analysis. In the fourth quarter of 2024, real GDP increased 2.4 percent. The decrease in real GDP in the first quarter primarily reflected an increase in imports, which are a subtraction in the calculation of GDP, and a decrease in government spending. These movements were partly [...] by [Harry Mamaysky](https://www.advisor```
```

!!! important
    You can see that the supervisor system appends **all** of the individual agent messages (i.e., their internal tool-calling loop) to the full message history. This means that on every supervisor turn, supervisor agent sees this full history. If you want more control over:

    * **how inputs are passed to agents**: you can use LangGraph @[`Send()`][Send] primitive to directly send data to the worker agents during the handoff. See the [task delegation](#4-create-delegation-tasks) example below
    * **how agent outputs are added**: you can control how much of the agent's internal message history is added to the overall supervisor message history by wrapping the agent in a separate node function:

        ```python
        def call_research_agent(state):
            # return agent's final response,
            # excluding inner monologue
            response = research_agent.invoke(state)
            # highlight-next-line
            return {"messages": response["messages"][-1]}
        ```

## 4. Create delegation tasks

So far the individual agents relied on **interpreting full message history** to determine their tasks. An alternative approach is to ask the supervisor to **formulate a task explicitly**. We can do so by adding a `task_description` parameter to the `handoff_tool` function.

```python
from langgraph.types import Send


def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            # highlight-next-line
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


assign_to_research_agent_with_description = create_task_description_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent.",
)

assign_to_math_agent_with_description = create_task_description_handoff_tool(
    agent_name="math_agent",
    description="Assign task to a math agent.",
)

supervisor_agent_with_description = create_react_agent(
    model="openai:gpt-4.1",
    tools=[
        assign_to_research_agent_with_description,
        assign_to_math_agent_with_description,
    ],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this assistant\n"
        "- a math agent. Assign math-related tasks to this assistant\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
)

supervisor_with_description = (
    StateGraph(MessagesState)
    .add_node(
        supervisor_agent_with_description, destinations=("research_agent", "math_agent")
    )
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()
)
```

!!! note
    We're using @[`Send()`][Send] primitive in the `handoff_tool`. This means that instead of receiving the full `supervisor` graph state as input, each worker agent only sees the contents of the `Send` payload. In this example, we're sending the task description as a single "human" message.

Let's now running it with the same input query:

```python
for chunk in supervisor_with_description.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
            }
        ]
    },
    subgraphs=True,
):
    pretty_print_messages(chunk, last_message=True)
```

**Output:**
```
Update from subgraph supervisor:


	Update from node agent:


	================================== Ai Message ==================================
	Name: supervisor
	Tool Calls:
	  transfer_to_research_agent (call_tk8q8py8qK6MQz6Kj6mijKua)
	 Call ID: call_tk8q8py8qK6MQz6Kj6mijKua
	  Args:
	    task_description: Find the 2024 GDP (Gross Domestic Product) for both the United States and New York state, using the most up-to-date and reputable sources available. Provide both GDP values and cite the data sources.


Update from subgraph research_agent:


	Update from node agent:


	================================== Ai Message ==================================
	Name: research_agent
	Tool Calls:
	  tavily_search (call_KqvhSvOIhAvXNsT6BOwbPlRB)
	 Call ID: call_KqvhSvOIhAvXNsT6BOwbPlRB
	  Args:
	    query: 2024 United States GDP value from a reputable source
	    search_depth: advanced
	  tavily_search (call_kbbAWBc9KwCWKHmM5v04H88t)
	 Call ID: call_kbbAWBc9KwCWKHmM5v04H88t
	  Args:
	    query: 2024 New York state GDP value from a reputable source
	    search_depth: advanced


Update from subgraph research_agent:


	Update from node tools:


	================================= Tool Message ==================================
	Name: tavily_search
	
	{"query": "2024 United States GDP value from a reputable source", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://www.focus-economics.com/countries/united-states/", "title": "United States Economy Overview - Focus Economics", "content": "The United States' Macroeconomic Analysis:\n------------------------------------------\n\n**Nominal GDP of USD 29,185 billion in 2024.**\n\n**Nominal GDP of USD 29,179 billion in 2024.**\n\n**GDP per capita of USD 86,635 compared to the global average of USD 10,589.**\n\n**GDP per capita of USD 86,652 compared to the global average of USD 10,589.**\n\n**Average real GDP growth of 2.5% over the last decade.**\n\n**Average real GDP growth of ```
``` 
