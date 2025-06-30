# Time travel

In a typical chatbot workflow, the user interacts with the bot one or more times to accomplish a task. [Memory](./3-add-memory.md) and a [human-in-the-loop](./4-human-in-the-loop.md) enable checkpoints in the graph state and control future responses.

What if you want a user to be able to start from a previous response and explore a different outcome? Or what if you want users to be able to rewind your chatbot's work to fix mistakes or try a different strategy, something that is common in applications like autonomous software engineers?

You can create these types of experiences using LangGraph's built-in **time travel** functionality. 

!!! note

    This tutorial builds on [Customize state](./5-customize-state.md).

## 1. Rewind your graph

:::python
Rewind your graph by fetching a checkpoint using the graph's `get_state_history` method. You can then resume execution at this previous point in time.
:::

:::js
Rewind your graph by fetching a checkpoint using the graph's `getStateHistory` method. You can then resume execution at this previous point in time.
:::

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

:::python
<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

```python
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```
:::

:::js
```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__...";
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Time Travel: LangGraphJS";
```

```typescript
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { BaseMessage } from "@langchain/core/messages";
import {
  MessagesAnnotation,
  StateGraph,
  START,
  Command,
  interrupt,
  MemorySaver
} from "@langchain/langgraph";

const model = new ChatOpenAI({ model: "gpt-4o" });


/**
 * Call LLM with structured output to get a natural language response as well as a target agent (node) to go to next.
 * @param messages list of messages to pass to the LLM
 * @param targetAgentNodes list of the node names of the target agents to navigate to
 */
function callLlm(messages: BaseMessage[], targetAgentNodes: string[]) {
  // define the schema for the structured output:
  // - model's text response (`response`)
  // - name of the node to go to next (or 'finish')
  const outputSchema = z.object({
    response: z.string().describe("A human readable response to the original question. Does not need to be a final response. Will be streamed back to the user."),
    goto: z.enum(["finish", ...targetAgentNodes]).describe("The next agent to call, or 'finish' if the user's query has been resolved. Must be one of the specified values."),
  })
  return model.withStructuredOutput(outputSchema, { name: "Response" }).invoke(messages)
}

async function travelAdvisor(
  state: typeof MessagesAnnotation.State
): Promise<Command> {
  const systemPrompt = 
      "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). " +
      "If you need specific sightseeing recommendations, ask 'sightseeingAdvisor' for help. " +
      "If you need hotel recommendations, ask 'hotelAdvisor' for help. " +
      "If you have enough information to respond to the user, return 'finish'. " +
      "Never mention other agents by name.";

  const messages = [{"role": "system", "content": systemPrompt}, ...state.messages] as BaseMessage[];
  const targetAgentNodes = ["sightseeingAdvisor", "hotelAdvisor"];
  const response = await callLlm(messages, targetAgentNodes);
  const aiMsg = {"role": "ai", "content": response.response, "name": "travelAdvisor"};
  
  let goto = response.goto;
  if (goto === "finish") {
      goto = "human";
  }

  return new Command({goto, update: { "messages": [aiMsg] } });
}

async function sightseeingAdvisor(
  state: typeof MessagesAnnotation.State
): Promise<Command> {
  const systemPrompt = 
      "You are a travel expert that can provide specific sightseeing recommendations for a given destination. " +
      "If you need general travel help, go to 'travelAdvisor' for help. " +
      "If you need hotel recommendations, go to 'hotelAdvisor' for help. " +
      "If you have enough information to respond to the user, return 'finish'. " +
      "Never mention other agents by name.";

  const messages = [{"role": "system", "content": systemPrompt}, ...state.messages] as BaseMessage[];
  const targetAgentNodes = ["travelAdvisor", "hotelAdvisor"];
  const response = await callLlm(messages, targetAgentNodes);
  const aiMsg = {"role": "ai", "content": response.response, "name": "sightseeingAdvisor"};
  
  let goto = response.goto;
  if (goto === "finish") {
      goto = "human";
  }

  return new Command({ goto, update: {"messages": [aiMsg] } });
}

async function hotelAdvisor(
  state: typeof MessagesAnnotation.State
): Promise<Command> {
  const systemPrompt = 
      "You are a travel expert that can provide hotel recommendations for a given destination. " +
      "If you need general travel help, ask 'travelAdvisor' for help. " +
      "If you need specific sightseeing recommendations, ask 'sightseeingAdvisor' for help. " +
      "If you have enough information to respond to the user, return 'finish'. " +
      "Never mention other agents by name.";

  const messages = [{"role": "system", "content": systemPrompt}, ...state.messages] as BaseMessage[];
  const targetAgentNodes = ["travelAdvisor", "sightseeingAdvisor"];
  const response = await callLlm(messages, targetAgentNodes);
  const aiMsg = {"role": "ai", "content": response.response, "name": "hotelAdvisor"};
  
  let goto = response.goto;
  if (goto === "finish") {
      goto = "human";
  }

  return new Command({ goto, update: {"messages": [aiMsg] } });
}

function humanNode(
  state: typeof MessagesAnnotation.State
): Command {
  const userInput: string = interrupt("Ready for user input.");

  let activeAgent: string | undefined = undefined;

  // Look up the active agent
  for (let i = state.messages.length - 1; i >= 0; i--) {
      if (state.messages[i].name) {
          activeAgent = state.messages[i].name;
          break;
      }
  }

  if (!activeAgent) {
      throw new Error("Could not determine the active agent.");
  }

  return new Command({
      goto: activeAgent,
      update: {
        "messages": [
            {
                "role": "human",
                "content": userInput,
            }
        ]
      }
  });
}

const builder = new StateGraph(MessagesAnnotation)
  .addNode("travelAdvisor", travelAdvisor, {
    ends: ["sightseeingAdvisor", "hotelAdvisor"]
  })
  .addNode("sightseeingAdvisor", sightseeingAdvisor, {
    ends: ["human", "travelAdvisor", "hotelAdvisor"]
  })
  .addNode("hotelAdvisor", hotelAdvisor, {
    ends: ["human", "travelAdvisor", "sightseeingAdvisor"]
  })
  // This adds a node to collect human input, which will route
  // back to the active agent.
  .addNode("human", humanNode, {
    ends: ["hotelAdvisor", "sightseeingAdvisor", "travelAdvisor", "human"]
  })
  // We'll always start with a general travel advisor.
  .addEdge(START, "travelAdvisor")

const checkpointer = new MemorySaver()
const graph = builder.compile({ checkpointer })
```

```typescript
import * as tslab from "tslab";

const drawableGraph = graph.getGraph();
const image = await drawableGraph.drawMermaidPng();
const arrayBuffer = await image.arrayBuffer();

await tslab.display.png(new Uint8Array(arrayBuffer));
```
:::

## 2. Add steps

Add steps to your graph. Every step will be checkpointed in its state history:

:::python
``` python
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

I'm learning LangGraph. Could you do some research on it for me?
================================== Ai Message ==================================

[{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and accurate information, I'll use the Tavily search engine to look this up. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01BscbfJJB9EWJFqGrN6E54e', 'input': {'query': 'LangGraph latest information and features'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01BscbfJJB9EWJFqGrN6E54e)
 Call ID: toolu_01BscbfJJB9EWJFqGrN6E54e
  Args:
    query: LangGraph latest information and features
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://blockchain.news/news/langchain-new-features-upcoming-events-update", "content": "LangChain, a leading platform in the AI development space, has released its latest updates, showcasing new use cases and enhancements across its ecosystem. According to the LangChain Blog, the updates cover advancements in LangGraph Platform, LangSmith's self-improving evaluators, and revamped documentation for LangGraph."}, {"url": "https://blog.langchain.dev/langgraph-platform-announce/", "content": "With these learnings under our belt, we decided to couple some of our latest offerings under LangGraph Platform. LangGraph Platform today includes LangGraph Server, LangGraph Studio, plus the CLI and SDK. ... we added features in LangGraph Server to deliver on a few key value areas. Below, we'll focus on these aspects of LangGraph Platform."}]
================================== Ai Message ==================================

Thank you for your patience. I've found some recent information about LangGraph for you. Let me summarize the key points:

1. LangGraph is part of the LangChain ecosystem, which is a leading platform in AI development.

2. Recent updates and features of LangGraph include:

   a. LangGraph Platform: This seems to be a cloud-based version of LangGraph, though specific details weren't provided in the search results.
...
3. Keep an eye on LangGraph Platform developments, as cloud-based solutions often provide an easier starting point for learners.
4. Consider how LangGraph fits into the broader LangChain ecosystem, especially its interaction with tools like LangSmith.

Is there any specific aspect of LangGraph you'd like to know more about? I'd be happy to do a more focused search on particular features or use cases.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

```python
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Ya that's helpful. Maybe I'll build an autonomous agent with it!
================================== Ai Message ==================================

[{'text': "That's an exciting idea! Building an autonomous agent with LangGraph is indeed a great application of this technology. LangGraph is particularly well-suited for creating complex, multi-step AI workflows, which is perfect for autonomous agents. Let me gather some more specific information about using LangGraph for building autonomous agents.", 'type': 'text'}, {'id': 'toolu_01QWNHhUaeeWcGXvA4eHT7Zo', 'input': {'query': 'Building autonomous agents with LangGraph examples and tutorials'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01QWNHhUaeeWcGXvA4eHT7Zo)
 Call ID: toolu_01QWNHhUaeeWcGXvA4eHT7Zo
  Args:
    query: Building autonomous agents with LangGraph examples and tutorials
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://towardsdatascience.com/building-autonomous-multi-tool-agents-with-gemini-2-0-and-langgraph-ad3d7bd5e79d", "content": "Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph | by Youness Mansar | Jan, 2025 | Towards Data Science Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph A practical tutorial with full code examples for building and running multi-tool agents Towards Data Science LLMs are remarkable — they can memorize vast amounts of information, answer general knowledge questions, write code, generate stories, and even fix your grammar. In this tutorial, we are going to build a simple LLM agent that is equipped with four tools that it can use to answer a user's question. This Agent will have the following specifications: Follow Published in Towards Data Science --------------------------------- Your home for data science and AI. Follow Follow Follow"}, {"url": "https://github.com/anmolaman20/Tools_and_Agents", "content": "GitHub - anmolaman20/Tools_and_Agents: This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository serves as a comprehensive guide for building AI-powered agents using Langchain and Langgraph. It provides hands-on examples, practical tutorials, and resources for developers and AI enthusiasts to master building intelligent systems and workflows. AI Agent Development: Gain insights into creating intelligent systems that think, reason, and adapt in real time. This repository is ideal for AI practitioners, developers exploring language models, or anyone interested in building intelligent systems. This repository provides resources for building AI agents using Langchain and Langgraph."}]
================================== Ai Message ==================================

Great idea! Building an autonomous agent with LangGraph is definitely an exciting project. Based on the latest information I've found, here are some insights and tips for building autonomous agents with LangGraph:

1. Multi-Tool Agents: LangGraph is particularly well-suited for creating autonomous agents that can use multiple tools. This allows your agent to have a diverse set of capabilities and choose the right tool for each task.

2. Integration with Large Language Models (LLMs): You can combine LangGraph with powerful LLMs like Gemini 2.0 to create more intelligent and capable agents. The LLM can serve as the "brain" of your agent, making decisions and generating responses.

3. Workflow Management: LangGraph excels at managing complex, multi-step AI workflows. This is crucial for autonomous agents that need to break down tasks into smaller steps and execute them in the right order.
...
6. Pay attention to how you structure the agent's decision-making process and workflow.
7. Don't forget to implement proper error handling and safety measures, especially if your agent will be interacting with external systems or making important decisions.

Building an autonomous agent is an iterative process, so be prepared to refine and improve your agent over time. Good luck with your project! If you need any more specific information as you progress, feel free to ask.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```
:::

:::js
```typescript
import { Command } from "@langchain/langgraph";
import { v4 as uuidv4 } from "uuid";

const threadConfig = { configurable: { thread_id: uuidv4() }, streamMode: "values" as const };

const inputs = [
  // 1st round of conversation
  {
    messages: [
      { role: "user", content: "i wanna go somewhere warm in the caribbean" }
    ]
  },
  // Since we're using `interrupt`, we'll need to resume using the Command primitive.
  // 2nd round of conversation
  new Command({
    resume: "could you recommend a nice hotel in one of the areas and tell me which area it is."
  }),
  // Third round of conversation
  new Command({ resume: "could you recommend something to do near the hotel?" }),
]

let iter = 0;
for await (const userInput of inputs) {
  iter += 1;
  console.log(`\n--- Conversation Turn ${iter} ---\n`);
  console.log(`User: ${JSON.stringify(userInput)}\n`);

  for await (const update of await graph.stream(userInput, threadConfig)) {
    const lastMessage = update.messages ? update.messages[update.messages.length - 1] : undefined;
    if (lastMessage && lastMessage._getType() === "ai") {
      console.log(`${lastMessage.name}: ${lastMessage.content}`)
    }
  }
}

```

```
--- Conversation Turn 1 ---

User: {"messages":[{"role":"user","content":"i wanna go somewhere warm in the caribbean"}]}

travelAdvisor: The Caribbean has so many wonderful warm destinations to choose from! Here are some fantastic options:

**Popular Caribbean Destinations:**
- **Barbados** - Beautiful beaches, friendly locals, and great weather year-round
- **Jamaica** - Vibrant culture, stunning beaches, and amazing food
- **Aruba** - Consistently sunny with gorgeous white sand beaches
- **Bahamas** - Crystal clear waters and easy access from the US
- **Costa Rica** - While technically Central America, offers Caribbean coastline with lush rainforests
- **Puerto Rico** - No passport needed for US citizens, rich history and culture
- **Dominican Republic** - Great value with beautiful resorts and beaches

Each destination offers something unique - some are better for nightlife and culture (like Jamaica), others for pure relaxation (like Aruba), and some for adventure activities (like Costa Rica).

What kind of experience are you looking for? Are you interested in:
- Pure beach relaxation
- Cultural experiences and local cuisine
- Adventure activities like snorkeling or hiking
- Vibrant nightlife
- Family-friendly activities

This will help me give you a more targeted recommendation!


--- Conversation Turn 2 ---

User: {"resume":"could you recommend a nice hotel in one of the areas and tell me which area it is."}

hotelAdvisor: I'd be happy to recommend a fantastic hotel! Let me suggest one of the Caribbean's most beautiful destinations:

**The Ocean Club, A Four Seasons Resort - Paradise Island, Bahamas**

**Location:** Paradise Island, Nassau, Bahamas

This is an absolutely stunning luxury resort that offers the perfect Caribbean experience. Here's why it's exceptional:

**Hotel Highlights:**
- Pristine white sand beaches with crystal-clear turquoise waters
- Beautifully landscaped gardens designed by the same architect who created Versailles
- Multiple pools including an adults-only pool
- World-class spa and fitness facilities
- Several excellent restaurants on-site
- Easy access to Atlantis Paradise Island next door for additional dining and entertainment

**The Area - Paradise Island:**
- Connected to Nassau by bridge, so you can easily explore the capital city
- Home to beautiful beaches like Cove Beach and Paradise Beach
- Close to great snorkeling and diving spots
- Duty-free shopping in Nassau
- Rich history and colonial architecture to explore
- Easy airport access (Nassau Airport is about 30 minutes away)

The Bahamas is perfect for a warm Caribbean getaway - it's consistently sunny, the water is incredibly beautiful, and there's a nice mix of relaxation and activities available. Plus, if you're coming from the US, no passport is required for citizens!

Would you like more details about this hotel or would you prefer recommendations for a different Caribbean destination?


--- Conversation Turn 3 ---

User: {"resume":"could you recommend something to do near the hotel?"}

sightseeingAdvisor: Excellent choice with The Ocean Club! Paradise Island and Nassau offer wonderful activities right near your hotel. Here are some fantastic things to do:

**On Paradise Island (walking distance/very close):**
- **Atlantis Paradise Island** - Right next door! Explore their massive aquariums, water slides, and marine exhibits even if you're not staying there
- **Paradise Beach** - One of the most beautiful beaches in the Caribbean, perfect for swimming and sunbathing
- **Cove Beach** - A more secluded, adults-oriented beach area
- **Golf at Ocean Club Golf Course** - Championship golf course designed by Tom Weiskopf

**Short trip to Nassau (15-20 minutes):**
- **Swimming with Dolphins** - Several locations offer this incredible experience
- **Straw Market** - Famous local market for souvenirs, crafts, and local goods
- **Queen's Staircase** - Historic 66-step staircase carved by slaves, beautiful and historically significant
- **Fort Charlotte** - Historic British colonial fort with great views and cannons
- **Junkanoo Beach** - Popular local beach with great vibes
- **Bay Street** - Main shopping and dining street with duty-free stores

**Water Activities Near the Hotel:**
- **Snorkeling and Diving** - Crystal clear waters with coral reefs nearby
- **Deep Sea Fishing** - Excellent fishing charters available
- **Boat Tours** - Island hopping, sunset cruises, or swimming with stingrays
- **Kayaking** - Explore the beautiful coastline

**Day Trips:**
- **Pig Beach** - Famous swimming with pigs experience (full day excursion)
- **Exuma Cays** - Stunning natural beauty and marine life

The great thing about this location is you have both luxury resort amenities and easy access to authentic Bahamian culture and adventures!
```
:::

## 3. Replay the full state history

Now that you have added steps to the chatbot, you can `replay` the full state history to see everything that occurred.

:::python
``` python
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state
```

```
Num Messages:  8 Next:  ()
--------------------------------------------------------------------------------
Num Messages:  7 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  6 Next:  ('tools',)
--------------------------------------------------------------------------------
Num Messages:  5 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  4 Next:  ('__start__',)
--------------------------------------------------------------------------------
Num Messages:  4 Next:  ()
--------------------------------------------------------------------------------
Num Messages:  3 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  2 Next:  ('tools',)
--------------------------------------------------------------------------------
Num Messages:  1 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  0 Next:  ('__start__',)
--------------------------------------------------------------------------------
```
:::

:::js
```typescript
let toReplay = null;
for await (const state of graph.getStateHistory(threadConfig)) {
    console.log(`Num Messages: ${state.values.messages.length}, Next: ${JSON.stringify(state.next)}`);
    console.log("-".repeat(80));
    if (state.values.messages.length === 6) {
        // We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        toReplay = state;
    }
}
```

```
Num Messages: 8, Next: []
--------------------------------------------------------------------------------
Num Messages: 7, Next: ["human"]
--------------------------------------------------------------------------------
Num Messages: 6, Next: ["sightseeingAdvisor"]
--------------------------------------------------------------------------------
Num Messages: 5, Next: ["human"]
--------------------------------------------------------------------------------
Num Messages: 4, Next: ["hotelAdvisor"]
--------------------------------------------------------------------------------
Num Messages: 3, Next: ["human"]
--------------------------------------------------------------------------------
Num Messages: 2, Next: ["travelAdvisor"]
--------------------------------------------------------------------------------
Num Messages: 1, Next: ["__start__"]
--------------------------------------------------------------------------------
```
:::

Checkpoints are saved for every step of the graph. This __spans invocations__ so you can rewind across a full thread's history.

## Resume from a checkpoint

:::python
Resume from the `to_replay` state, which is after the `chatbot` node in the second graph invocation. Resuming from this point will call the **action** node next.
:::

:::js
Resume from the `to_replay` state, which is after a specific node in one of the graph invocations. Resuming from this point will call the next scheduled node.
:::

:::python
```python
print(to_replay.next)
print(to_replay.config)
```

```
('tools',)
{'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1efd43e3-0c1f-6c4e-8006-891877d65740'}}
```
:::

:::js
```typescript
console.log(toReplay.next);
console.log(toReplay.config);
```

```
["sightseeingAdvisor"]
{
  configurable: {
    thread_id: "dcf6a444-3094-41c9-ba7d-9f4c7b99b5a0",
    checkpoint_ns: "",
    checkpoint_id: "1eff9b04-63e7-672c-8010-5fb09def85f8"
  }
}
```
:::

## 4. Load a state from a moment-in-time

The checkpoint's `to_replay.config` contains a `checkpoint_id` timestamp. Providing this `checkpoint_id` value tells LangGraph's checkpointer to **load** the state from that moment in time.

:::python
``` python
# The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================== Ai Message ==================================

[{'text': "That's an exciting idea! Building an autonomous agent with LangGraph is indeed a great application of this technology. LangGraph is particularly well-suited for creating complex, multi-step AI workflows, which is perfect for autonomous agents. Let me gather some more specific information about using LangGraph for building autonomous agents.", 'type': 'text'}, {'id': 'toolu_01QWNHhUaeeWcGXvA4eHT7Zo', 'input': {'query': 'Building autonomous agents with LangGraph examples and tutorials'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01QWNHhUaeeWcGXvA4eHT7Zo)
 Call ID: toolu_01QWNHhUaeeWcGXvA4eHT7Zo
  Args:
    query: Building autonomous agents with LangGraph examples and tutorials
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://towardsdatascience.com/building-autonomous-multi-tool-agents-with-gemini-2-0-and-langgraph-ad3d7bd5e79d", "content": "Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph | by Youness Mansar | Jan, 2025 | Towards Data Science Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph A practical tutorial with full code examples for building and running multi-tool agents Towards Data Science LLMs are remarkable — they can memorize vast amounts of information, answer general knowledge questions, write code, generate stories, and even fix your grammar. In this tutorial, we are going to build a simple LLM agent that is equipped with four tools that it can use to answer a user's question. This Agent will have the following specifications: Follow Published in Towards Data Science --------------------------------- Your home for data science and AI. Follow Follow Follow"}, {"url": "https://github.com/anmolaman20/Tools_and_Agents", "content": "GitHub - anmolaman20/Tools_and_Agents: This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository serves as a comprehensive guide for building AI-powered agents using Langchain and Langgraph. It provides hands-on examples, practical tutorials, and resources for developers and AI enthusiasts to master building intelligent systems and workflows. AI Agent Development: Gain insights into creating intelligent systems that think, reason, and adapt in real time. This repository is ideal for AI practitioners, developers exploring language models, or anyone interested in building intelligent systems. This repository provides resources for building AI agents using Langchain and Langgraph."}]
================================== Ai Message ==================================

Great idea! Building an autonomous agent with LangGraph is definitely an exciting project. Based on the latest information I've found, here are some insights and tips for building autonomous agents with LangGraph:

1. Multi-Tool Agents: LangGraph is particularly well-suited for creating autonomous agents that can use multiple tools. This allows your agent to have a diverse set of capabilities and choose the right tool for each task.

2. Integration with Large Language Models (LLMs): You can combine LangGraph with powerful LLMs like Gemini 2.0 to create more intelligent and capable agents. The LLM can serve as the "brain" of your agent, making decisions and generating responses.

3. Workflow Management: LangGraph excels at managing complex, multi-step AI workflows. This is crucial for autonomous agents that need to break down tasks into smaller steps and execute them in the right order.
...

Remember, building an autonomous agent is an iterative process. Start simple and gradually increase complexity as you become more comfortable with LangGraph and its capabilities.

Would you like more information on any specific aspect of building your autonomous agent with LangGraph?
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

The graph resumed execution from the `action` node. You can tell this is the case since the first value printed above is the response from our search engine tool.
:::

:::js
```typescript
// The `checkpoint_id` in the `toReplay.config` corresponds to a state we've persisted to our checkpointer.
for await (const event of await graph.stream(null, { ...toReplay.config, streamMode: "values" as const })) {
    const lastMessage = event.messages ? event.messages[event.messages.length - 1] : undefined;
    if (lastMessage && lastMessage._getType() === "ai") {
        console.log(`${lastMessage.name}: ${lastMessage.content}`)
    }
}
```

```
sightseeingAdvisor: Excellent choice with The Ocean Club! Paradise Island and Nassau offer wonderful activities right near your hotel. Here are some fantastic things to do:

**On Paradise Island (walking distance/very close):**
- **Atlantis Paradise Island** - Right next door! Explore their massive aquariums, water slides, and marine exhibits even if you're not staying there
- **Paradise Beach** - One of the most beautiful beaches in the Caribbean, perfect for swimming and sunbathing
- **Cove Beach** - A more secluded, adults-oriented beach area
- **Golf at Ocean Club Golf Course** - Championship golf course designed by Tom Weiskopf

**Short trip to Nassau (15-20 minutes):**
- **Swimming with Dolphins** - Several locations offer this incredible experience
- **Straw Market** - Famous local market for souvenirs, crafts, and local goods
- **Queen's Staircase** - Historic 66-step staircase carved by slaves, beautiful and historically significant
- **Fort Charlotte** - Historic British colonial fort with great views and cannons
- **Junkanoo Beach** - Popular local beach with great vibes
- **Bay Street** - Main shopping and dining street with duty-free stores

**Water Activities Near the Hotel:**
- **Snorkeling and Diving** - Crystal clear waters with coral reefs nearby
- **Deep Sea Fishing** - Excellent fishing charters available
- **Boat Tours** - Island hopping, sunset cruises, or swimming with stingrays
- **Kayaking** - Explore the beautiful coastline

**Day Trips:**
- **Pig Beach** - Famous swimming with pigs experience (full day excursion)
- **Exuma Cays** - Stunning natural beauty and marine life

The great thing about this location is you have both luxury resort amenities and easy access to authentic Bahamian culture and adventures!
```

The graph resumed execution from the `sightseeingAdvisor` node. You can tell this is the case since the first value printed above is the response from our sightseeing advisor agent.
:::

**Congratulations!** You've now used time-travel checkpoint traversal in LangGraph. Being able to rewind and explore alternative paths opens up a world of possibilities for debugging, experimentation, and interactive applications.

## Learn more

Take your LangGraph journey further by exploring deployment and advanced features:

- **[LangGraph Server quickstart](../../tutorials/langgraph-platform/local-server.md)**: Launch a LangGraph server locally and interact with it using the REST API and LangGraph Studio Web UI.
- **[LangGraph Platform quickstart](../../cloud/quick_start.md)**: Deploy your LangGraph app using LangGraph Platform.
- **[LangGraph Platform concepts](../../concepts/langgraph_platform.md)**: Understand the foundational concepts of the LangGraph Platform.