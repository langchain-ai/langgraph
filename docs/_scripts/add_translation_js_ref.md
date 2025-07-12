```typescript
// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__...";
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
// process.env.LANGCHAIN_TRACING_V2 = "true";
// process.env.LANGCHAIN_PROJECT = "Quickstart: LangGraphJS";
```

```typescript
// agent.ts

// IMPORTANT - Add your API keys here. Be careful not to publish them.
process.env.OPENAI_API_KEY = "sk-...";
process.env.TAVILY_API_KEY = "tvly-...";

import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// Define the tools for the agent to use
const agentTools = [new TavilySearch({ maxResults: 3 })];
const agentModel = new ChatOpenAI({ temperature: 0 });

// Initialize memory to persist state between graph runs
const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
  llm: agentModel,
  tools: agentTools,
  checkpointSaver: agentCheckpointer,
});

// Now it's time to use!
const agentFinalState = await agent.invoke(
  { messages: [new HumanMessage("what is the current weather in sf")] },
  { configurable: { thread_id: "42" } }
);

console.log(agentFinalState.messages.at(-1)?.text);

const agentNextState = await agent.invoke(
  { messages: [new HumanMessage("what about ny")] },
  { configurable: { thread_id: "42" } }
);

console.log(agentNextState.messages.at(-1)?.text);
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("chatbot-with-tools.png", imageBuffer);
```

```typescript
// agent.ts

// IMPORTANT - Add your API keys here. Be careful not to publish them.
process.env.OPENAI_API_KEY = "sk-...";
process.env.TAVILY_API_KEY = "tvly-...";

import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";

// Define the tools for the agent to use
const tools = [new TavilySearch({ maxResults: 3 })];
const toolNode = new ToolNode(tools);

// Create a model and give it access to the tools
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
}).bindTools(tools);

// Define the function that determines whether to continue or not
function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
  const lastMessage = messages.at(-1) as AIMessage | undefined;

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
}

// Define the function that calls the model
async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke(state.messages);

  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent") // __start__ is a special name for the entrypoint
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

// Finally, we compile it into a LangChain Runnable.
const app = workflow.compile();

// Use the agent
const finalState = await app.invoke({
  messages: [new HumanMessage("what is the weather in sf")],
});
console.log(finalState.messages.at(-1)?.text);

const nextState = await app.invoke({
  // Including the messages from the previous run gives the LLM context.
  // This way it knows we're asking about the weather in NY
  messages: [...finalState.messages, new HumanMessage("what about ny")],
});
console.log(nextState.messages.at(-1)?.text);
```

```typescript
import {
  StateGraph,
  Annotation,
  START,
  END,
  interrupt,
  MemorySaver,
} from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
  input: Annotation<string>,
  userFeedback: Annotation<string>,
});

const step1 = (_state: typeof StateAnnotation.State) => {
  console.log("---Step 1---");
  return {};
};

const humanFeedback = (_state: typeof StateAnnotation.State) => {
  console.log("--- humanFeedback ---");
  const feedback: string = interrupt("Please provide feedback");
  return {
    userFeedback: feedback,
  };
};

const step3 = (_state: typeof StateAnnotation.State) => {
  console.log("---Step 3---");
  return {};
};

const builder = new StateGraph(StateAnnotation)
  .addNode("step1", step1)
  .addNode("humanFeedback", humanFeedback)
  .addNode("step3", step3)
  .addEdge(START, "step1")
  .addEdge("step1", "humanFeedback")
  .addEdge("humanFeedback", "step3")
  .addEdge("step3", END);

// Set up memory
const memory = new MemorySaver();

// Add
const graph = builder.compile({
  checkpointer: memory,
});
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
// Input
const initialInput = { input: "hello world" };

// Thread
const config = { configurable: { thread_id: "1" } };

// Run the graph until the first interruption
for await (const event of await graph.stream(initialInput, config)) {
  console.log(event);
}

// Will log when the graph is interrupted, after step 2.
console.log("--- GRAPH INTERRUPTED ---");
```

```typescript
import { Command } from "@langchain/langgraph";

// Continue the graph execution
for await (const event of await graph.stream(
  new Command({ resume: "go to step 3! " }),
  config
)) {
  console.log(event);
  console.log("\n====\n");
}
```

```typescript
(await graph.getState(config)).values;
```

```typescript
// Set up the tool
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  MemorySaver,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { z } from "zod";

const search = tool(
  (_) => {
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈.";
  },
  {
    name: "search",
    description: "Call to surf the web.",
    schema: z.string(),
  }
);

const tools = [search];
const toolNode = new ToolNode<typeof MessagesAnnotation.State>(tools);

// Set up the model
const model = new ChatAnthropic({ model: "claude-3-5-sonnet-20240620" });

const askHumanTool = tool(
  (_) => {
    return "The human said XYZ";
  },
  {
    name: "askHuman",
    description: "Ask the human for input.",
    schema: z.string(),
  }
);

const modelWithTools = model.bindTools([...tools, askHumanTool]);

// Define nodes and conditional edges

// Define the function that determines whether to continue or not
function shouldContinue(
  state: typeof MessagesAnnotation.State
): "action" | "askHuman" | typeof END {
  const lastMessage = state.messages.at(-1) as AIMessage | undefined;
  // If there is no function call, then we finish
  if (lastMessage && !lastMessage.tool_calls?.length) {
    return END;
  }
  // If tool call is askHuman, we return that node
  // You could also add logic here to let some system know that there's something that requires Human input
  // For example, send a slack message, etc
  if (lastMessage.tool_calls?.[0]?.name === "askHuman") {
    console.log("--- ASKING HUMAN ---");
    return "askHuman";
  }
  // Otherwise if it isn't, we continue with the action node
  return "action";
}

// Define the function that calls the model
async function callModel(
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  const messages = state.messages;
  const response = await modelWithTools.invoke(messages);
  // We return an object with a messages property, because this will get added to the existing list
  return { messages: [response] };
}

// We define a fake node to ask the human
function askHuman(
  state: typeof MessagesAnnotation.State
): Partial<typeof MessagesAnnotation.State> {
  const lastMessage = state.messages.at(-1) as AIMessage | undefined;
  const toolCallId = lastMessage?.tool_calls?.[0].id;
  const location: string = interrupt("Please provide your location:");
  const newToolMessage = new ToolMessage({
    tool_call_id: toolCallId!,
    content: location,
  });
  return { messages: [newToolMessage] };
}

// Define a new graph
const messagesWorkflow = new StateGraph(MessagesAnnotation)
  // Define the two nodes we will cycle between
  .addNode("agent", callModel)
  .addNode("action", toolNode)
  .addNode("askHuman", askHuman)
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `agent`.
    // This means these are the edges taken after the `agent` node is called.
    "agent",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue
  )
  // We now add a normal edge from `action` to `agent`.
  // This means that after `action` is called, `agent` node is called next.
  .addEdge("action", "agent")
  // After we get back the human response, we go back to the agent
  .addEdge("askHuman", "agent")
  // Set the entrypoint as `agent`
  // This means that this node is the first one called
  .addEdge(START, "agent");

// Setup memory
const messagesMemory = new MemorySaver();

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const messagesApp = messagesWorkflow.compile({
  checkpointer: messagesMemory,
});
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await messagesApp.getGraphAsync();
const image = await drawableGraph2.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
// Input
const input = {
  role: "user",
  content:
    "Use the search tool to ask the user where they are, then look up the weather there",
};

// Thread
const config2 = {
  configurable: { thread_id: "3" },
  streamMode: "values" as const,
};

for await (const event of await messagesApp.stream(
  { messages: [input] },
  config2
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
console.log("next: ", (await messagesApp.getState(config2)).next);
```

```typescript
import { Command } from "@langchain/langgraph";

// Continue the graph execution
for await (const event of await messagesApp.stream(
  new Command({ resume: "San Francisco" }),
  config2
)) {
  console.log(event);
  console.log("\n====\n");
}
```

```typescript
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Tool for getting travel recommendations
const getTravelRecommendations = tool(
  async () => {
    const destinations = ["aruba", "turks and caicos"];
    return destinations[Math.floor(Math.random() * destinations.length)];
  },
  {
    name: "getTravelRecommendations",
    description: "Get recommendation for travel destinations",
    schema: z.object({}),
  }
);

// Tool for getting hotel recommendations
const getHotelRecommendations = tool(
  async (input: { location: "aruba" | "turks and caicos" }) => {
    const recommendations = {
      aruba: [
        "The Ritz-Carlton, Aruba (Palm Beach)",
        "Bucuti & Tara Beach Resort (Eagle Beach)",
      ],
      "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
    };
    return recommendations[input.location];
  },
  {
    name: "getHotelRecommendations",
    description: "Get hotel recommendations for a given destination.",
    schema: z.object({
      location: z.enum(["aruba", "turks and caicos"]),
    }),
  }
);

// Define a tool to signal intent to hand off to a different agent
// Note: this is not using Command(goto) syntax for navigating to different agents:
// `workflow()` below handles the handoffs explicitly
const transferToHotelAdvisor = tool(
  async () => {
    return "Successfully transferred to hotel advisor";
  },
  {
    name: "transferToHotelAdvisor",
    description: "Ask hotel advisor agent for help.",
    schema: z.object({}),
    // Hint to our agent implementation that it should stop
    // immediately after invoking this tool
    returnDirect: true,
  }
);

const transferToTravelAdvisor = tool(
  async () => {
    return "Successfully transferred to travel advisor";
  },
  {
    name: "transferToTravelAdvisor",
    description: "Ask travel advisor agent for help.",
    schema: z.object({}),
    // Hint to our agent implementation that it should stop
    // immediately after invoking this tool
    returnDirect: true,
  }
);
```

```typescript
import {
  AIMessage,
  type BaseMessage,
  type BaseMessageLike,
} from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {
  addMessages,
  entrypoint,
  task,
  MemorySaver,
  interrupt,
} from "@langchain/langgraph";

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
});

const travelAdvisorTools = [getTravelRecommendations, transferToHotelAdvisor];

// Define travel advisor ReAct agent
const travelAdvisor = createReactAgent({
  llm: model,
  tools: travelAdvisorTools,
  stateModifier: [
    "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc).",
    "If you need hotel recommendations, ask 'hotel_advisor' for help.",
    "You MUST include human-readable response before transferring to another agent.",
  ].join(" "),
});

// You can also add additional logic like changing the input to the agent / output from the agent, etc.
// NOTE: we're invoking the ReAct agent with the full history of messages in the state
const callTravelAdvisor = task(
  "callTravelAdvisor",
  async (messages: BaseMessageLike[]) => {
    const response = await travelAdvisor.invoke({ messages });
    return response.messages;
  }
);

const hotelAdvisorTools = [getHotelRecommendations, transferToTravelAdvisor];

// Define hotel advisor ReAct agent
const hotelAdvisor = createReactAgent({
  llm: model,
  tools: hotelAdvisorTools,
  stateModifier: [
    "You are a hotel expert that can provide hotel recommendations for a given destination.",
    "If you need help picking travel destinations, ask 'travel_advisor' for help.",
    "You MUST include a human-readable response before transferring to another agent.",
  ].join(" "),
});

// Add task for hotel advisor
const callHotelAdvisor = task(
  "callHotelAdvisor",
  async (messages: BaseMessageLike[]) => {
    const response = await hotelAdvisor.invoke({ messages });
    return response.messages;
  }
);

const checkpointer = new MemorySaver();

const multiTurnGraph = entrypoint(
  {
    name: "multiTurnGraph",
    checkpointer,
  },
  async (messages: BaseMessageLike[]) => {
    let callActiveAgent = callTravelAdvisor;
    let agentMessages: BaseMessage[];
    let currentMessages = messages;
    while (true) {
      agentMessages = await callActiveAgent(currentMessages);

      // Find the last AI message
      // If one of the handoff tools is called, the last message returned
      // by the agent will be a ToolMessages because we set them to have
      // "returnDirect: true". This means that the last AIMessage will
      // have tool calls.
      // Otherwise, the last returned message will be an AIMessage with
      // no tool calls, which means we are ready for new input.
      const reversedMessages = [...agentMessages].reverse();
      const aiMsgIndex = reversedMessages.findIndex(
        (m): m is AIMessage => m.getType() === "ai"
      );

      const aiMsg: AIMessage = reversedMessages[aiMsgIndex];

      // We append all messages up to the last AI message to the current messages.
      // This may include ToolMessages (if the handoff tool was called)
      const messagesToAdd = reversedMessages.slice(0, aiMsgIndex + 1).reverse();

      // Add the agent's responses
      currentMessages = addMessages(currentMessages, messagesToAdd);

      if (!aiMsg?.tool_calls?.length) {
        const userInput = await interrupt("Ready for user input.");
        if (typeof userInput !== "string") {
          throw new Error("User input must be a string.");
        }
        if (userInput.toLowerCase() === "done") {
          break;
        }
        currentMessages = addMessages(currentMessages, [
          {
            role: "human",
            content: userInput,
          },
        ]);
        continue;
      }

      const toolCall = aiMsg.tool_calls.at(-1)!;
      if (toolCall.name === "transferToHotelAdvisor") {
        callActiveAgent = callHotelAdvisor;
      } else if (toolCall.name === "transferToTravelAdvisor") {
        callActiveAgent = callTravelAdvisor;
      } else {
        throw new Error(`Expected transfer tool, got '${toolCall.name}'`);
      }
    }

    return entrypoint.final({
      value: agentMessages.at(-1),
      save: currentMessages,
    });
  }
);
```

```typescript
import { v4 as uuidv4 } from "uuid";
import { Command } from "@langchain/langgraph";
import { isBaseMessage } from "@langchain/core/messages";

const threadConfig = {
  configurable: {
    thread_id: uuidv4(),
  },
  streamMode: "updates" as const,
};

const inputs = [
  // 1st round of conversation
  [{ role: "user", content: "i wanna go somewhere warm in the caribbean" }],
  // Since we're using `interrupt`, we'll need to resume using the Command primitive
  // 2nd round of conversation
  new Command({
    resume:
      "could you recommend a nice hotel in one of the areas and tell me which area it is.",
  }),
  // 3rd round of conversation
  new Command({
    resume:
      "i like the first one. could you recommend something to do near the hotel?",
  }),
];

const runConversation = async () => {
  for (const [idx, userInput] of inputs.entries()) {
    console.log();
    console.log(`--- Conversation Turn ${idx + 1} ---`);
    console.log();
    console.log(`User: ${JSON.stringify(userInput, null, 2)}`);
    console.log();

    const stream = await multiTurnGraph.stream(userInput as any, threadConfig);

    for await (const update of stream) {
      if (update.__metadata__?.cached) {
        continue;
      }
      for (const [nodeId, value] of Object.entries(update)) {
        if (Array.isArray(value) && value.length > 0) {
          const lastMessage = value.at(-1);
          if (isBaseMessage(lastMessage) && lastMessage?.getType() === "ai") {
            console.log(`${nodeId}: ${lastMessage.content}`);
          }
        }
      }
    }
  }
};

// Execute the conversation
try {
  await runConversation();
} catch (e) {
  console.error(e);
}
```

```typescript
process.env.OPENAI_API_KEY = "YOUR_API_KEY";
```

```typescript
import { task, interrupt } from "@langchain/langgraph";

const step1 = task("step1", async (inputQuery: string) => {
  return `${inputQuery} bar`;
});

const humanFeedback = task("humanFeedback", async (inputQuery: string) => {
  const feedback = interrupt(`Please provide feedback: ${inputQuery}`);
  return `${inputQuery} ${feedback}`;
});

const step3 = task("step3", async (inputQuery: string) => {
  return `${inputQuery} qux`;
});
```

```typescript
import { MemorySaver, entrypoint } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const graph = entrypoint(
  {
    name: "graph",
    checkpointer,
  },
  async (inputQuery: string) => {
    const result1 = await step1(inputQuery);
    const result2 = await humanFeedback(result1);
    const result3 = await step3(result2);
    return result3;
  }
);
```

```typescript
const config = {
  configurable: {
    thread_id: "1",
  },
};

const stream = await graph.stream("foo", config);

for await (const event of stream) {
  console.log(event);
}
```

```typescript
import { Command } from "@langchain/langgraph";

const resumeStream = await graph.stream(
  new Command({
    resume: "baz",
  }),
  config
);

// Continue execution
for await (const event of resumeStream) {
  if (event.__metadata__?.cached) {
    continue;
  }
  console.log(event);
}
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const getWeather = tool(
  async ({ location }) => {
    // This is a placeholder for the actual implementation
    const lowercaseLocation = location.toLowerCase();
    if (
      lowercaseLocation.includes("sf") ||
      lowercaseLocation.includes("san francisco")
    ) {
      return "It's sunny!";
    } else if (lowercaseLocation.includes("boston")) {
      return "It's rainy!";
    } else {
      return `I am not sure what the weather is in ${location}`;
    }
  },
  {
    name: "getWeather",
    schema: z.object({
      location: z.string().describe("Location to get the weather for"),
    }),
    description: "Call to get the weather from a specific location.",
  }
);
```

```typescript
import { interrupt } from "@langchain/langgraph";
import { z } from "zod";

const humanAssistance = tool(
  async ({ query }) => {
    const humanResponse = interrupt({ query });
    return humanResponse.data;
  },
  {
    name: "humanAssistance",
    description: "Request assistance from a human.",
    schema: z.object({
      query: z.string().describe("Human readable question for the human"),
    }),
  }
);

const tools = [getWeather, humanAssistance];
```

```typescript
import {
  type BaseMessageLike,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { type ToolCall } from "@langchain/core/messages/tool";
import { task } from "@langchain/langgraph";

const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));

const callModel = task("callModel", async (messages: BaseMessageLike[]) => {
  const response = await model.bindTools(tools).invoke(messages);
  return response;
});

const callTool = task(
  "callTool",
  async (toolCall: ToolCall): Promise<AIMessage> => {
    const tool = toolsByName[toolCall.name];
    const observation = await tool.invoke(toolCall.args);
    return new ToolMessage({ content: observation, tool_call_id: toolCall.id });
    // Can also pass toolCall directly into the tool to return a ToolMessage
    // return tool.invoke(toolCall);
  }
);
```

```typescript
import { entrypoint, addMessages, MemorySaver } from "@langchain/langgraph";

const agent = entrypoint(
  {
    name: "agent",
    checkpointer: new MemorySaver(),
  },
  async (messages: BaseMessageLike[]) => {
    let currentMessages = messages;
    let llmResponse = await callModel(currentMessages);
    while (true) {
      if (!llmResponse.tool_calls?.length) {
        break;
      }

      // Execute tools
      const toolResults = await Promise.all(
        llmResponse.tool_calls.map((toolCall) => {
          return callTool(toolCall);
        })
      );

      // Append to message list
      currentMessages = addMessages(currentMessages, [
        llmResponse,
        ...toolResults,
      ]);

      // Call model again
      llmResponse = await callModel(currentMessages);
    }

    return llmResponse;
  }
);
```

```typescript
import { BaseMessage, isAIMessage } from "@langchain/core/messages";

const prettyPrintMessage = (message: BaseMessage) => {
  console.log("=".repeat(30), `${message.getType()} message`, "=".repeat(30));
  console.log(message.content);
  if (isAIMessage(message) && message.tool_calls?.length) {
    console.log(JSON.stringify(message.tool_calls, null, 2));
  }
};

const prettyPrintStep = (step: Record<string, any>) => {
  if (step.__metadata__?.cached) {
    return;
  }
  for (const [taskName, update] of Object.entries(step)) {
    const message = update as BaseMessage;
    // Only print task updates
    if (taskName === "agent") continue;
    console.log(`\n${taskName}:`);
    if (taskName === "__interrupt__") {
      console.log(update);
    } else {
      prettyPrintMessage(message);
    }
  }
};
```

```typescript
const userMessage = {
  role: "user",
  content: [
    "Can you reach out for human assistance: what should I feed my cat?",
    "Separately, can you check the weather in San Francisco?",
  ].join(" "),
};
console.log(userMessage);

const agentStream = await agent.stream([userMessage], {
  configurable: {
    thread_id: "1",
  },
});

let lastStep;

for await (const step of agentStream) {
  prettyPrintStep(step);
  lastStep = step;
}
```

```typescript
console.log(JSON.stringify(lastStep));
```

```typescript
import { Command } from "@langchain/langgraph";

const humanResponse = "You should feed your cat a fish.";
const humanCommand = new Command({
  resume: { data: humanResponse },
});

const resumeStream2 = await agent.stream(humanCommand, config);

for await (const step of resumeStream2) {
  prettyPrintStep(step);
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk-...";
```

```typescript
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});
```

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const searchTool = tool(
  async ({ query: _query }: { query: string }) => {
    // This is a placeholder for the actual implementation
    return "Cold, with a low of 3℃";
  },
  {
    name: "search",
    description:
      "Use to surf the web, fetch current information, check the weather, and retrieve other information.",
    schema: z.object({
      query: z.string().describe("The query to use in your search."),
    }),
  }
);

await searchTool.invoke({ query: "What's the weather like?" });

const tools = [searchTool];
```

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

const toolNode = new ToolNode(tools);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({ model: "gpt-4o" });
```

```typescript
const boundModel = model.bindTools(tools);
```

```typescript
import { END, START, StateGraph } from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";

const routeMessage = (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  // If no tools are called, we can finish (respond to the user)
  if (!lastMessage?.tool_calls?.length) {
    return END;
  }
  // Otherwise if there is, we continue and call the tools
  return "tools";
};

const callModel = async (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const responseMessage = await boundModel.invoke(messages);
  return { messages: [responseMessage] };
};

const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", routeMessage)
  .addEdge("tools", "agent");

const graph = workflow.compile();
```

```typescript
let inputs = {
  messages: [{ role: "user", content: "what's the weather in sf" }],
};

for await (const chunk of await graph.stream(inputs, {
  streamMode: "updates",
})) {
  for (const [node, values] of Object.entries(chunk)) {
    console.log(`Receiving update from node: ${node}`);
    console.log(values);
    console.log("\n====\n");
  }
}
```

```typescript
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Tool for getting travel recommendations
const getTravelRecommendations = tool(
  async () => {
    const destinations = ["aruba", "turks and caicos"];
    return destinations[Math.floor(Math.random() * destinations.length)];
  },
  {
    name: "getTravelRecommendations",
    description: "Get recommendation for travel destinations",
    schema: z.object({}),
  }
);

// Tool for getting hotel recommendations
const getHotelRecommendations = tool(
  async (input: { location: "aruba" | "turks and caicos" }) => {
    const recommendations = {
      aruba: [
        "The Ritz-Carlton, Aruba (Palm Beach)",
        "Bucuti & Tara Beach Resort (Eagle Beach)",
      ],
      "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
    };
    return recommendations[input.location];
  },
  {
    name: "getHotelRecommendations",
    description: "Get hotel recommendations for a given destination.",
    schema: z.object({
      location: z.enum(["aruba", "turks and caicos"]),
    }),
  }
);

// Define a tool to signal intent to hand off to a different agent
// Note: this is not using Command(goto) syntax for navigating to different agents:
// `workflow()` below handles the handoffs explicitly
const transferToHotelAdvisor = tool(
  async () => {
    return "Successfully transferred to hotel advisor";
  },
  {
    name: "transferToHotelAdvisor",
    description: "Ask hotel advisor agent for help.",
    schema: z.object({}),
    // Hint to our agent implementation that it should stop
    // immediately after invoking this tool
    returnDirect: true,
  }
);

const transferToTravelAdvisor = tool(
  async () => {
    return "Successfully transferred to travel advisor";
  },
  {
    name: "transferToTravelAdvisor",
    description: "Ask travel advisor agent for help.",
    schema: z.object({}),
    // Hint to our agent implementation that it should stop
    // immediately after invoking this tool
    returnDirect: true,
  }
);
```

```typescript
import { AIMessage, type BaseMessageLike } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { addMessages, entrypoint, task } from "@langchain/langgraph";

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
});

const travelAdvisorTools = [getTravelRecommendations, transferToHotelAdvisor];

// Define travel advisor ReAct agent
const travelAdvisor = createReactAgent({
  llm: model,
  tools: travelAdvisorTools,
  stateModifier: [
    "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc).",
    "If you need hotel recommendations, ask 'hotel_advisor' for help.",
    "You MUST include human-readable response before transferring to another agent.",
  ].join(" "),
});

// You can also add additional logic like changing the input to the agent / output from the agent, etc.
// NOTE: we're invoking the ReAct agent with the full history of messages in the state
const callTravelAdvisor = task(
  "callTravelAdvisor",
  async (messages: BaseMessageLike[]) => {
    const response = await travelAdvisor.invoke({ messages });
    return response.messages;
  }
);

const hotelAdvisorTools = [getHotelRecommendations, transferToTravelAdvisor];

// Define hotel advisor ReAct agent
const hotelAdvisor = createReactAgent({
  llm: model,
  tools: hotelAdvisorTools,
  stateModifier: [
    "You are a hotel expert that can provide hotel recommendations for a given destination.",
    "If you need help picking travel destinations, ask 'travel_advisor' for help.",
    "You MUST include a human-readable response before transferring to another agent.",
  ].join(" "),
});

// Add task for hotel advisor
const callHotelAdvisor = task(
  "callHotelAdvisor",
  async (messages: BaseMessageLike[]) => {
    const response = await hotelAdvisor.invoke({ messages });
    return response.messages;
  }
);

const networkGraph = entrypoint(
  "networkGraph",
  async (messages: BaseMessageLike[]) => {
    // Converts inputs to LangChain messages as a side-effect
    let currentMessages = addMessages([], messages);

    let callActiveAgent = callTravelAdvisor;
    while (true) {
      const agentMessages = await callActiveAgent(currentMessages);
      currentMessages = addMessages(currentMessages, agentMessages);

      // Find the last AI message
      // If one of the handoff tools is called, the last message returned
      // by the agent will be a ToolMessage because we set them to have
      // "returnDirect: true". This means that the last AIMessage will
      // have tool calls.
      // Otherwise, the last returned message will be an AIMessage with
      // no tool calls, which means we are ready for new input.
      const aiMsg = [...agentMessages]
        .reverse()
        .find((m): m is AIMessage => m.getType() === "ai");

      // If no tool calls, we're done
      if (!aiMsg?.tool_calls?.length) {
        break;
      }

      // Get the last tool call and determine next agent
      const toolCall = aiMsg.tool_calls.at(-1)!;
      if (toolCall.name === "transferToTravelAdvisor") {
        callActiveAgent = callTravelAdvisor;
      } else if (toolCall.name === "transferToHotelAdvisor") {
        callActiveAgent = callHotelAdvisor;
      } else {
        throw new Error(`Expected transfer tool, got '${toolCall.name}'`);
      }
    }

    return messages;
  }
);
```

```typescript
const prettyPrintMessages = (update: Record<string, any>) => {
  // Handle tuple case with namespace
  if (Array.isArray(update)) {
    const [ns, updateData] = update;
    // Skip parent graph updates in the printouts
    if (ns.length === 0) {
      return;
    }

    const graphId = ns[ns.length - 1].split(":")[0];
    console.log(`Update from subgraph ${graphId}:\n`);
    update = updateData;
  }

  if (update.__metadata__?.cached) {
    return;
  }
  // Print updates for each node
  for (const [nodeName, updateValue] of Object.entries(update)) {
    console.log(`Update from node ${nodeName}:\n`);

    const coercedMessages = addMessages([], updateValue.messages);
    for (const message of coercedMessages) {
      const textContent =
        typeof message.content === "string"
          ? message.content
          : JSON.stringify(message.content);
      // Print message content based on role
      if (message.getType() === "ai") {
        console.log("=".repeat(33) + " Assistant Message " + "=".repeat(33));
        console.log(textContent);
        console.log();
      } else if (message.getType() === "human") {
        console.log("=".repeat(33) + " Human Message " + "=".repeat(33));
        console.log(textContent);
        console.log();
      } else if (message.getType() === "tool") {
        console.log("=".repeat(33) + " Tool Message " + "=".repeat(33));
        console.log(textContent);
        console.log();
      }
    }
    console.log("\n");
  }
};
```

```typescript
const stream = await networkGraph.stream(
  [
    {
      role: "user",
      content:
        "i wanna go somewhere warm in the caribbean. pick one destination and give me hotel recommendations",
    },
  ],
  { subgraphs: true }
);

for await (const chunk of stream) {
  prettyPrintMessages(chunk);
}
```

```typescript
import { Annotation, Command } from "@langchain/langgraph";

// Define graph state
const StateAnnotation = Annotation.Root({
  foo: Annotation<string>,
});

// Define the nodes
const nodeA = async (_state: typeof StateAnnotation.State) => {
  console.log("Called A");
  // this is a replacement for a real conditional edge function
  const goto = Math.random() > 0.5 ? "nodeB" : "nodeC";
  // note how Command allows you to BOTH update the graph state AND route to the next node
  return new Command({
    // this is the state update
    update: {
      foo: "a",
    },
    // this is a replacement for an edge
    goto,
  });
};

// Nodes B and C are unchanged
const nodeB = async (state: typeof StateAnnotation.State) => {
  console.log("Called B");
  return {
    foo: state.foo + "|b",
  };
};

const nodeC = async (state: typeof StateAnnotation.State) => {
  console.log("Called C");
  return {
    foo: state.foo + "|c",
  };
};
```

```typescript
import { StateGraph } from "@langchain/langgraph";

// NOTE: there are no edges between nodes A, B and C!
const graph = new StateGraph(StateAnnotation)
  .addNode("nodeA", nodeA, {
    ends: ["nodeB", "nodeC"],
  })
  .addNode("nodeB", nodeB)
  .addNode("nodeC", nodeC)
  .addEdge("__start__", "nodeA")
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
await graph.invoke({ foo: "" });
```

```typescript
// Define the nodes
const nodeASubgraph = async (_state: typeof StateAnnotation.State) => {
  console.log("Called A");
  // this is a replacement for a real conditional edge function
  const goto = Math.random() > 0.5 ? "nodeB" : "nodeC";
  // note how Command allows you to BOTH update the graph state AND route to the next node
  return new Command({
    update: {
      foo: "a",
    },
    goto,
    // this tells LangGraph to navigate to node_b or node_c in the parent graph
    // NOTE: this will navigate to the closest parent graph relative to the subgraph
    graph: Command.PARENT,
  });
};

const subgraph = new StateGraph(StateAnnotation)
  .addNode("nodeA", nodeASubgraph)
  .addEdge("__start__", "nodeA")
  .compile();

const parentGraph = new StateGraph(StateAnnotation)
  .addNode("subgraph", subgraph, { ends: ["nodeB", "nodeC"] })
  .addNode("nodeB", nodeB)
  .addNode("nodeC", nodeC)
  .addEdge("__start__", "subgraph")
  .compile();

await parentGraph.invoke({ foo: "" });
```

```typescript
import { StateGraph, START, Annotation } from "@langchain/langgraph";

const GrandChildAnnotation = Annotation.Root({
  myGrandchildKey: Annotation<string>,
});

const grandchild1 = (state: typeof GrandChildAnnotation.State) => {
  // NOTE: child or parent keys will not be accessible here
  return {
    myGrandchildKey: state.myGrandchildKey + ", how are you",
  };
};

const grandchild = new StateGraph(GrandChildAnnotation)
  .addNode("grandchild1", grandchild1)
  .addEdge(START, "grandchild1");

const grandchildGraph = grandchild.compile();
```

```typescript
await grandchildGraph.invoke({ myGrandchildKey: "hi Bob" });
```

```typescript
import { StateGraph, START, Annotation } from "@langchain/langgraph";

const ChildAnnotation = Annotation.Root({
  myChildKey: Annotation<string>,
});

const callGrandchildGraph = async (state: typeof ChildAnnotation.State) => {
  // NOTE: parent or grandchild keys won't be accessible here
  // we're transforming the state from the child state channels (`myChildKey`)
  // to the grandchild state channels (`myGrandchildKey`)
  const grandchildGraphInput = { myGrandchildKey: state.myChildKey };
  // we're transforming the state from the grandchild state channels (`myGrandchildKey`)
  // back to the child state channels (`myChildKey`)
  const grandchildGraphOutput = await grandchildGraph.invoke(
    grandchildGraphInput
  );
  return {
    myChildKey: grandchildGraphOutput.myGrandchildKey + " today?",
  };
};

const child = new StateGraph(ChildAnnotation)
  // NOTE: we're passing a function here instead of just compiled graph (`childGraph`)
  .addNode("child1", callGrandchildGraph)
  .addEdge(START, "child1");

const childGraph = child.compile();
```

```typescript
await childGraph.invoke({ myChildKey: "hi Bob" });
```

```typescript
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";

const ParentAnnotation = Annotation.Root({
  myKey: Annotation<string>,
});

const parent1 = (state: typeof ParentAnnotation.State) => {
  // NOTE: child or grandchild keys won't be accessible here
  return { myKey: "hi " + state.myKey };
};

const parent2 = (state: typeof ParentAnnotation.State) => {
  return { myKey: state.myKey + " bye!" };
};

const callChildGraph = async (state: typeof ParentAnnotation.State) => {
  // we're transforming the state from the parent state channels (`myKey`)
  // to the child state channels (`myChildKey`)
  const childGraphInput = { myChildKey: state.myKey };
  // we're transforming the state from the child state channels (`myChildKey`)
  // back to the parent state channels (`myKey`)
  const childGraphOutput = await childGraph.invoke(childGraphInput);
  return { myKey: childGraphOutput.myChildKey };
};

const parent = new StateGraph(ParentAnnotation)
  .addNode("parent1", parent1)
  // NOTE: we're passing a function here instead of just a compiled graph (`childGraph`)
  .addNode("child", callChildGraph)
  .addNode("parent2", parent2)
  .addEdge(START, "parent1")
  .addEdge("parent1", "child")
  .addEdge("child", "parent2")
  .addEdge("parent2", END);

const parentGraph = parent.compile();
```

```typescript
await parentGraph.invoke({ myKey: "Bob" });
```

```typescript
import { z } from "zod";
import { tool } from "@langchain/core/tools";

const getWeather = tool(
  async (input: { city: "sf" | "nyc" }) => {
    if (input.city === "nyc") {
      return "It might be cloudy in nyc";
    } else if (input.city === "sf") {
      return "It's always sunny in sf";
    } else {
      throw new Error("Unknown city");
    }
  },
  {
    name: "get_weather",
    description: "Use this to get weather information.",
    schema: z.object({
      city: z.enum(["sf", "nyc"]),
    }),
  }
);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

import pg from "pg";

const { Pool } = pg;

const pool = new Pool({
  connectionString: "postgresql://user:password@localhost:5434/testdb",
});

const checkpointer = new PostgresSaver(pool);

// NOTE: you need to call .setup() the first time you're using your checkpointer

await checkpointer.setup();

const graph = createReactAgent({
  tools: [getWeather],
  llm: new ChatOpenAI({
    model: "gpt-4o-mini",
  }),
  checkpointSaver: checkpointer,
});
const config = { configurable: { thread_id: "1" } };

await graph.invoke(
  {
    messages: [
      {
        role: "user",
        content: "what's the weather in sf",
      },
    ],
  },
  config
);
```

```typescript
await checkpointer.get(config);
```

```typescript
const checkpointerFromConnString = PostgresSaver.fromConnString(
  "postgresql://user:password@localhost:5434/testdb"
);

const graph2 = createReactAgent({
  tools: [getWeather],
  llm: new ChatOpenAI({
    model: "gpt-4o-mini",
  }),
  checkpointSaver: checkpointerFromConnString,
});
const config2 = { configurable: { thread_id: "2" } };

await graph2.invoke(
  {
    messages: [
      {
        role: "user",
        content: "what's the weather in sf",
      },
    ],
  },
  config2
);
```

```typescript
await checkpointerFromConnString.get(config2);
```

```typescript
const graph = graphBuilder.compile({
    interruptBefore: ["nodeA"],
    interruptAfter: ["nodeB", "nodeC"],
    checkpointer: ..., // Specify a checkpointer
});

const threadConfig = {
    configurable: {
        thread_id: "someThread"
    }
};

// Run the graph until the breakpoint
await graph.invoke(inputs, threadConfig);

// Optionally update the graph state based on user input
await graph.updateState(update, threadConfig);

// Resume the graph
await graph.invoke(null, threadConfig);
```

```typescript
await graph.invoke(inputs, {
  configurable: { thread_id: "someThread" },
  interruptBefore: ["nodeA"],
  interruptAfter: ["nodeB", "nodeC"],
});

const threadConfig = {
  configurable: {
    thread_id: "someThread",
  },
};

// Run the graph until the breakpoint
await graph.invoke(inputs, threadConfig);

// Optionally update the graph state based on user input
await graph.updateState(update, threadConfig);

// Resume the graph
await graph.invoke(null, threadConfig);
```

```typescript
function myNode(state: typeof GraphAnnotation.State) {
  if (state.input.length > 5) {
    throw new NodeInterrupt(
      `Received input that is longer than 5 characters: ${state.input}`
    );
  }
  return state;
}
```

```typescript
// Attempt to continue the graph execution with no change to state after we hit the dynamic breakpoint
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
// Update the state to pass the dynamic breakpoint
await graph.updateState({ input: "foo" }, threadConfig);

for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
// This update will skip the node `myNode` altogether
await graph.updateState(null, threadConfig, "myNode");

for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
import { StateGraph, Annotation } from "@langchain/langgraph";

// subgraph

const SubgraphStateAnnotation = Annotation.Root({
  foo: Annotation<string>,
  bar: Annotation<string>,
});

const subgraphNode1 = async (state: typeof SubgraphStateAnnotation.State) => {
  return { bar: "bar" };
};

const subgraphNode2 = async (state: typeof SubgraphStateAnnotation.State) => {
  // note that this node is using a state key ('bar') that is only available in the subgraph
  // and is sending update on the shared state key ('foo')
  return { foo: state.foo + state.bar };
};

const subgraph = new StateGraph(SubgraphStateAnnotation)
  .addNode("subgraphNode1", subgraphNode1)
  .addNode("subgraphNode2", subgraphNode2)
  .addEdge("__start__", "subgraphNode1")
  .addEdge("subgraphNode1", "subgraphNode2")
  .compile();

// parent graph
const StateAnnotation = Annotation.Root({
  foo: Annotation<string>,
});

const node1 = async (state: typeof StateAnnotation.State) => {
  return {
    foo: "hi! " + state.foo,
  };
};

const builder = new StateGraph(StateAnnotation)
  .addNode("node1", node1)
  // note that we're adding the compiled subgraph as a node to the parent graph
  .addNode("node2", subgraph)
  .addEdge("__start__", "node1")
  .addEdge("node1", "node2");
```

```typescript
import { MemorySaver } from "@langchain/langgraph-checkpoint";

const checkpointer = new MemorySaver();

// You must only pass checkpointer when compiling the parent graph.
// LangGraph will automatically propagate the checkpointer to the child subgraphs.

const graph = builder.compile({
  checkpointer: checkpointer,
});
```

```typescript
const config = { configurable: { thread_id: "1" } };
```

```typescript
const stream = await graph.stream(
  {
    foo: "foo",
  },
  {
    ...config,
    subgraphs: true,
  }
);

for await (const [_source, chunk] of stream) {
  console.log(chunk);
}
```

```typescript
(await graph.getState(config)).values;
```

```typescript
let stateWithSubgraph;

const graphHistories = await graph.getStateHistory(config);

for await (const state of graphHistories) {
  if (state.next[0] === "node2") {
    stateWithSubgraph = state;
    break;
  }
}
```

```typescript
const subgraphConfig = stateWithSubgraph.tasks[0].state;

console.log(subgraphConfig);
```

```typescript
(await graph.getState(subgraphConfig)).values;
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__..."
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "ReAct Agent with memory: LangGraphJS";
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";

const model = new ChatOpenAI({
  model: "gpt-4o",
});

const getWeather = tool(
  (input) => {
    if (input.location === "sf") {
      return "It's always sunny in sf";
    } else {
      return "It might be cloudy in nyc";
    }
  },
  {
    name: "get_weather",
    description: "Call to get the current weather.",
    schema: z.object({
      location: z
        .enum(["sf", "nyc"])
        .describe("Location to get the weather for."),
    }),
  }
);

// Here we only save in-memory
const memory = new MemorySaver();

const agent = createReactAgent({
  llm: model,
  tools: [getWeather],
  checkpointSaver: memory,
});
```

```typescript
let inputs = {
  messages: [{ role: "user", content: "what is the weather in NYC?" }],
};
let config = { configurable: { thread_id: "1" } };
let stream = await agent.stream(inputs, {
  ...config,
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  } else if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  } else {
    console.log(msg);
  }
  console.log("-----\n");
}
```

```typescript
inputs = { messages: [{ role: "user", content: "What's it known for?" }] };
stream = await agent.stream(inputs, {
  ...config,
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  } else if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  } else {
    console.log(msg);
  }
  console.log("-----\n");
}
```

```typescript
inputs = {
  messages: [{ role: "user", content: "how close is it to boston?" }],
};
config = { configurable: { thread_id: "2" } };
stream = await agent.stream(inputs, {
  ...config,
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  } else if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  } else {
    console.log(msg);
  }
  console.log("-----\n");
}
```

```typescript
function humanReviewNode(state: typeof GraphAnnotation.State) {
  // this is the value we'll be providing via new Command({ resume: <human_review> })
  const humanReview = interrupt({
    question: "Is this correct?",
    // Surface tool calls for review
    tool_call,
  });

  const [reviewAction, reviewData] = humanReview;

  // Approve the tool call and continue
  if (reviewAction === "continue") {
    return new Command({ goto: "run_tool" });
  }

  // Modify the tool call manually and then continue
  if (reviewAction === "update") {
    const updatedMsg = getUpdatedMsg(reviewData);
    return new Command({
      goto: "run_tool",
      update: { messages: [updatedMsg] },
    });
  }

  // Give natural language feedback, and then pass that back to the agent
  if (reviewAction === "feedback") {
    const feedbackMsg = getFeedbackMsg(reviewData);
    return new Command({
      goto: "call_llm",
      update: { messages: [feedbackMsg] },
    });
  }

  throw new Error("Unreachable");
}
```

```typescript
import {
  MessagesAnnotation,
  StateGraph,
  START,
  END,
  MemorySaver,
  Command,
  interrupt,
} from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { ToolCall } from "@langchain/core/messages/tool";

const weatherSearch = tool(
  (input: { city: string }) => {
    console.log("----");
    console.log(`Searching for: ${input.city}`);
    console.log("----");
    return "Sunny!";
  },
  {
    name: "weather_search",
    description: "Search for the weather",
    schema: z.object({
      city: z.string(),
    }),
  }
);

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
}).bindTools([weatherSearch]);

const callLLM = async (state: typeof MessagesAnnotation.State) => {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
};

const humanReviewNode = async (
  state: typeof MessagesAnnotation.State
): Promise<Command> => {
  const lastMessage = state.messages.at(-1) as AIMessage | undefined;
  const toolCall = lastMessage?.tool_calls?.at(-1);

  const humanReview = interrupt<
    {
      question: string;
      toolCall: ToolCall;
    },
    {
      action: string;
      data: any;
    }
  >({
    question: "Is this correct?",
    toolCall: toolCall,
  });

  const reviewAction = humanReview.action;
  const reviewData = humanReview.data;

  if (reviewAction === "continue") {
    return new Command({ goto: "run_tool" });
  }

  if (reviewAction === "update") {
    const updatedMessage = {
      role: "ai",
      content: lastMessage.content,
      tool_calls: [
        {
          id: toolCall.id,
          name: toolCall.name,
          args: reviewData,
        },
      ],
      id: lastMessage.id,
    };
    return new Command({
      goto: "run_tool",
      update: { messages: [updatedMessage] },
    });
  }

  if (reviewAction === "feedback") {
    const toolMessage = new ToolMessage({
      name: toolCall.name,
      content: reviewData,
      tool_call_id: toolCall.id,
    });
    return new Command({
      goto: "call_llm",
      update: { messages: [toolMessage] },
    });
  }

  throw new Error("Invalid review action");
};

const runTool = async (state: typeof MessagesAnnotation.State) => {
  const newMessages: ToolMessage[] = [];
  const tools = { weather_search: weatherSearch };
  const lastMessage = state.messages.at(-1) as AIMessage | undefined;
  const toolCalls = lastMessage.tool_calls!;

  for (const toolCall of toolCalls) {
    const tool = tools[toolCall.name as keyof typeof tools];
    const result = await tool.invoke(toolCall.args);
    newMessages.push(
      new ToolMessage({
        name: toolCall.name,
        content: result,
        tool_call_id: toolCall.id,
      })
    );
  }
  return { messages: newMessages };
};

const routeAfterLLM = (
  state: typeof MessagesAnnotation.State
): typeof END | "human_review_node" => {
  const lastMessage = state.messages.at(-1) as AIMessage | undefined;
  if (!lastMessage?.tool_calls?.length) {
    return END;
  }
  return "human_review_node";
};

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("call_llm", callLLM)
  .addNode("run_tool", runTool)
  .addNode("human_review_node", humanReviewNode, {
    ends: ["run_tool", "call_llm"],
  })
  .addEdge(START, "call_llm")
  .addConditionalEdges("call_llm", routeAfterLLM, ["human_review_node", END])
  .addEdge("run_tool", "call_llm");

const memory = new MemorySaver();

const graph = workflow.compile({ checkpointer: memory });
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
let inputs = { messages: [{ role: "user", content: "hi!" }] };
let config = {
  configurable: { thread_id: "1" },
  streamMode: "values" as const,
};

let stream = await graph.stream(inputs, config);

for await (const event of stream) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
let state = await graph.getState(config);
console.log(state.next);
```

```typescript
inputs = { messages: [{ role: "user", content: "what's the weather in SF?" }] };
config = { configurable: { thread_id: "2" }, streamMode: "values" as const };

stream = await graph.stream(inputs, config);

for await (const event of stream) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
state = await graph.getState(config);
console.log(state.next);
```

```typescript
import { Command } from "@langchain/langgraph";

for await (const event of await graph.stream(
  new Command({ resume: { action: "continue" } }),
  config
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
inputs = { messages: [{ role: "user", content: "what's the weather in SF?" }] };
config = { configurable: { thread_id: "3" }, streamMode: "values" as const };

stream = await graph.stream(inputs, config);

for await (const event of stream) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
state = await graph.getState(config);
console.log(state.next);
```

```typescript
for await (const event of await graph.stream(
  new Command({
    resume: {
      action: "update",
      data: { city: "San Francisco" },
    },
  }),
  config
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
inputs = { messages: [{ role: "user", content: "what's the weather in SF?" }] };
config = { configurable: { thread_id: "4" }, streamMode: "values" as const };

stream = await graph.stream(inputs, config);

for await (const event of stream) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
state = await graph.getState(config);
console.log(state.next);
```

```typescript
for await (const event of await graph.stream(
  new Command({
    resume: {
      action: "feedback",
      data: "User requested changes: use <city, country> format for location",
    },
  }),
  config
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
state = await graph.getState(config);
console.log(state.next);
```

```typescript
for await (const event of await graph.stream(
  new Command({
    resume: {
      action: "continue",
    },
  }),
  config
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
const threadConfig = { configurable: { thread_id: "1" }, streamMode: "values" };

for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
const allCheckpoints = [];

for await (const state of graph.getStateHistory(threadConfig)) {
  allCheckpoints.push(state);
}
```

```typescript
const threadConfig = {
  configurable: { thread_id: "1", checkpoint_id: "xyz" },
  streamMode: "values",
};

for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
const threadConfig = { configurable: { thread_id: "1", checkpoint_id: "xyz" } };

graph.updateState(threadConfig, { state: "updated state" });
```

```typescript
const threadConfig = {
  configurable: { thread_id: "1", checkpoint_id: "xyz-fork" },
  streamMode: "values",
};

for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
import {
  Annotation,
  MemorySaver,
  NodeInterrupt,
  StateGraph,
} from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
  input: Annotation<string>,
});

const step1 = async (state: typeof StateAnnotation.State) => {
  console.log("---Step 1---");
  return state;
};

const step2 = async (state: typeof StateAnnotation.State) => {
  // Let's optionally raise a NodeInterrupt
  // if the length of the input is longer than 5 characters
  if (state.input?.length > 5) {
    throw new NodeInterrupt(
      `Received input that is longer than 5 characters: ${state.input}`
    );
  }
  console.log("---Step 2---");
  return state;
};

const step3 = async (state: typeof StateAnnotation.State) => {
  console.log("---Step 3---");
  return state;
};

const checkpointer = new MemorySaver();

const graph = new StateGraph(StateAnnotation)
  .addNode("step1", step1)
  .addNode("step2", step2)
  .addNode("step3", step3)
  .addEdge("__start__", "step1")
  .addEdge("step1", "step2")
  .addEdge("step2", "step3")
  .addEdge("step3", "__end__")
  .compile({ checkpointer });
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
const initialInput = { input: "hello" };
const config = {
  configurable: {
    thread_id: "1",
  },
  streamMode: "values" as const,
};

const stream = await graph.stream(initialInput, config);

for await (const event of stream) {
  console.log(event);
}
```

```typescript
const state = await graph.getState(config);
console.log(state.next);
console.log(state.tasks);
```

```typescript
const longInput = { input: "hello world" };
const config2 = {
  configurable: {
    thread_id: "2",
  },
  streamMode: "values" as const,
};

const streamWithInterrupt = await graph.stream(longInput, config2);

for await (const event of streamWithInterrupt) {
  console.log(event);
}
```

```typescript
const state2 = await graph.getState(config2);
console.log(state2.next);
console.log(JSON.stringify(state2.tasks, null, 2));
```

```typescript
// NOTE: to resume the graph from a dynamic interrupt we use the same syntax as
// regular interrupts -- we pass null as the input
const resumedStream = await graph.stream(null, config2);

for await (const event of resumedStream) {
  console.log(event);
}
```

```typescript
const state3 = await graph.getState(config2);
console.log(state3.next);
console.log(JSON.stringify(state2.tasks, null, 2));
```

```typescript
// NOTE: this update will be applied as of the last successful node before the interrupt,
// i.e. `step1`, right before the node with an interrupt
await graph.updateState(config2, { input: "short" });

const updatedStream = await graph.stream(null, config2);

for await (const event of updatedStream) {
  console.log(event);
}

const state4 = await graph.getState(config2);
console.log(state4.next);
console.log(state4.values);
```

```typescript
const config3 = {
  configurable: {
    thread_id: "3",
  },
  streamMode: "values" as const,
};

const skipStream = await graph.stream({ input: "hello world" }, config3);

// Run the graph until the first interruption
for await (const event of skipStream) {
  console.log(event);
}
```

```typescript
// NOTE: this update will skip the node `step2` entirely
await graph.updateState(config3, undefined, "step2");

// Resume the stream
for await (const event of await graph.stream(null, config3)) {
  console.log(event);
}

const state5 = await graph.getState(config3);
console.log(state5.next);
console.log(state5.values);
```

```typescript
// Import from "@langchain/langgraph/web"
import { END, START, StateGraph, Annotation } from "@langchain/langgraph/web";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

const nodeFn = async (_state: typeof GraphState.State) => {
  return { messages: [new HumanMessage("Hello from the browser!")] };
};

// Define a new graph
const workflow = new StateGraph(GraphState)
  .addNode("node", nodeFn)
  .addEdge(START, "node")
  .addEdge("node", END);

const app = workflow.compile({});

// Use the Runnable
const finalState = await app.invoke({ messages: [] });

console.log(finalState.messages.at(-1)?.text);
```

```typescript
// Import from "@langchain/langgraph/web"
import { END, START, StateGraph, Annotation } from "@langchain/langgraph/web";
import { BaseMessage } from "@langchain/core/messages";
import { RunnableLambda } from "@langchain/core/runnables";
import { type StreamEvent } from "@langchain/core/tracers/log_stream";

const GraphState2 = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

const nodeFn2 = async (_state: typeof GraphState2.State) => {
  // Note that we do not pass any `config` through here
  const nestedFn = RunnableLambda.from(async (input: string) => {
    return new HumanMessage(`Hello from ${input}!`);
  }).withConfig({ runName: "nested" });
  const responseMessage = await nestedFn.invoke("a nested function");
  return { messages: [responseMessage] };
};

// Define a new graph
const workflow2 = new StateGraph(GraphState2)
  .addNode("node", nodeFn2)
  .addEdge(START, "node")
  .addEdge("node", END);

const app2 = workflow2.compile({});

// Stream intermediate steps from the graph
const eventStream2 = app2.streamEvents(
  { messages: [] },
  { version: "v2" },
  { includeNames: ["nested"] }
);

const events2: StreamEvent[] = [];
for await (const event of eventStream2) {
  console.log(event);
  events2.push(event);
}

console.log(`Received ${events2.length} events from the nested function`);
```

```typescript
// Import from "@langchain/langgraph/web"
import { END, START, StateGraph, Annotation } from "@langchain/langgraph/web";
import { BaseMessage } from "@langchain/core/messages";
import { type RunnableConfig, RunnableLambda } from "@langchain/core/runnables";
import { type StreamEvent } from "@langchain/core/tracers/log_stream";

const GraphState3 = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

// Note the second argument here.
const nodeFn3 = async (
  _state: typeof GraphState3.State,
  config?: RunnableConfig
) => {
  // If you need to nest deeper, remember to pass `_config` when invoking
  const nestedFn = RunnableLambda.from(
    async (input: string, _config?: RunnableConfig) => {
      return new HumanMessage(`Hello from ${input}!`);
    }
  ).withConfig({ runName: "nested" });
  const responseMessage = await nestedFn.invoke("a nested function", config);
  return { messages: [responseMessage] };
};

// Define a new graph
const workflow3 = new StateGraph(GraphState3)
  .addNode("node", nodeFn3)
  .addEdge(START, "node")
  .addEdge("node", END);

const app3 = workflow3.compile({});

// Stream intermediate steps from the graph
const eventStream3 = app3.streamEvents(
  { messages: [] },
  { version: "v2" },
  { includeNames: ["nested"] }
);

const events3: StreamEvent[] = [];
for await (const event of eventStream3) {
  console.log(event);
  events3.push(event);
}

console.log(`Received ${events3.length} events from the nested function`);
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__...";
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Configuration: LangGraphJS";
```

```typescript
import { BaseMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableConfig } from "@langchain/core/runnables";
import { END, START, StateGraph, Annotation } from "@langchain/langgraph";

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  userInfo: Annotation<string | undefined>({
    reducer: (x, y) => {
      return y ? y : x ? x : "N/A";
    },
    default: () => "N/A",
  }),
});

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant.\n\n## User Info:\n{userInfo}"],
  ["placeholder", "{messages}"],
]);

const callModel = async (
  state: typeof AgentState.State,
  config?: RunnableConfig
) => {
  const { messages, userInfo } = state;
  const modelName = config?.configurable?.model;
  const model =
    modelName === "claude"
      ? new ChatAnthropic({ model: "claude-3-haiku-20240307" })
      : new ChatOpenAI({ model: "gpt-4o" });
  const chain = promptTemplate.pipe(model);
  const response = await chain.invoke(
    {
      messages,
      userInfo,
    },
    config
  );
  return { messages: [response] };
};

const fetchUserInformation = async (
  _: typeof AgentState.State,
  config?: RunnableConfig
) => {
  const userDB = {
    user1: {
      name: "John Doe",
      email: "jod@langchain.ai",
      phone: "+1234567890",
    },
    user2: {
      name: "Jane Doe",
      email: "jad@langchain.ai",
      phone: "+0987654321",
    },
  };
  const userId = config?.configurable?.user;
  if (userId) {
    const user = userDB[userId as keyof typeof userDB];
    if (user) {
      return {
        userInfo: `Name: ${user.name}\nEmail: ${user.email}\nPhone: ${user.phone}`,
      };
    }
  }
  return { userInfo: "N/A" };
};

const workflow = new StateGraph(AgentState)
  .addNode("fetchUserInfo", fetchUserInformation)
  .addNode("agent", callModel)
  .addEdge(START, "fetchUserInfo")
  .addEdge("fetchUserInfo", "agent")
  .addEdge("agent", END);

const graph = workflow.compile();
```

```typescript
import { HumanMessage } from "@langchain/core/messages";

const config = {
  configurable: {
    model: "openai",
    user: "user1",
  },
};
const inputs = {
  messages: [new HumanMessage("Could you remind me of my email??")],
};
for await (const { messages } of await graph.stream(inputs, {
  ...config,
  streamMode: "values",
})) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  } else if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  } else {
    console.log(msg);
  }
  console.log("-----\n");
}
```

```typescript
const config2 = {
  configurable: {
    model: "openai",
    user: "user2",
  },
};
const inputs2 = {
  messages: [new HumanMessage("Could you remind me of my email??")],
};
for await (const { messages } of await graph.stream(inputs2, {
  ...config2,
  streamMode: "values",
})) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  } else if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  } else {
    console.log(msg);
  }
  console.log("-----\n");
}
```

```typescript
import { MessagesAnnotation } from "@langchain/langgraph";

const ConfigurableAnnotation = Annotation.Root({
  expectedField: Annotation<string>,
});

const printNode = async (
  state: typeof MessagesAnnotation.State,
  config: RunnableConfig<typeof ConfigurableAnnotation.State>
) => {
  console.log("Expected", config.configurable?.expectedField);
  // @ts-expect-error This type will be present even though is not in the typing
  console.log("Unexpected", config.configurable?.unexpectedField);
  return {};
};

const graphWithConfigSchema = new StateGraph(
  MessagesAnnotation,
  ConfigurableAnnotation
)
  .addNode("printNode", printNode)
  .addEdge(START, "printNode")
  .compile();

const result = await graphWithConfigSchema.invoke(
  {
    messages: [{ role: "user", content: "Echo!" }],
  },
  {
    configurable: {
      expectedField: "I am expected",
      unexpectedField: "I am unexpected but present",
    },
  }
);
```

```typescript
import { z } from "zod";
import { tool } from "@langchain/core/tools";

const getWeather = tool(
  async ({ location }) => {
    if (location === "SAN FRANCISCO") {
      return "It's 60 degrees and foggy";
    } else if (location.toLowerCase() === "san francisco") {
      throw new Error("Input queries must be all capitals");
    } else {
      throw new Error("Invalid input.");
    }
  },
  {
    name: "get_weather",
    description: "Call to get the current weather",
    schema: z.object({
      location: z.string(),
    }),
  }
);
```

```typescript
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatAnthropic } from "@langchain/anthropic";
import { BaseMessage, isAIMessage } from "@langchain/core/messages";

const toolNode = new ToolNode([getWeather]);

const modelWithTools = new ChatAnthropic({
  model: "claude-3-haiku-20240307",
  temperature: 0,
}).bindTools([getWeather]);

const shouldContinue = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1);
  if (isAIMessage(lastMessage) && lastMessage.tool_calls?.length) {
    return "tools";
  }
  return "__end__";
};

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const response = await modelWithTools.invoke(messages);
  return { messages: [response] };
};

const app = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue, {
    // Explicitly list possible destinations so that
    // we can automatically draw the graph below.
    tools: "tools",
    __end__: "__end__",
  })
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
const response = await app.invoke({
  messages: [
    { role: "user", content: "what is the weather in san francisco?" },
  ],
});

for (const message of response.messages) {
  // Anthropic returns tool calls in content as well as in `AIMessage.tool_calls`
  const content = JSON.stringify(message.content, null, 2);
  console.log(`${message.getType().toUpperCase()}: ${content}`);
}
```

```typescript
import { StringOutputParser } from "@langchain/core/output_parsers";

const haikuRequestSchema = z.object({
  topic: z.array(z.string()).length(3),
});

const masterHaikuGenerator = tool(
  async ({ topic }) => {
    const model = new ChatAnthropic({
      model: "claude-3-haiku-20240307",
      temperature: 0,
    });
    const chain = model.pipe(new StringOutputParser());
    const topics = topic.join(", ");
    const haiku = await chain.invoke(`Write a haiku about ${topics}`);
    return haiku;
  },
  {
    name: "master_haiku_generator",
    description: "Generates a haiku based on the provided topics.",
    schema: haikuRequestSchema,
  }
);

const customStrategyToolNode = new ToolNode([masterHaikuGenerator]);

const customStrategyModel = new ChatAnthropic({
  model: "claude-3-haiku-20240307",
  temperature: 0,
});
const customStrategyModelWithTools = customStrategyModel.bindTools([
  masterHaikuGenerator,
]);

const customStrategyShouldContinue = async (
  state: typeof MessagesAnnotation.State
) => {
  const { messages } = state;
  const lastMessage = messages.at(-1);
  if (isAIMessage(lastMessage) && lastMessage.tool_calls?.length) {
    return "tools";
  }
  return "__end__";
};

const customStrategyCallModel = async (
  state: typeof MessagesAnnotation.State
) => {
  const { messages } = state;
  const response = await customStrategyModelWithTools.invoke(messages);
  return { messages: [response] };
};

const customStrategyApp = new StateGraph(MessagesAnnotation)
  .addNode("tools", customStrategyToolNode)
  .addNode("agent", customStrategyCallModel)
  .addEdge("__start__", "agent")
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", customStrategyShouldContinue, {
    // Explicitly list possible destinations so that
    // we can automatically draw the graph below.
    tools: "tools",
    __end__: "__end__",
  })
  .compile();

const response2 = await customStrategyApp.invoke(
  {
    messages: [
      { role: "user", content: "Write me an incredible haiku about water." },
    ],
  },
  { recursionLimit: 10 }
);

for (const message of response2.messages) {
  // Anthropic returns tool calls in content as well as in `AIMessage.tool_calls`
  const content = JSON.stringify(message.content, null, 2);
  console.log(`${message.getType().toUpperCase()}: ${content}`);
}
```

```typescript
import {
  AIMessage,
  ToolMessage,
  RemoveMessage,
} from "@langchain/core/messages";

const haikuRequestSchema2 = z.object({
  topic: z.array(z.string()).length(3),
});

const masterHaikuGenerator2 = tool(
  async ({ topic }) => {
    const model = new ChatAnthropic({
      model: "claude-3-haiku-20240307",
      temperature: 0,
    });
    const chain = model.pipe(new StringOutputParser());
    const topics = topic.join(", ");
    const haiku = await chain.invoke(`Write a haiku about ${topics}`);
    return haiku;
  },
  {
    name: "master_haiku_generator",
    description: "Generates a haiku based on the provided topics.",
    schema: haikuRequestSchema2,
  }
);

const callTool2 = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const toolsByName = { master_haiku_generator: masterHaikuGenerator };
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  const outputMessages: ToolMessage[] = [];
  for (const toolCall of lastMessage.tool_calls) {
    try {
      const toolResult = await toolsByName[toolCall.name].invoke(toolCall);
      outputMessages.push(toolResult);
    } catch (error: any) {
      // Return the error if the tool call fails
      outputMessages.push(
        new ToolMessage({
          content: error.message,
          name: toolCall.name,
          tool_call_id: toolCall.id!,
          additional_kwargs: { error },
        })
      );
    }
  }
  return { messages: outputMessages };
};

const model = new ChatAnthropic({
  model: "claude-3-haiku-20240307",
  temperature: 0,
});
const modelWithTools2 = model.bindTools([masterHaikuGenerator2]);

const betterModel = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
  temperature: 0,
});
const betterModelWithTools = betterModel.bindTools([masterHaikuGenerator2]);

const shouldContinue2 = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1);
  if (isAIMessage(lastMessage) && lastMessage.tool_calls?.length) {
    return "tools";
  }
  return "__end__";
};

const shouldFallback = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const failedToolMessages = messages.find((message) => {
    return (
      message.getType() === "tool" &&
      message.additional_kwargs.error !== undefined
    );
  });
  if (failedToolMessages) {
    return "remove_failed_tool_call_attempt";
  }
  return "agent";
};

const callModel2 = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const response = await modelWithTools2.invoke(messages);
  return { messages: [response] };
};

const removeFailedToolCallAttempt = async (
  state: typeof MessagesAnnotation.State
) => {
  const { messages } = state;
  // Remove all messages from the most recent
  // instance of AIMessage onwards.
  const lastAIMessageIndex = messages
    .map((msg, index) => ({ msg, index }))
    .reverse()
    .findIndex(({ msg }) => isAIMessage(msg));
  const messagesToRemove = messages.slice(lastAIMessageIndex);
  return {
    messages: messagesToRemove.map((m) => new RemoveMessage({ id: m.id })),
  };
};

const callFallbackModel = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const response = await betterModelWithTools.invoke(messages);
  return { messages: [response] };
};

const app2 = new StateGraph(MessagesAnnotation)
  .addNode("tools", callTool2)
  .addNode("agent", callModel2)
  .addNode("remove_failed_tool_call_attempt", removeFailedToolCallAttempt)
  .addNode("fallback_agent", callFallbackModel)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue2, {
    // Explicitly list possible destinations so that
    // we can automatically draw the graph below.
    tools: "tools",
    __end__: "__end__",
  })
  .addConditionalEdges("tools", shouldFallback, {
    remove_failed_tool_call_attempt: "remove_failed_tool_call_attempt",
    agent: "agent",
  })
  .addEdge("remove_failed_tool_call_attempt", "fallback_agent")
  .addEdge("fallback_agent", "tools")
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app2.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
const stream = await app2.stream(
  {
    messages: [
      { role: "user", content: "Write me an incredible haiku about water." },
    ],
  },
  { recursionLimit: 10 }
);

for await (const chunk of stream) {
  console.log(chunk);
}
```

```typescript
import { StateGraph, Annotation } from "@langchain/langgraph";

const SubgraphStateAnnotation = Annotation.Root({
  foo: Annotation<string>, // note that this key is shared with the parent graph state
  bar: Annotation<string>,
});

const subgraphNode1 = async (state: typeof SubgraphStateAnnotation.State) => {
  return { bar: "bar" };
};

const subgraphNode2 = async (state: typeof SubgraphStateAnnotation.State) => {
  // note that this node is using a state key ('bar') that is only available in the subgraph
  // and is sending update on the shared state key ('foo')
  return { foo: state.foo + state.bar };
};

const subgraphBuilder = new StateGraph(SubgraphStateAnnotation)
  .addNode("subgraphNode1", subgraphNode1)
  .addNode("subgraphNode2", subgraphNode2)
  .addEdge("__start__", "subgraphNode1")
  .addEdge("subgraphNode1", "subgraphNode2");

const subgraph = subgraphBuilder.compile();

// Define parent graph
const ParentStateAnnotation = Annotation.Root({
  foo: Annotation<string>,
});

const node1 = async (state: typeof ParentStateAnnotation.State) => {
  return {
    foo: "hi! " + state.foo,
  };
};

const builder = new StateGraph(ParentStateAnnotation)
  .addNode("node1", node1)
  // note that we're adding the compiled subgraph as a node to the parent graph
  .addNode("node2", subgraph)
  .addEdge("__start__", "node1")
  .addEdge("node1", "node2");

const graph = builder.compile();
```

```typescript
const stream = await graph.stream({ foo: "foo" });

for await (const chunk of stream) {
  console.log(chunk);
}
```

```typescript
const streamWithSubgraphs = await graph.stream(
  { foo: "foo" },
  { subgraphs: true }
);

for await (const chunk of streamWithSubgraphs) {
  console.log(chunk);
}
```

```typescript
import { StateGraph, Annotation } from "@langchain/langgraph";

const SubgraphAnnotation = Annotation.Root({
  bar: Annotation<string>, // note that this key is shared with the parent graph state
  baz: Annotation<string>,
});

const subgraphNodeOne = async (state: typeof SubgraphAnnotation.State) => {
  return { baz: "baz" };
};

const subgraphNodeTwo = async (state: typeof SubgraphAnnotation.State) => {
  return { bar: state.bar + state.baz };
};

const subgraphCalledInFunction = new StateGraph(SubgraphAnnotation)
  .addNode("subgraphNode1", subgraphNodeOne)
  .addNode("subgraphNode2", subgraphNodeTwo)
  .addEdge("__start__", "subgraphNode1")
  .addEdge("subgraphNode1", "subgraphNode2")
  .compile();

// Define parent graph
const ParentAnnotation = Annotation.Root({
  foo: Annotation<string>,
});

const nodeOne = async (state: typeof ParentAnnotation.State) => {
  return {
    foo: "hi! " + state.foo,
  };
};

const nodeTwo = async (state: typeof ParentAnnotation.State) => {
  const response = await subgraphCalledInFunction.invoke({
    bar: state.foo,
  });
  return { foo: response.bar };
};

const graphWithFunction = new StateGraph(ParentStateAnnotation)
  .addNode("node1", nodeOne)
  // note that we're adding the compiled subgraph as a node to the parent graph
  .addNode("node2", nodeTwo)
  .addEdge("__start__", "node1")
  .addEdge("node1", "node2")
  .compile();
```

```typescript
const graphWithFunctionStream = await graphWithFunction.stream(
  { foo: "foo" },
  { subgraphs: true }
);
for await (const chunk of graphWithFunctionStream) {
  console.log(chunk);
}
```

```typescript
process.env.ANTHROPIC_API_KEY = "your-anthropic-api-key";
```

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const getWeather = tool(
  (input) => {
    if (["sf", "san francisco"].includes(input.location.toLowerCase())) {
      return "It's 60 degrees and foggy.";
    } else {
      return "It's 90 degrees and sunny.";
    }
  },
  {
    name: "get_weather",
    description: "Call to get the current weather.",
    schema: z.object({
      location: z.string().describe("Location to get the weather for."),
    }),
  }
);

const getCoolestCities = tool(
  () => {
    return "nyc, sf";
  },
  {
    name: "get_coolest_cities",
    description: "Get a list of coolest cities",
    schema: z.object({
      noOp: z.string().optional().describe("No-op parameter."),
    }),
  }
);
```

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

const tools = [getWeather, getCoolestCities];
const toolNode = new ToolNode(tools);
```

```typescript
import { AIMessage } from "@langchain/core/messages";

const messageWithSingleToolCall = new AIMessage({
  content: "",
  tool_calls: [
    {
      name: "get_weather",
      args: { location: "sf" },
      id: "tool_call_id",
      type: "tool_call",
    },
  ],
});

await toolNode.invoke({ messages: [messageWithSingleToolCall] });
```

```typescript
const messageWithMultipleToolCalls = new AIMessage({
  content: "",
  tool_calls: [
    {
      name: "get_coolest_cities",
      args: {},
      id: "tool_call_id",
      type: "tool_call",
    },
    {
      name: "get_weather",
      args: { location: "sf" },
      id: "tool_call_id_2",
      type: "tool_call",
    },
  ],
});

await toolNode.invoke({ messages: [messageWithMultipleToolCalls] });
```

```typescript
import { ChatAnthropic } from "@langchain/anthropic";

const modelWithTools = new ChatAnthropic({
  model: "claude-3-haiku-20240307",
  temperature: 0,
}).bindTools(tools);
```

```typescript
const responseMessage = await modelWithTools.invoke(
  "what's the weather in sf?"
);

responseMessage.tool_calls;
```

```typescript
await toolNode.invoke({
  messages: [await modelWithTools.invoke("what's the weather in sf?")],
});
```

```typescript
import {
  StateGraph,
  MessagesAnnotation,
  END,
  START,
} from "@langchain/langgraph";

const toolNodeForGraph = new ToolNode(tools);

const shouldContinue = (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1);
  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls?.length
  ) {
    return "tools";
  }
  return END;
};

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;
  const response = await modelWithTools.invoke(messages);
  return { messages: response };
};

const workflow = new StateGraph(MessagesAnnotation)
  // Define the two nodes we will cycle between
  .addNode("agent", callModel)
  .addNode("tools", toolNodeForGraph)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, ["tools", END])
  .addEdge("tools", "agent");

const app = workflow.compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
import { HumanMessage } from "@langchain/core/messages";

// example with a single tool call
const stream = await app.stream(
  {
    messages: [{ role: "user", content: "what's the weather in sf?" }],
  },
  {
    streamMode: "values",
  }
);
for await (const chunk of stream) {
  const lastMessage = chunk.messages.at(-1);
  const type = lastMessage.getType();
  const content = lastMessage.content;
  const toolCalls = lastMessage.tool_calls;
  console.dir(
    {
      type,
      content,
      toolCalls,
    },
    { depth: null }
  );
}
```

```typescript
// example with a multiple tool calls in succession
const streamWithMultiToolCalls = await app.stream(
  {
    messages: [
      { role: "user", content: "what's the weather in the coolest cities?" },
    ],
  },
  {
    streamMode: "values",
  }
);
for await (const chunk of streamWithMultiToolCalls) {
  const lastMessage = chunk.messages.at(-1);
  const type = lastMessage.getType();
  const content = lastMessage.content;
  const toolCalls = lastMessage.tool_calls;
  console.dir(
    {
      type,
      content,
      toolCalls,
    },
    { depth: null }
  );
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__...";
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
// process.env.LANGCHAIN_TRACING = "true";
// process.env.LANGCHAIN_PROJECT = "Stream Tokens: LangGraphJS";
```

```typescript
import { Annotation } from "@langchain/langgraph";
import type { BaseMessageLike } from "@langchain/core/messages";

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessageLike[]>({
    reducer: (x, y) => x.concat(y),
  }),
});
```

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const searchTool = tool(
  (_) => {
    // This is a placeholder for the actual implementation
    return "Cold, with a low of 3℃";
  },
  {
    name: "search",
    description:
      "Use to surf the web, fetch current information, check the weather, and retrieve other information.",
    schema: z.object({
      query: z.string().describe("The query to use in your search."),
    }),
  }
);

await searchTool.invoke({ query: "What's the weather like?" });

const tools = [searchTool];
```

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

const toolNode = new ToolNode(tools);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});
```

```typescript
const boundModel = model.bindTools(tools);
```

```typescript
import { StateGraph, END } from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";

const routeMessage = (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  // If no tools are called, we can finish (respond to the user)
  if (!lastMessage?.tool_calls?.length) {
    return END;
  }
  // Otherwise if there is, we continue and call the tools
  return "tools";
};

const callModel = async (state: typeof StateAnnotation.State) => {
  // For versions of @langchain/core < 0.2.3, you must call `.stream()`
  // and aggregate the message from chunks instead of calling `.invoke()`.
  const { messages } = state;
  const responseMessage = await boundModel.invoke(messages);
  return { messages: [responseMessage] };
};

const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", routeMessage)
  .addEdge("tools", "agent");

const agent = workflow.compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
import { isAIMessageChunk } from "@langchain/core/messages";

const stream = await agent.stream(
  {
    messages: [
      { role: "user", content: "What's the current weather in Nepal?" },
    ],
  },
  { streamMode: "messages" }
);

for await (const [message, _metadata] of stream) {
  if (isAIMessageChunk(message) && message.tool_call_chunks?.length) {
    console.log(
      `${message.getType()} MESSAGE TOOL CALL CHUNK: ${
        message.tool_call_chunks[0].args
      }`
    );
  } else {
    console.log(`${message.getType()} MESSAGE CONTENT: ${message.content}`);
  }
}
```

```typescript
import { RunnableLambda } from "@langchain/core/runnables";

const unstreamed = async (_: typeof StateAnnotation.State) => {
  const model = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
  });
  const res = await model.invoke("How are you?");
  console.log("LOGGED UNSTREAMED MESSAGE", res.content);
  // Don't update the state, this is just to show a call that won't be streamed
  return {};
};

const agentWithNoStream = new StateGraph(StateAnnotation)
  .addNode(
    "unstreamed",
    // Add a "nostream" tag to the entire node
    RunnableLambda.from(unstreamed).withConfig({
      tags: ["nostream"],
    })
  )
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  // Run the unstreamed node before the agent
  .addEdge("__start__", "unstreamed")
  .addEdge("unstreamed", "agent")
  .addConditionalEdges("agent", routeMessage)
  .addEdge("tools", "agent")
  .compile();

const stream = await agentWithNoStream.stream(
  {
    messages: [
      { role: "user", content: "What's the current weather in Nepal?" },
    ],
  },
  { streamMode: "messages" }
);

for await (const [message, _metadata] of stream) {
  if (isAIMessageChunk(message) && message.tool_call_chunks?.length) {
    console.log(
      `${message.getType()} MESSAGE TOOL CALL CHUNK: ${
        message.tool_call_chunks[0].args
      }`
    );
  } else {
    console.log(`${message.getType()} MESSAGE CONTENT: ${message.content}`);
  }
}
```

```typescript
const eventStream = await agent.streamEvents(
  { messages: [{ role: "user", content: "What's the weather like today?" }] },
  {
    version: "v2",
  }
);

for await (const { event, data } of eventStream) {
  if (event === "on_chat_model_stream" && isAIMessageChunk(data.chunk)) {
    if (
      data.chunk.tool_call_chunks !== undefined &&
      data.chunk.tool_call_chunks.length > 0
    ) {
      console.log(data.chunk.tool_call_chunks);
    }
  }
}
```

```typescript
import { RetryPolicy } from "@langchain/langgraph";

const retryPolicy: RetryPolicy = {};
```

```typescript
import Database from "better-sqlite3";
import { ChatAnthropic } from "@langchain/anthropic";
import {
  MessagesAnnotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";

// Create an in-memory database
const db: typeof Database.prototype = new Database(":memory:");

const model = new ChatAnthropic({ model: "claude-3-5-sonnet-20240620" });

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
};

const queryDatabase = async (state: typeof MessagesAnnotation.State) => {
  const queryResult: string = JSON.stringify(
    db.prepare("SELECT * FROM Artist LIMIT 10;").all()
  );

  return { messages: [new AIMessage({ content: "queryResult" })] };
};

const workflow = new StateGraph(MessagesAnnotation)
  // Define the two nodes we will cycle between
  .addNode("call_model", callModel, { retryPolicy: { maxAttempts: 5 } })
  .addNode("query_database", queryDatabase, {
    retryPolicy: {
      retryOn: (e: any): boolean => {
        if (e instanceof Database.SqliteError) {
          // Retry on "SQLITE_BUSY" error
          return e.code === "SQLITE_BUSY";
        }
        return false; // Don't retry on other errors
      },
    },
  })
  .addEdge(START, "call_model")
  .addEdge("call_model", "query_database")
  .addEdge("query_database", END);

const graph = workflow.compile();
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";
```

```typescript
// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__...";
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Managing Agent Steps: LangGraphJS";
```

```typescript
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});
```

```typescript
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";

const searchTool = new DynamicStructuredTool({
  name: "search",
  description: "Call to surf the web.",
  schema: z.object({
    query: z.string().describe("The query to use in your search."),
  }),
  func: async ({}: { query: string }) => {
    // This is a placeholder, but don't tell the LLM that...
    return "Try again in a few seconds! Checking with the weathermen... Call be again next.";
  },
});

const tools = [searchTool];
```

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

const toolNode = new ToolNode<typeof AgentState.State>(tools);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});
```

```typescript
// After we've done this, we should make sure the model knows that it has these tools available to call.
// We can do this by binding the tools to the model class.
const boundModel = model.bindTools(tools);
```

```typescript
import { END } from "@langchain/langgraph";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";

// Define the function that determines whether to continue or not
const shouldContinue = (state: typeof AgentState.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  // If there is no function call, then we finish
  if (!lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
    return END;
  }
  // Otherwise if there is, we continue
  return "tools";
};

// **MODIFICATION**
//
// Here we don't pass all messages to the model but rather only pass the `N` most recent. Note that this is a terribly simplistic way to handle messages meant as an illustration, and there may be other methods you may want to look into depending on your use case. We also have to make sure we don't truncate the chat history to include the tool message first, as this would cause an API error.
const callModel = async (
  state: typeof AgentState.State,
  config?: RunnableConfig
) => {
  let modelMessages = [];
  for (let i = state.messages.length - 1; i >= 0; i--) {
    modelMessages.push(state.messages[i]);
    if (modelMessages.length >= 5) {
      if (!ToolMessage.isInstance(modelMessages.at(-1))) {
        break;
      }
    }
  }
  modelMessages.reverse();

  const response = await boundModel.invoke(modelMessages, config);
  // We return an object, because this will get added to the existing list
  return { messages: [response] };
};
```

```typescript
import { START, StateGraph } from "@langchain/langgraph";

// Define a new graph
const workflow = new StateGraph(AgentState)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent");

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const app = workflow.compile();
```

```typescript
import { HumanMessage, isAIMessage } from "@langchain/core/messages";
import { GraphRecursionError } from "@langchain/langgraph";

const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message.getType()}]: ${message.content}`;
  if (
    (isAIMessage(message) && (message as AIMessage)?.tool_calls?.length) ||
    0 > 0
  ) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

const inputs = {
  messages: [
    new HumanMessage(
      "what is the weather in sf? Don't give up! Keep using your tools."
    ),
  ],
};
// Setting the recursionLimit will set a max number of steps. We expect this to endlessly loop :)
try {
  for await (const output of await app.stream(inputs, {
    streamMode: "values",
    recursionLimit: 10,
  })) {
    const lastMessage = output.messages.at(-1);
    prettyPrint(lastMessage);
    console.log("-----\n");
  }
} catch (e) {
  // Since we are truncating the chat history, the agent never gets the chance
  // to see enough information to know to stop, so it will keep looping until we hit the
  // maximum recursion limit.
  if ((e as GraphRecursionError).name === "GraphRecursionError") {
    console.log("As expected, maximum steps reached. Exiting.");
  } else {
    console.error(e);
  }
}
```

```typescript
const USER_ID_TO_USER_INFO = {
  abc123: {
    user_id: "abc123",
    name: "Bob Dylan",
    location: "New York, NY",
  },
  zyx987: {
    user_id: "zyx987",
    name: "Taylor Swift",
    location: "Beverly Hills, CA",
  },
};
```

```typescript
import { Annotation, Command, MessagesAnnotation } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";

import { z } from "zod";

const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  // user provided
  lastName: Annotation<string>,
  // updated by the tool
  userInfo: Annotation<Record<string, any>>,
});

const lookupUserInfo = tool(
  async (_, config) => {
    const userId = config.configurable?.user_id;
    if (userId === undefined) {
      throw new Error("Please provide a user id in config.configurable");
    }
    if (USER_ID_TO_USER_INFO[userId] === undefined) {
      throw new Error(`User "${userId}" not found`);
    }
    // Populated when a tool is called with a tool call from a model as input
    const toolCallId = config.toolCall.id;
    return new Command({
      update: {
        // update the state keys
        userInfo: USER_ID_TO_USER_INFO[userId],
        // update the message history
        messages: [
          {
            role: "tool",
            content: "Successfully looked up user information",
            tool_call_id: toolCallId,
          },
        ],
      },
    });
  },
  {
    name: "lookup_user_info",
    description:
      "Always use this to look up information about the user to better assist them with their questions.",
    schema: z.object({}),
  }
);
```

```typescript
const stateModifier = (state: typeof StateAnnotation.State) => {
  const userInfo = state.userInfo;
  if (userInfo == null) {
    return state.messages;
  }
  const systemMessage = `User name is ${userInfo.name}. User lives in ${userInfo.location}`;
  return [
    {
      role: "system",
      content: systemMessage,
    },
    ...state.messages,
  ];
};
```

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o",
});

const agent = createReactAgent({
  llm: model,
  tools: [lookupUserInfo],
  stateSchema: StateAnnotation,
  stateModifier: stateModifier,
});
```

```typescript
const stream = await agent.stream(
  {
    messages: [
      {
        role: "user",
        content: "hi, what should i do this weekend?",
      },
    ],
  },
  {
    // provide user ID in the config
    configurable: { user_id: "abc123" },
  }
);

for await (const chunk of stream) {
  console.log(chunk);
}
```

```typescript
const taylorStream = await agent.stream(
  {
    messages: [
      {
        role: "user",
        content: "hi, what should i do this weekend?",
      },
    ],
  },
  {
    // provide user ID in the config
    configurable: { user_id: "zyx987" },
  }
);

for await (const chunk of taylorStream) {
  console.log(chunk);
}
```

```typescript
import {
  MessagesAnnotation,
  isCommand,
  Command,
  StateGraph,
} from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { isAIMessage } from "@langchain/core/messages";

import { z } from "zod";

const myTool = tool(
  async () => {
    return new Command({
      update: {
        messages: [
          {
            role: "assistant",
            content: "hi there!",
            name: "Greeter",
          },
        ],
      },
    });
  },
  {
    name: "greeting",
    description: "Updates the current state with a greeting",
    schema: z.object({}),
  }
);

const toolExecutor = async (state: typeof MessagesAnnotation.State) => {
  const message = state.messages.at(-1);
  if (
    !isAIMessage(message) ||
    message.tool_calls === undefined ||
    message.tool_calls.length === 0
  ) {
    throw new Error(
      "Most recent message must be an AIMessage with a tool call."
    );
  }
  const outputs = await Promise.all(
    message.tool_calls.map(async (toolCall) => {
      // Using a single tool for simplicity, would need to select tools by toolCall.name
      // in practice.
      const toolResult = await myTool.invoke(toolCall);
      return toolResult;
    })
  );
  // Handle mixed Command and non-Command outputs
  const combinedOutputs = outputs.map((output) => {
    if (isCommand(output)) {
      return output;
    }
    // Tool invocation result is a ToolMessage, return a normal state update
    return { messages: [output] };
  });
  // Return an array of values instead of an object
  return combinedOutputs;
};

// Simple one node graph
const customGraph = new StateGraph(MessagesAnnotation)
  .addNode("runTools", toolExecutor)
  .addEdge("__start__", "runTools")
  .compile();

await customGraph.invoke({
  messages: [
    {
      role: "user",
      content: "how are you?",
    },
    {
      role: "assistant",
      content: "Let me call the greeting tool and find out!",
      tool_calls: [
        {
          id: "123",
          args: {},
          name: "greeting",
        },
      ],
    },
  ],
});
```

```typescript
// process.env.OPENAI_API_KEY = "sk-...";
```

```typescript
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});
```

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const searchTool = tool(
  async ({ query: _query }: { query: string }) => {
    // This is a placeholder for the actual implementation
    return "Cold, with a low of 3℃";
  },
  {
    name: "search",
    description:
      "Use to surf the web, fetch current information, check the weather, and retrieve other information.",
    schema: z.object({
      query: z.string().describe("The query to use in your search."),
    }),
  }
);

await searchTool.invoke({ query: "What's the weather like?" });

const tools = [searchTool];
```

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

const toolNode = new ToolNode(tools);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({ model: "gpt-4o" });
```

```typescript
const boundModel = model.bindTools(tools);
```

```typescript
import { END, START, StateGraph } from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";

const routeMessage = (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  // If no tools are called, we can finish (respond to the user)
  if (!lastMessage?.tool_calls?.length) {
    return END;
  }
  // Otherwise if there is, we continue and call the tools
  return "tools";
};

const callModel = async (state: typeof StateAnnotation.State) => {
  // For versions of @langchain/core < 0.2.3, you must call `.stream()`
  // and aggregate the message from chunks instead of calling `.invoke()`.
  const { messages } = state;
  const responseMessage = await boundModel.invoke(messages);
  return { messages: [responseMessage] };
};

const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", routeMessage)
  .addEdge("tools", "agent");

const graph = workflow.compile();
```

```typescript
let inputs = {
  messages: [{ role: "user", content: "what's the weather in sf" }],
};

for await (const chunk of await graph.stream(inputs, {
  streamMode: "values",
})) {
  console.log(chunk["messages"]);
  console.log("\n====\n");
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__..."
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Direct Return: LangGraphJS";
```

```typescript
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";

const SearchTool = z.object({
  query: z.string().describe("query to look up online"),
  // **IMPORTANT** We are adding an **extra** field here
  // that isn't used directly by the tool - it's used by our
  // graph instead to determine whether or not to return the
  // result directly to the user
  return_direct: z
    .boolean()
    .describe(
      "Whether or not the result of this should be returned directly to the user without you seeing what it is"
    )
    .default(false),
});

const searchTool = new DynamicStructuredTool({
  name: "search",
  description: "Call to surf the web.",
  // We are overriding the default schema here to
  // add an extra field
  schema: SearchTool,
  func: async ({}: { query: string }) => {
    // This is a placeholder for the actual implementation
    // Don't let the LLM know this though 😊
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈.";
  },
});

const tools = [searchTool];
```

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

const toolNode = new ToolNode(tools);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  temperature: 0,
  model: "gpt-3.5-turbo",
});
// This formats the tools as json schema for the model API.
// The model then uses this like a system prompt.
const boundModel = model.bindTools(tools);
```

```typescript
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});
```

```typescript
import { RunnableConfig } from "@langchain/core/runnables";
import { END } from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";

// Define the function that determines whether to continue or not
const shouldContinue = (state: typeof AgentState.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  // If there is no function call, then we finish
  if (!lastMessage?.tool_calls?.length) {
    return END;
  } // Otherwise if there is, we check if it's suppose to return direct
  else {
    const args = lastMessage.tool_calls[0].args;
    if (args?.return_direct) {
      return "final";
    } else {
      return "tools";
    }
  }
};

// Define the function that calls the model
const callModel = async (
  state: typeof AgentState.State,
  config?: RunnableConfig
) => {
  const messages = state.messages;
  const response = await boundModel.invoke(messages, config);
  // We return an object, because this will get added to the existing list
  return { messages: [response] };
};
```

```typescript
import { START, StateGraph } from "@langchain/langgraph";

// Define a new graph
const workflow = new StateGraph(AgentState)
  // Define the two nodes we will cycle between
  .addNode("agent", callModel)
  // Note the "action" and "final" nodes are identical!
  .addNode("tools", toolNode)
  .addNode("final", toolNode)
  // Set the entrypoint as `agent`
  .addEdge(START, "agent")
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `agent`.
    "agent",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue
  )
  // We now add a normal edge from `tools` to `agent`.
  .addEdge("tools", "agent")
  .addEdge("final", END);

// Finally, we compile it!
const app = workflow.compile();
```

```typescript
import { HumanMessage, isAIMessage } from "@langchain/core/messages";

const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message.getType()}]: ${message.content}`;
  if (
    (isAIMessage(message) && (message as AIMessage)?.tool_calls?.length) ||
    0 > 0
  ) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

const inputs = { messages: [new HumanMessage("what is the weather in sf")] };
for await (const output of await app.stream(inputs, { streamMode: "values" })) {
  const lastMessage = output.messages.at(-1);
  prettyPrint(lastMessage);
  console.log("-----\n");
}
```

```typescript
const inputs2 = {
  messages: [
    new HumanMessage(
      "what is the weather in sf? return this result directly by setting return_direct = True"
    ),
  ],
};
for await (const output of await app.stream(inputs2, {
  streamMode: "values",
})) {
  const lastMessage = output.messages.at(-1);
  prettyPrint(lastMessage);
  console.log("-----\n");
}
```

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { InMemoryStore } from "@langchain/langgraph";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

const store = new InMemoryStore({
  index: {
    embeddings,
    dims: 1536,
  },
});
```

```typescript
// Store some memories
await store.put(["user_123", "memories"], "1", { text: "I love pizza" });
await store.put(["user_123", "memories"], "2", {
  text: "I prefer Italian food",
});
await store.put(["user_123", "memories"], "3", {
  text: "I don't like spicy food",
});
await store.put(["user_123", "memories"], "3", {
  text: "I am studying econometrics",
});
await store.put(["user_123", "memories"], "3", { text: "I am a plumber" });
```

```typescript
// Find memories about food preferences

const memories = await store.search(["user_123", "memories"], {
  query: "I like food?",
  limit: 5,
});

for (const memory of memories) {
  console.log(`Memory: ${memory.value.text} (similarity: ${memory.score})`);
}
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {
  MessagesAnnotation,
  LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { getContextVariable } from "@langchain/core/context";

import { z } from "zod";
import { v4 as uuidv4 } from "uuid";

const addMemories = async (
  state: typeof MessagesAnnotation.State,
  config: LangGraphRunnableConfig
) => {
  const store = config.store;
  // Search based on user's last message
  const items = await store.search(["user_123", "memories"], {
    // Assume it's not a complex message
    query: state.messages.at(-1)?.text as string,
    limit: 2,
  });

  const memories = items.length
    ? `## Memories of user\n${items.map((item) => item.value.text).join("\n")}`
    : "";

  // Add retrieved memories to system message
  return [
    { role: "system", content: `You are a helpful assistant.\n${memories}` },
    ...state.messages,
  ];
};

const upsertMemoryTool = tool(
  async (input, config: LangGraphRunnableConfig): Promise<string> => {
    const store = config.store;
    if (!store) {
      throw new Error("No store provided to tool.");
    }
    const memoryId = getContextVariable("memoryId") || uuidv4();
    await store.put(["user_123", "memories"], memoryId, {
      text: input.content,
    });
    return `Stored memory ${memoryId}`;
  },
  {
    name: "upsert_memory",
    schema: z.object({
      content: z.string(),
    }),
    description: "Upsert a memory in the database.",
  }
);

const agent = createReactAgent({
  llm: new ChatOpenAI({ model: "gpt-4o-mini" }),
  tools: [upsertMemoryTool],
  stateModifier: addMemories,
  store: store,
});
```

```typescript
const stream = await agent.stream(
  {
    messages: [
      {
        role: "user",
        content: "I'm hungry",
      },
    ],
  },
  {
    streamMode: "messages",
  }
);

for await (const [message, _metadata] of stream) {
  console.log(message.content);
}
```

```typescript
import { InMemoryStore } from "@langchain/langgraph";

// Configure store to embed both memory content and emotional context
const multiVectorStore = new InMemoryStore({
  index: {
    embeddings: embeddings,
    dims: 1536,
    fields: ["memory", "emotional_context"],
  },
});

// Store memories with different content/emotion pairs
await multiVectorStore.put(["user_123", "memories"], "mem1", {
  memory: "Had pizza with friends at Mario's",
  emotional_context: "felt happy and connected",
  this_isnt_indexed: "I prefer ravioli though",
});
await multiVectorStore.put(["user_123", "memories"], "mem2", {
  memory: "Ate alone at home",
  emotional_context: "felt a bit lonely",
  this_isnt_indexed: "I like pie",
});

// Search focusing on emotional state - matches mem2
const results = await multiVectorStore.search(["user_123", "memories"], {
  query: "times they felt isolated",
  limit: 1,
});

console.log("Expect mem 2");

for (const r of results) {
  console.log(`Item: ${r.key}; Score(${r.score})`);
  console.log(`Memory: ${r.value.memory}`);
  console.log(`Emotion: ${r.value.emotional_context}`);
}
```

```typescript
import { InMemoryStore } from "@langchain/langgraph";

const overrideStore = new InMemoryStore({
  index: {
    embeddings: embeddings,
    dims: 1536,
    // Default to embed memory field
    fields: ["memory"],
  },
});

// Store one memory with default indexing
await overrideStore.put(["user_123", "memories"], "mem1", {
  memory: "I love spicy food",
  context: "At a Thai restaurant",
});

// Store another overriding which fields to embed
await overrideStore.put(["user_123", "memories"], "mem2", {
  memory: "I love spicy food",
  context: "At a Thai restaurant",
  // Override: only embed the context
  index: ["context"],
});

// Search about food - matches mem1 (using default field)
console.log("Expect mem1");
const results2 = await overrideStore.search(["user_123", "memories"], {
  query: "what food do they like",
  limit: 1,
});

for (const r of results2) {
  console.log(`Item: ${r.key}; Score(${r.score})`);
  console.log(`Memory: ${r.value.memory}`);
}

// Search about restaurant atmosphere - matches mem2 (using overridden field)
console.log("Expect mem2");
const results3 = await overrideStore.search(["user_123", "memories"], {
  query: "restaurant environment",
  limit: 1,
});

for (const r of results3) {
  console.log(`Item: ${r.key}; Score(${r.score})`);
  console.log(`Memory: ${r.value.memory}`);
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__..."
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Branching: LangGraphJS";
```

```typescript
import { END, START, StateGraph, Annotation } from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
  aggregate: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

// Create the graph
const nodeA = (state: typeof StateAnnotation.State) => {
  console.log(`Adding I'm A to ${state.aggregate}`);
  return { aggregate: [`I'm A`] };
};
const nodeB = (state: typeof StateAnnotation.State) => {
  console.log(`Adding I'm B to ${state.aggregate}`);
  return { aggregate: [`I'm B`] };
};
const nodeC = (state: typeof StateAnnotation.State) => {
  console.log(`Adding I'm C to ${state.aggregate}`);
  return { aggregate: [`I'm C`] };
};
const nodeD = (state: typeof StateAnnotation.State) => {
  console.log(`Adding I'm D to ${state.aggregate}`);
  return { aggregate: [`I'm D`] };
};

const builder = new StateGraph(StateAnnotation)
  .addNode("a", nodeA)
  .addEdge(START, "a")
  .addNode("b", nodeB)
  .addNode("c", nodeC)
  .addNode("d", nodeD)
  .addEdge("a", "b")
  .addEdge("a", "c")
  .addEdge("b", "d")
  .addEdge("c", "d")
  .addEdge("d", END);

const graph = builder.compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
// Invoke the graph
const baseResult = await graph.invoke({ aggregate: [] });
console.log("Base Result: ", baseResult);
```

```typescript
const ConditionalBranchingAnnotation = Annotation.Root({
  aggregate: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  which: Annotation<string>({
    reducer: (x: string, y: string) => y ?? x,
  }),
});

// Create the graph
const nodeA2 = (state: typeof ConditionalBranchingAnnotation.State) => {
  console.log(`Adding I'm A to ${state.aggregate}`);
  return { aggregate: [`I'm A`] };
};
const nodeB2 = (state: typeof ConditionalBranchingAnnotation.State) => {
  console.log(`Adding I'm B to ${state.aggregate}`);
  return { aggregate: [`I'm B`] };
};
const nodeC2 = (state: typeof ConditionalBranchingAnnotation.State) => {
  console.log(`Adding I'm C to ${state.aggregate}`);
  return { aggregate: [`I'm C`] };
};
const nodeD2 = (state: typeof ConditionalBranchingAnnotation.State) => {
  console.log(`Adding I'm D to ${state.aggregate}`);
  return { aggregate: [`I'm D`] };
};
const nodeE2 = (state: typeof ConditionalBranchingAnnotation.State) => {
  console.log(`Adding I'm E to ${state.aggregate}`);
  return { aggregate: [`I'm E`] };
};

// Define the route function
function routeCDorBC(
  state: typeof ConditionalBranchingAnnotation.State
): string[] {
  if (state.which === "cd") {
    return ["c", "d"];
  }
  return ["b", "c"];
}

const builder2 = new StateGraph(ConditionalBranchingAnnotation)
  .addNode("a", nodeA2)
  .addEdge(START, "a")
  .addNode("b", nodeB2)
  .addNode("c", nodeC2)
  .addNode("d", nodeD2)
  .addNode("e", nodeE2)
  // Add conditional edges
  // Third parameter is to support visualizing the graph
  .addConditionalEdges("a", routeCDorBC, ["b", "c", "d"])
  .addEdge("b", "e")
  .addEdge("c", "e")
  .addEdge("d", "e")
  .addEdge("e", END);

const graph2 = builder2.compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph2.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
// Invoke the graph
let g2result = await graph2.invoke({ aggregate: [], which: "bc" });
console.log("Result 1: ", g2result);
```

```typescript
g2result = await graph2.invoke({ aggregate: [], which: "cd" });
console.log("Result 2: ", g2result);
```

```typescript
type ScoredValue = {
  value: string;
  score: number;
};

const reduceFanouts = (left?: ScoredValue[], right?: ScoredValue[]) => {
  if (!left) {
    left = [];
  }
  if (!right || right?.length === 0) {
    // Overwrite. Similar to redux.
    return [];
  }
  return left.concat(right);
};

const StableSortingAnnotation = Annotation.Root({
  aggregate: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  which: Annotation<string>({
    reducer: (x: string, y: string) => y ?? x,
  }),
  fanoutValues: Annotation<ScoredValue[]>({
    reducer: reduceFanouts,
  }),
});

class ParallelReturnNodeValue {
  private _value: string;
  private _score: number;

  constructor(nodeSecret: string, score: number) {
    this._value = nodeSecret;
    this._score = score;
  }

  public call(state: typeof StableSortingAnnotation.State) {
    console.log(`Adding ${this._value} to ${state.aggregate}`);
    return { fanoutValues: [{ value: this._value, score: this._score }] };
  }
}

// Create the graph

const nodeA3 = (state: typeof StableSortingAnnotation.State) => {
  console.log(`Adding I'm A to ${state.aggregate}`);
  return { aggregate: ["I'm A"] };
};

const nodeB3 = new ParallelReturnNodeValue("I'm B", 0.1);
const nodeC3 = new ParallelReturnNodeValue("I'm C", 0.9);
const nodeD3 = new ParallelReturnNodeValue("I'm D", 0.3);

const aggregateFanouts = (state: typeof StableSortingAnnotation.State) => {
  // Sort by score (reversed)
  state.fanoutValues.sort((a, b) => b.score - a.score);
  return {
    aggregate: state.fanoutValues.map((v) => v.value).concat(["I'm E"]),
    fanoutValues: [],
  };
};

// Define the route function
function routeBCOrCD(state: typeof StableSortingAnnotation.State): string[] {
  if (state.which === "cd") {
    return ["c", "d"];
  }
  return ["b", "c"];
}

const builder3 = new StateGraph(StableSortingAnnotation)
  .addNode("a", nodeA3)
  .addEdge(START, "a")
  .addNode("b", nodeB3.call.bind(nodeB3))
  .addNode("c", nodeC3.call.bind(nodeC3))
  .addNode("d", nodeD3.call.bind(nodeD3))
  .addNode("e", aggregateFanouts)
  .addConditionalEdges("a", routeBCOrCD, ["b", "c", "d"])
  .addEdge("b", "e")
  .addEdge("c", "e")
  .addEdge("d", "e")
  .addEdge("e", END);

const graph3 = builder3.compile();

// Invoke the graph
let g3result = await graph3.invoke({ aggregate: [], which: "bc" });
console.log("Result 1: ", g3result);
```

```typescript
let g3result2 = await graph3.invoke({ aggregate: [], which: "cd" });
console.log("Result 2: ", g3result2);
```

```typescript
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

```typescript
import { z } from "zod";
import { ChatAnthropic } from "@langchain/anthropic";
import { StateGraph, END, START, Annotation, Send } from "@langchain/langgraph";

/* Model and prompts */

// Define model and prompts we will use
const subjectsPrompt =
  "Generate a comma separated list of between 2 and 5 examples related to: {topic}.";
const jokePrompt = "Generate a joke about {subject}";
const bestJokePrompt = `Below are a bunch of jokes about {topic}. Select the best one! Return the ID (index) of the best one.

{jokes}`;

// Zod schemas for getting structured output from the LLM
const Subjects = z.object({
  subjects: z.array(z.string()),
});
const Joke = z.object({
  joke: z.string(),
});
const BestJoke = z.object({
  id: z.number(),
});

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
});

/* Graph components: define the components that will make up the graph */

// This will be the overall state of the main graph.
// It will contain a topic (which we expect the user to provide)
// and then will generate a list of subjects, and then a joke for
// each subject
const OverallState = Annotation.Root({
  topic: Annotation<string>,
  subjects: Annotation<string[]>,
  // Notice here we pass a reducer function.
  // This is because we want combine all the jokes we generate
  // from individual nodes back into one list.
  jokes: Annotation<string[]>({
    reducer: (state, update) => state.concat(update),
  }),
  bestSelectedJoke: Annotation<string>,
});

// This will be the state of the node that we will "map" all
// subjects to in order to generate a joke
interface JokeState {
  subject: string;
}

// This is the function we will use to generate the subjects of the jokes
const generateTopics = async (
  state: typeof OverallState.State
): Promise<Partial<typeof OverallState.State>> => {
  const prompt = subjectsPrompt.replace("topic", state.topic);
  const response = await model
    .withStructuredOutput(Subjects, { name: "subjects" })
    .invoke(prompt);
  return { subjects: response.subjects };
};

// Function to generate a joke
const generateJoke = async (state: JokeState): Promise<{ jokes: string[] }> => {
  const prompt = jokePrompt.replace("subject", state.subject);
  const response = await model
    .withStructuredOutput(Joke, { name: "joke" })
    .invoke(prompt);
  return { jokes: [response.joke] };
};

// Here we define the logic to map out over the generated subjects
// We will use this an edge in the graph
const continueToJokes = (state: typeof OverallState.State) => {
  // We will return a list of `Send` objects
  // Each `Send` object consists of the name of a node in the graph
  // as well as the state to send to that node
  return state.subjects.map((subject) => new Send("generateJoke", { subject }));
};

// Here we will judge the best joke
const bestJoke = async (
  state: typeof OverallState.State
): Promise<Partial<typeof OverallState.State>> => {
  const jokes = state.jokes.join("\n\n");
  const prompt = bestJokePrompt
    .replace("jokes", jokes)
    .replace("topic", state.topic);
  const response = await model
    .withStructuredOutput(BestJoke, { name: "best_joke" })
    .invoke(prompt);
  return { bestSelectedJoke: state.jokes[response.id] };
};

// Construct the graph: here we put everything together to construct our graph
const graph = new StateGraph(OverallState)
  .addNode("generateTopics", generateTopics)
  .addNode("generateJoke", generateJoke)
  .addNode("bestJoke", bestJoke)
  .addEdge(START, "generateTopics")
  .addConditionalEdges("generateTopics", continueToJokes)
  .addEdge("generateJoke", "bestJoke")
  .addEdge("bestJoke", END);

const app = graph.compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
// Call the graph: here we call it to generate a list of jokes
for await (const s of await app.stream({ topic: "animals" })) {
  console.log(s);
}
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const model = new ChatOpenAI({
  model: "gpt-4o",
});

const getWeather = tool(
  (input) => {
    if (
      ["sf", "san francisco", "san francisco, ca"].includes(
        input.location.toLowerCase()
      )
    ) {
      return "It's 60 degrees and foggy.";
    } else {
      return "It's 90 degrees and sunny.";
    }
  },
  {
    name: "get_weather",
    description: "Call to get the current weather.",
    schema: z.object({
      location: z.string().describe("Location to get the weather for."),
    }),
  }
);

const graph = createReactAgent({ llm: model, tools: [getWeather] });
```

```typescript
let inputs = {
  messages: [{ role: "user", content: "what's the weather in sf?" }],
};

let stream = await graph.stream(inputs, {
  streamMode: ["updates", "debug"],
});

for await (const chunk of stream) {
  console.log(`Receiving new event of type: ${chunk[0]}`);
  console.log(chunk[1]);
  console.log("\n====\n");
}
```

```typescript
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";

const GraphState = Annotation.Root({
  input: Annotation<string>,
});

const step1 = (state: typeof GraphState.State) => {
  console.log("---Step 1---");
  return state;
};

const step2 = (state: typeof GraphState.State) => {
  console.log("---Step 2---");
  return state;
};

const step3 = (state: typeof GraphState.State) => {
  console.log("---Step 3---");
  return state;
};

const builder = new StateGraph(GraphState)
  .addNode("step1", step1)
  .addNode("step2", step2)
  .addNode("step3", step3)
  .addEdge(START, "step1")
  .addEdge("step1", "step2")
  .addEdge("step2", "step3")
  .addEdge("step3", END);

// Set up memory
const graphStateMemory = new MemorySaver();

const graph = builder.compile({
  checkpointer: graphStateMemory,
  interruptBefore: ["step2"],
});
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
// Input
const initialInput = { input: "hello world" };

// Thread
const graphStateConfig = {
  configurable: { thread_id: "1" },
  streamMode: "values" as const,
};

// Run the graph until the first interruption
for await (const event of await graph.stream(initialInput, graphStateConfig)) {
  console.log(`--- ${event.input} ---`);
}

// Will log when the graph is interrupted, after step 2.
console.log("--- GRAPH INTERRUPTED ---");
```

```typescript
console.log("Current state!");
const currState = await graph.getState(graphStateConfig);
console.log(currState.values);

await graph.updateState(graphStateConfig, { input: "hello universe!" });

console.log("---\n---\nUpdated state!");
const updatedState = await graph.getState(graphStateConfig);
console.log(updatedState.values);
```

```typescript
// Continue the graph execution
for await (const event of await graph.stream(null, graphStateConfig)) {
  console.log(`--- ${event.input} ---`);
}
```

```typescript
// Set up the tool
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import {
  StateGraph,
  START,
  END,
  MessagesAnnotation,
} from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage } from "@langchain/core/messages";
import { z } from "zod";

const search = tool(
  (_) => {
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈.";
  },
  {
    name: "search",
    description: "Call to surf the web.",
    schema: z.string(),
  }
);

const tools = [search];
const toolNode = new ToolNode(tools);

// Set up the model
const model = new ChatAnthropic({ model: "claude-3-5-sonnet-20240620" });
const modelWithTools = model.bindTools(tools);

// Define nodes and conditional edges

// Define the function that determines whether to continue or not
function shouldContinue(
  state: typeof MessagesAnnotation.State
): "action" | typeof END {
  const lastMessage = state.messages.at(-1);
  // If there is no function call, then we finish
  if (lastMessage && !(lastMessage as AIMessage).tool_calls?.length) {
    return END;
  }
  // Otherwise if there is, we continue
  return "action";
}

// Define the function that calls the model
async function callModel(
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  const messages = state.messages;
  const response = await modelWithTools.invoke(messages);
  // We return an object with a messages property, because this will get added to the existing list
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  // Define the two nodes we will cycle between
  .addNode("agent", callModel)
  .addNode("action", toolNode)
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `agent`.
    // This means these are the edges taken after the `agent` node is called.
    "agent",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue
  )
  // We now add a normal edge from `action` to `agent`.
  // This means that after `action` is called, `agent` node is called next.
  .addEdge("action", "agent")
  // Set the entrypoint as `agent`
  // This means that this node is the first one called
  .addEdge(START, "agent");

// Setup memory
const memory = new MemorySaver();

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const app = workflow.compile({
  checkpointer: memory,
  interruptBefore: ["action"],
});
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
// Thread
const config = {
  configurable: { thread_id: "3" },
  streamMode: "values" as const,
};

for await (const event of await app.stream(
  {
    messages: [{ role: "human", content: "search for the weather in sf now" }],
  },
  config
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
// First, lets get the current state
const currentState = await app.getState(config);

// Let's now get the last message in the state
// This is the one with the tool calls that we want to update
let lastMessage = currentState.values.messages.at(-1);

// Let's now update the args for that tool call
lastMessage.tool_calls[0].args = { query: "current weather in SF" };

// Let's now call `updateState` to pass in this message in the `messages` key
// This will get treated as any other update to the state
// It will get passed to the reducer function for the `messages` key
// That reducer function will use the ID of the message to update it
// It's important that it has the right ID! Otherwise it would get appended
// as a new message
await app.updateState(config, { messages: lastMessage });
```

```typescript
const newState = await app.getState(config);
const updatedStateToolCalls = newState.values.messages.at(-1)?.tool_calls;
console.log(updatedStateToolCalls);
```

```typescript
for await (const event of await app.stream(null, config)) {
  console.log(event);
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  if (recentMsg.getType() === "tool") {
    console.log({
      name: recentMsg.name,
      content: recentMsg.content,
    });
  } else if (recentMsg.getType() === "ai") {
    console.log(recentMsg.content);
  }
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__..."
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "ReAct Agent with system prompt: LangGraphJS";
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const model = new ChatOpenAI({
  model: "gpt-4o",
});

const getWeather = tool(
  (input) => {
    if (input.location === "sf") {
      return "It's always sunny in sf";
    } else {
      return "It might be cloudy in nyc";
    }
  },
  {
    name: "get_weather",
    description: "Call to get the current weather.",
    schema: z.object({
      location: z
        .enum(["sf", "nyc"])
        .describe("Location to get the weather for."),
    }),
  }
);

// We can add our system prompt here
const prompt = "Respond in Italian";

const agent = createReactAgent({
  llm: model,
  tools: [getWeather],
  stateModifier: prompt,
});
```

```typescript
let inputs = {
  messages: [{ role: "user", content: "what is the weather in NYC?" }],
};
let stream = await agent.stream(inputs, {
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  } else if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  } else {
    console.log(msg);
  }
  console.log("-----\n");
}
```

```typescript
process.env.OPENAI_API_KEY = "YOUR_API_KEY";
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const getWeather = tool(
  async ({ location }) => {
    const lowercaseLocation = location.toLowerCase();
    if (
      lowercaseLocation.includes("sf") ||
      lowercaseLocation.includes("san francisco")
    ) {
      return "It's sunny!";
    } else if (lowercaseLocation.includes("boston")) {
      return "It's rainy!";
    } else {
      return `I am not sure what the weather is in ${location}`;
    }
  },
  {
    name: "getWeather",
    schema: z.object({
      location: z.string().describe("location to get the weather for"),
    }),
    description: "Call to get the weather from a specific location.",
  }
);

const tools = [getWeather];
```

```typescript
import {
  type BaseMessageLike,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { type ToolCall } from "@langchain/core/messages/tool";
import { task } from "@langchain/langgraph";

const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));

const callModel = task("callModel", async (messages: BaseMessageLike[]) => {
  const response = await model.bindTools(tools).invoke(messages);
  return response;
});

const callTool = task(
  "callTool",
  async (toolCall: ToolCall): Promise<AIMessage> => {
    const tool = toolsByName[toolCall.name];
    const observation = await tool.invoke(toolCall.args);
    return new ToolMessage({ content: observation, tool_call_id: toolCall.id });
    // Can also pass toolCall directly into the tool to return a ToolMessage
    // return tool.invoke(toolCall);
  }
);
```

```typescript
import { entrypoint, addMessages } from "@langchain/langgraph";

const agent = entrypoint("agent", async (messages: BaseMessageLike[]) => {
  let currentMessages = messages;
  let llmResponse = await callModel(currentMessages);
  while (true) {
    if (!llmResponse.tool_calls?.length) {
      break;
    }

    // Execute tools
    const toolResults = await Promise.all(
      llmResponse.tool_calls.map((toolCall) => {
        return callTool(toolCall);
      })
    );

    // Append to message list
    currentMessages = addMessages(currentMessages, [
      llmResponse,
      ...toolResults,
    ]);

    // Call model again
    llmResponse = await callModel(currentMessages);
  }

  return llmResponse;
});
```

```typescript
import { BaseMessage, isAIMessage } from "@langchain/core/messages";

const prettyPrintMessage = (message: BaseMessage) => {
  console.log("=".repeat(30), `${message.getType()} message`, "=".repeat(30));
  console.log(message.content);
  if (isAIMessage(message) && message.tool_calls?.length) {
    console.log(JSON.stringify(message.tool_calls, null, 2));
  }
};

// Usage example
const userMessage = {
  role: "user",
  content: "What's the weather in san francisco?",
};
console.log(userMessage);

const stream = await agent.stream([userMessage]);

for await (const step of stream) {
  for (const [taskName, update] of Object.entries(step)) {
    const message = update as BaseMessage;
    // Only print task updates
    if (taskName === "agent") continue;
    console.log(`\n${taskName}:`);
    prettyPrintMessage(message);
  }
}
```

```typescript hl_lines="6 10 37 38 39 40"
import { MemorySaver, getPreviousState } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const agentWithMemory = entrypoint(
  {
    name: "agentWithMemory",
    checkpointer,
  },
  async (messages: BaseMessageLike[]) => {
    const previous = getPreviousState<BaseMessage>() ?? [];
    let currentMessages = addMessages(previous, messages);
    let llmResponse = await callModel(currentMessages);
    while (true) {
      if (!llmResponse.tool_calls?.length) {
        break;
      }

      // Execute tools
      const toolResults = await Promise.all(
        llmResponse.tool_calls.map((toolCall) => {
          return callTool(toolCall);
        })
      );

      // Append to message list
      currentMessages = addMessages(currentMessages, [
        llmResponse,
        ...toolResults,
      ]);

      // Call model again
      llmResponse = await callModel(currentMessages);
    }

    // Append final response for storage
    currentMessages = addMessages(currentMessages, llmResponse);

    return entrypoint.final({
      value: llmResponse,
      save: currentMessages,
    });
  }
);
```

```typescript
const config = { configurable: { thread_id: "1" } };
```

```typescript
const streamWithMemory = await agentWithMemory.stream(
  [
    {
      role: "user",
      content: "What's the weather in san francisco?",
    },
  ],
  config
);

for await (const step of streamWithMemory) {
  for (const [taskName, update] of Object.entries(step)) {
    const message = update as BaseMessage;
    // Only print task updates
    if (taskName === "agentWithMemory") continue;
    console.log(`\n${taskName}:`);
    prettyPrintMessage(message);
  }
}
```

```typescript
const followupStreamWithMemory = await agentWithMemory.stream(
  [
    {
      role: "user",
      content: "How does it compare to Boston, MA?",
    },
  ],
  config
);

for await (const step of followupStreamWithMemory) {
  for (const [taskName, update] of Object.entries(step)) {
    const message = update as BaseMessage;
    // Only print task updates
    if (taskName === "agentWithMemory") continue;
    console.log(`\n${taskName}:`);
    prettyPrintMessage(message);
  }
}
```

```typescript
import { Annotation, StateGraph } from "@langchain/langgraph";

const InputAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const OutputAnnotation = Annotation.Root({
  answer: Annotation<string>,
});

const answerNode = (_state: typeof InputAnnotation.State) => {
  return { answer: "bye" };
};

const graph = new StateGraph({
  input: InputAnnotation,
  output: OutputAnnotation,
})
  .addNode("answerNode", answerNode)
  .addEdge("__start__", "answerNode")
  .compile();

await graph.invoke({
  question: "hi",
});
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { Command, MessagesAnnotation, StateGraph } from "@langchain/langgraph";

import { z } from "zod";

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.1,
});

const makeAgentNode = (params: {
  name: string;
  destinations: string[];
  systemPrompt: string;
}) => {
  return async (state: typeof MessagesAnnotation.State) => {
    const possibleDestinations = ["__end__", ...params.destinations] as const;
    // define schema for the structured output:
    // - model's text response (`response`)
    // - name of the node to go to next (or '__end__')
    const responseSchema = z.object({
      response: z
        .string()
        .describe(
          "A human readable response to the original question. Does not need to be a final response. Will be streamed back to the user."
        ),
      goto: z
        .enum(possibleDestinations)
        .describe(
          "The next agent to call, or __end__ if the user's query has been resolved. Must be one of the specified values."
        ),
    });
    const messages = [
      {
        role: "system",
        content: params.systemPrompt,
      },
      ...state.messages,
    ];
    const response = await model
      .withStructuredOutput(responseSchema, {
        name: "router",
      })
      .invoke(messages);

    // handoff to another agent or halt
    const aiMessage = {
      role: "assistant",
      content: response.response,
      name: params.name,
    };
    return new Command({
      goto: response.goto,
      update: { messages: aiMessage },
    });
  };
};

const travelAdvisor = makeAgentNode({
  name: "travel_advisor",
  destinations: ["sightseeing_advisor", "hotel_advisor"],
  systemPrompt: [
    "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). ",
    "If you need specific sightseeing recommendations, ask 'sightseeing_advisor' for help. ",
    "If you need hotel recommendations, ask 'hotel_advisor' for help. ",
    "If you have enough information to respond to the user, return '__end__'. ",
    "Never mention other agents by name.",
  ].join(""),
});

const sightseeingAdvisor = makeAgentNode({
  name: "sightseeing_advisor",
  destinations: ["travel_advisor", "hotel_advisor"],
  systemPrompt: [
    "You are a travel expert that can provide specific sightseeing recommendations for a given destination. ",
    "If you need general travel help, go to 'travel_advisor' for help. ",
    "If you need hotel recommendations, go to 'hotel_advisor' for help. ",
    "If you have enough information to respond to the user, return 'finish'. ",
    "Never mention other agents by name.",
  ].join(""),
});

const hotelAdvisor = makeAgentNode({
  name: "hotel_advisor",
  destinations: ["travel_advisor", "sightseeing_advisor"],
  systemPrompt: [
    "You are a booking expert that provides hotel recommendations for a given destination. ",
    "If you need general travel help, ask 'travel_advisor' for help. ",
    "If you need specific sightseeing recommendations, ask 'sightseeing_advisor' for help. ",
    "If you have enough information to respond to the user, return 'finish'. ",
    "Never mention other agents by name.",
  ].join(""),
});

const graph = new StateGraph(MessagesAnnotation)
  .addNode("travel_advisor", travelAdvisor, {
    ends: ["sightseeing_advisor", "hotel_advisor", "__end__"],
  })
  .addNode("sightseeing_advisor", sightseeingAdvisor, {
    ends: ["travel_advisor", "hotel_advisor", "__end__"],
  })
  .addNode("hotel_advisor", hotelAdvisor, {
    ends: ["travel_advisor", "sightseeing_advisor", "__end__"],
  })
  // we'll always start with a general travel advisor
  .addEdge("__start__", "travel_advisor")
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
const simpleStream = await graph.stream({
  messages: [
    {
      role: "user",
      content: "i wanna go somewhere warm in the caribbean",
    },
  ],
});

for await (const chunk of simpleStream) {
  console.log(chunk);
}
```

```typescript
const recommendationStream = await graph.stream({
  messages: [
    {
      role: "user",
      content:
        "i wanna go somewhere warm in the caribbean. pick one destination, give me some things to do and hotel recommendations",
    },
  ],
});

for await (const chunk of recommendationStream) {
  console.log(chunk);
}
```

```typescript
import { Command, StateGraph, Annotation } from "@langchain/langgraph";

const GameStateAnnotation = Annotation.Root({
  // note that we're defining a reducer (operator.add) here.
  // This will allow all agents to write their updates for resources concurrently.
  wood: Annotation<number>({
    default: () => 0,
    reducer: (a, b) => a + b,
  }),
  food: Annotation<number>({
    default: () => 0,
    reducer: (a, b) => a + b,
  }),
  gold: Annotation<number>({
    default: () => 0,
    reducer: (a, b) => a + b,
  }),
  guardOnDuty: Annotation<boolean>,
});

/** Villager NPC that gathers wood and food. */
const villager = async (state: typeof GameStateAnnotation.State) => {
  const currentResources = state.wood + state.food;
  // Continue gathering until we have enough resources
  if (currentResources < 15) {
    console.log("Villager gathering resources.");
    return new Command({
      goto: "villager",
      update: {
        wood: 3,
        food: 1,
      },
    });
  }
  // NOTE: Returning Command({goto: "__end__"}) is not necessary for the graph to run correctly
  // but it's useful for visualization, to show that the agent actually halts
  return new Command({
    goto: "__end__",
  });
};

/** Guard NPC that protects gold and consumes food. */
const guard = async (state: typeof GameStateAnnotation.State) => {
  if (!state.guardOnDuty) {
    return new Command({
      goto: "__end__",
    });
  }
  // Guard needs food to keep patrolling
  if (state.food > 0) {
    console.log("Guard patrolling.");
    // Loop back to the 'guard' agent
    return new Command({
      goto: "guard",
      update: { food: -1 },
    });
  }
  console.log("Guard leaving to get food.");
  return new Command({
    goto: "__end__",
    update: {
      guardOnDuty: false,
    },
  });
};

/** Merchant NPC that trades wood for gold. */
const merchant = async (state: typeof GameStateAnnotation.State) => {
  // Trade wood for gold when available
  if (state.wood >= 5) {
    console.log("Merchant trading wood for gold.");
    return new Command({
      goto: "merchant",
      update: {
        wood: -5,
        gold: 1,
      },
    });
  }
  return new Command({
    goto: "__end__",
  });
};

/** Thief NPC that steals gold if the guard leaves to get food. */
const thief = async (state: typeof GameStateAnnotation.State) => {
  if (!state.guardOnDuty) {
    console.log("Thief stealing gold.");
    return new Command({
      goto: "__end__",
      update: { gold: -state.gold },
    });
  }
  // keep thief on standby (loop back to the 'thief' agent)
  return new Command({
    goto: "thief",
  });
};

const gameGraph = new StateGraph(GameStateAnnotation)
  .addNode("villager", villager, {
    ends: ["villager", "__end__"],
  })
  .addNode("guard", guard, {
    ends: ["guard", "__end__"],
  })
  .addNode("merchant", merchant, {
    ends: ["merchant", "__end__"],
  })
  .addNode("thief", thief, {
    ends: ["thief", "__end__"],
  })
  .addEdge("__start__", "villager")
  .addEdge("__start__", "guard")
  .addEdge("__start__", "merchant")
  .addEdge("__start__", "thief")
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await gameGraph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
const gameStream = await gameGraph.stream(
  {
    wood: 10,
    food: 3,
    gold: 10,
    guardOnDuty: true,
  },
  {
    streamMode: "values",
  }
);

for await (const state of gameStream) {
  console.log("Game state", state);
  console.log("-".repeat(50));
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__...";
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Force Calling a Tool First: LangGraphJS";
```

```typescript
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";

const searchTool = new DynamicStructuredTool({
  name: "search",
  description:
    "Use to surf the web, fetch current information, check the weather, and retrieve other information.",
  schema: z.object({
    query: z.string().describe("The query to use in your search."),
  }),
  func: async ({}: { query: string }) => {
    // This is a placeholder for the actual implementation
    return "Cold, with a low of 13 ℃";
  },
});

await searchTool.invoke({ query: "What's the weather like?" });

const tools = [searchTool];
```

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

const toolNode = new ToolNode(tools);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  temperature: 0,
  model: "gpt-4o",
});
```

```typescript
const boundModel = model.bindTools(tools);
```

```typescript
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});
```

```typescript
import { AIMessage, AIMessageChunk } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";
import { concat } from "@langchain/core/utils/stream";

// Define logic that will be used to determine which conditional edge to go down
const shouldContinue = (state: typeof AgentState.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  // If there is no function call, then we finish
  if (!lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
    return "end";
  }
  // Otherwise if there is, we continue
  return "continue";
};

// Define the function that calls the model
const callModel = async (
  state: typeof AgentState.State,
  config?: RunnableConfig
) => {
  const { messages } = state;
  let response: AIMessageChunk | undefined;
  for await (const message of await boundModel.stream(messages, config)) {
    if (!response) {
      response = message;
    } else {
      response = concat(response, message);
    }
  }
  // We return an object, because this will get added to the existing list
  return {
    messages: response ? [response as AIMessage] : [],
  };
};
```

```typescript
// This is the new first - the first call of the model we want to explicitly hard-code some action
const firstModel = async (state: typeof AgentState.State) => {
  const humanInput = state.messages.at(-1)?.text || "";
  return {
    messages: [
      new AIMessage({
        content: "",
        tool_calls: [
          {
            name: "search",
            args: {
              query: humanInput,
            },
            id: "tool_abcd123",
          },
        ],
      }),
    ],
  };
};
```

```typescript
import { END, START, StateGraph } from "@langchain/langgraph";

// Define a new graph
const workflow = new StateGraph(AgentState)
  // Define the new entrypoint
  .addNode("first_agent", firstModel)
  // Define the two nodes we will cycle between
  .addNode("agent", callModel)
  .addNode("action", toolNode)
  // Set the entrypoint as `first_agent`
  // by creating an edge from the virtual __start__ node to `first_agent`
  .addEdge(START, "first_agent")
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `agent`.
    // This means these are the edges taken after the `agent` node is called.
    "agent",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue,
    // Finally we pass in a mapping.
    // The keys are strings, and the values are other nodes.
    // END is a special node marking that the graph should finish.
    // What will happen is we will call `should_continue`, and then the output of that
    // will be matched against the keys in this mapping.
    // Based on which one it matches, that node will then be called.
    {
      // If `tools`, then we call the tool node.
      continue: "action",
      // Otherwise we finish.
      end: END,
    }
  )
  // We now add a normal edge from `tools` to `agent`.
  // This means that after `tools` is called, `agent` node is called next.
  .addEdge("action", "agent")
  // After we call the first agent, we know we want to go to action
  .addEdge("first_agent", "action");

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const app = workflow.compile();
```

```typescript
import { HumanMessage } from "@langchain/core/messages";

const inputs = {
  messages: [new HumanMessage("what is the weather in sf")],
};

for await (const output of await app.stream(inputs)) {
  console.log(output);
  console.log("-----\n");
}
```

```typescript
// process.env.OPENAI_API_KEY = "YOUR_API_KEY";
```

```typescript
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import {
  StateGraph,
  MessagesAnnotation,
  Annotation,
} from "@langchain/langgraph";

const getWeather = tool(
  async ({ city }) => {
    return `It's sunny in ${city}`;
  },
  {
    name: "get_weather",
    description: "Get the weather for a specific city",
    schema: z.object({
      city: z.string().describe("A city name"),
    }),
  }
);

const rawModel = new ChatOpenAI({ model: "gpt-4o-mini" });
const model = rawModel.withStructuredOutput(getWeather);

// Extend the base MessagesAnnotation state with another field
const SubGraphAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  city: Annotation<string>,
});

const modelNode = async (state: typeof SubGraphAnnotation.State) => {
  const result = await model.invoke(state.messages);
  return { city: result.city };
};

const weatherNode = async (state: typeof SubGraphAnnotation.State) => {
  const result = await getWeather.invoke({ city: state.city });
  return {
    messages: [
      {
        role: "assistant",
        content: result,
      },
    ],
  };
};

const subgraph = new StateGraph(SubGraphAnnotation)
  .addNode("modelNode", modelNode)
  .addNode("weatherNode", weatherNode)
  .addEdge("__start__", "modelNode")
  .addEdge("modelNode", "weatherNode")
  .addEdge("weatherNode", "__end__")
  .compile({ interruptBefore: ["weatherNode"] });
```

```typescript
import { MemorySaver } from "@langchain/langgraph";

const memory = new MemorySaver();

const RouterStateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  route: Annotation<"weather" | "other">,
});

const routerModel = rawModel.withStructuredOutput(
  z.object({
    route: z
      .enum(["weather", "other"])
      .describe(
        "A step that should execute next to based on the currnet input"
      ),
  }),
  {
    name: "router",
  }
);

const routerNode = async (state: typeof RouterStateAnnotation.State) => {
  const systemMessage = {
    role: "system",
    content: "Classify the incoming query as either about weather or not.",
  };
  const messages = [systemMessage, ...state.messages];
  const { route } = await routerModel.invoke(messages);
  return { route };
};

const normalLLMNode = async (state: typeof RouterStateAnnotation.State) => {
  const responseMessage = await rawModel.invoke(state.messages);
  return { messages: [responseMessage] };
};

const routeAfterPrediction = async (
  state: typeof RouterStateAnnotation.State
) => {
  if (state.route === "weather") {
    return "weatherGraph";
  } else {
    return "normalLLMNode";
  }
};

const graph = new StateGraph(RouterStateAnnotation)
  .addNode("routerNode", routerNode)
  .addNode("normalLLMNode", normalLLMNode)
  .addNode("weatherGraph", subgraph)
  .addEdge("__start__", "routerNode")
  .addConditionalEdges("routerNode", routeAfterPrediction)
  .addEdge("normalLLMNode", "__end__")
  .addEdge("weatherGraph", "__end__")
  .compile({ checkpointer: memory });
```

```typescript
const config = { configurable: { thread_id: "1" } };

const inputs = { messages: [{ role: "user", content: "hi!" }] };

const stream = await graph.stream(inputs, { ...config, streamMode: "updates" });

for await (const update of stream) {
  console.log(update);
}
```

```typescript
const config2 = { configurable: { thread_id: "2" } };

const streamWithBreakpoint = await graph.stream(
  {
    messages: [
      {
        role: "user",
        content: "what's the weather in sf",
      },
    ],
  },
  { ...config2, streamMode: "updates" }
);

for await (const update of streamWithBreakpoint) {
  console.log(update);
}
```

```typescript
const streamWithSubgraphs = await graph.stream(
  {
    messages: [
      {
        role: "user",
        content: "what's the weather in sf",
      },
    ],
  },
  { configurable: { thread_id: "3" }, streamMode: "updates", subgraphs: true }
);

for await (const update of streamWithSubgraphs) {
  console.log(update);
}
```

```typescript
const state = await graph.getState({ configurable: { thread_id: "3" } });
state.next;
```

```typescript
JSON.stringify(state.tasks, null, 2);
```

```typescript
const stateWithSubgraphs = await graph.getState(
  { configurable: { thread_id: "3" } },
  { subgraphs: true }
);
JSON.stringify(stateWithSubgraphs.tasks, null, 2);
```

```typescript
const resumedStream = await graph.stream(null, {
  configurable: { thread_id: "3" },
  streamMode: "values",
  subgraphs: true,
});

for await (const update of resumedStream) {
  console.log(update);
}
```

```typescript
let parentGraphStateBeforeSubgraph;

const histories = await graph.getStateHistory({
  configurable: { thread_id: "3" },
});

for await (const historyEntry of histories) {
  if (historyEntry.next[0] === "weatherGraph") {
    parentGraphStateBeforeSubgraph = historyEntry;
  }
}
```

```typescript
let subgraphStateBeforeModelNode;

const subgraphHistories = await graph.getStateHistory(
  parentGraphStateBeforeSubgraph.tasks[0].state
);

for await (const subgraphHistoryEntry of subgraphHistories) {
  if (subgraphHistoryEntry.next[0] === "modelNode") {
    subgraphStateBeforeModelNode = subgraphHistoryEntry;
  }
}

console.log(subgraphStateBeforeModelNode);
```

```typescript
subgraphStateBeforeModelNode.next;
```

```typescript
const resumeSubgraphStream = await graph.stream(null, {
  ...subgraphStateBeforeModelNode.config,
  streamMode: "updates",
  subgraphs: true,
});

for await (const value of resumeSubgraphStream) {
  console.log(value);
}
```

```typescript
const graphStream = await graph.stream(
  {
    messages: [
      {
        role: "user",
        content: "what's the weather in sf",
      },
    ],
  },
  {
    configurable: {
      thread_id: "4",
    },
  }
);

for await (const update of graphStream) {
  console.log(update);
}
```

```typescript
const outerGraphState = await graph.getState(
  {
    configurable: {
      thread_id: "4",
    },
  },
  { subgraphs: true }
);

console.log(outerGraphState.tasks[0].state);
```

```typescript
import type { StateSnapshot } from "@langchain/langgraph";

await graph.updateState(
  (outerGraphState.tasks[0].state as StateSnapshot).config,
  { city: "la" }
);
```

```typescript
const resumedStreamWithUpdatedState = await graph.stream(null, {
  configurable: {
    thread_id: "4",
  },
  streamMode: "updates",
  subgraphs: true,
});

for await (const update of resumedStreamWithUpdatedState) {
  console.log(JSON.stringify(update, null, 2));
}
```

```typescript
const streamWithAsNode = await graph.stream(
  {
    messages: [
      {
        role: "user",
        content: "What's the weather in sf",
      },
    ],
  },
  {
    configurable: {
      thread_id: "14",
    },
  }
);

for await (const update of streamWithAsNode) {
  console.log(update);
}

// Graph execution should stop before the weather node
console.log("interrupted!");

const interruptedState = await graph.getState(
  {
    configurable: {
      thread_id: "14",
    },
  },
  { subgraphs: true }
);

console.log(interruptedState);

// We update the state by passing in the message we want returned from the weather node
// and make sure to pass `"weatherNode"` to signify that we want to act as this node.
await graph.updateState(
  (interruptedState.tasks[0].state as StateSnapshot).config,
  {
    messages: [
      {
        role: "assistant",
        content: "rainy",
      },
    ],
  },
  "weatherNode"
);

const resumedStreamWithAsNode = await graph.stream(null, {
  configurable: {
    thread_id: "14",
  },
  streamMode: "updates",
  subgraphs: true,
});

for await (const update of resumedStreamWithAsNode) {
  console.log(update);
}

console.log(
  await graph.getState(
    {
      configurable: {
        thread_id: "14",
      },
    },
    { subgraphs: true }
  )
);
```

```typescript
const entireSubgraphExampleStream = await graph.stream(
  {
    messages: [
      {
        role: "user",
        content: "what's the weather in sf",
      },
    ],
  },
  {
    configurable: {
      thread_id: "8",
    },
    streamMode: "updates",
    subgraphs: true,
  }
);

for await (const update of entireSubgraphExampleStream) {
  console.log(update);
}

// Graph execution should stop before the weather node
console.log("interrupted!");

// We update the state by passing in the message we want returned from the weather graph.
// Note that we don't need to pass in the subgraph config, since we aren't updating the state inside the subgraph
await graph.updateState(
  {
    configurable: {
      thread_id: "8",
    },
  },
  {
    messages: [{ role: "assistant", content: "rainy" }],
  },
  "weatherGraph"
);

const resumedEntireSubgraphExampleStream = await graph.stream(null, {
  configurable: {
    thread_id: "8",
  },
  streamMode: "updates",
});

for await (const update of resumedEntireSubgraphExampleStream) {
  console.log(update);
}

const currentStateAfterUpdate = await graph.getState({
  configurable: {
    thread_id: "8",
  },
});

console.log(currentStateAfterUpdate.values.messages);
```

```typescript
const parentGraph = new StateGraph(RouterStateAnnotation)
  .addNode("routerNode", routerNode)
  .addNode("normalLLMNode", normalLLMNode)
  .addNode("weatherGraph", subgraph)
  .addEdge("__start__", "routerNode")
  .addConditionalEdges("routerNode", routeAfterPrediction)
  .addEdge("normalLLMNode", "__end__")
  .addEdge("weatherGraph", "__end__")
  .compile();
```

```typescript
const checkpointer = new MemorySaver();

const GrandfatherStateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  toContinue: Annotation<boolean>,
});

const grandparentRouterNode = async (
  _state: typeof GrandfatherStateAnnotation.State
) => {
  // Dummy logic that will always continue
  return { toContinue: true };
};

const grandparentConditionalEdge = async (
  state: typeof GrandfatherStateAnnotation.State
) => {
  if (state.toContinue) {
    return "parentGraph";
  } else {
    return "__end__";
  }
};

const grandparentGraph = new StateGraph(GrandfatherStateAnnotation)
  .addNode("routerNode", grandparentRouterNode)
  .addNode("parentGraph", parentGraph)
  .addEdge("__start__", "routerNode")
  .addConditionalEdges("routerNode", grandparentConditionalEdge)
  .addEdge("parentGraph", "__end__")
  .compile({ checkpointer });
```

```typescript
const grandparentConfig = {
  configurable: { thread_id: "123" },
};

const grandparentGraphStream = await grandparentGraph.stream(
  {
    messages: [
      {
        role: "user",
        content: "what's the weather in SF",
      },
    ],
  },
  {
    ...grandparentConfig,
    streamMode: "updates",
    subgraphs: true,
  }
);

for await (const update of grandparentGraphStream) {
  console.log(update);
}
```

```typescript
const grandparentGraphState = await grandparentGraph.getState(
  grandparentConfig,
  { subgraphs: true }
);

const parentGraphState = grandparentGraphState.tasks[0].state as StateSnapshot;
const subgraphState = parentGraphState.tasks[0].state as StateSnapshot;

console.log("Grandparent State:");
console.log(grandparentGraphState.values);
console.log("---------------");
console.log("Parent Graph State:");
console.log(parentGraphState.values);
console.log("---------------");
console.log("Subgraph State:");
console.log(subgraphState.values);
```

```typescript
await grandparentGraph.updateState(
  subgraphState.config,
  {
    messages: [
      {
        role: "assistant",
        content: "rainy",
      },
    ],
  },
  "weatherNode"
);

const updatedGrandparentGraphStream = await grandparentGraph.stream(null, {
  ...grandparentConfig,
  streamMode: "updates",
  subgraphs: true,
});

for await (const update of updatedGrandparentGraphStream) {
  console.log(update);
}

console.log(
  (await grandparentGraph.getState(grandparentConfig)).values.messages
);
```

```typescript
const grandparentStateHistories = await grandparentGraph.getStateHistory(
  grandparentConfig
);
for await (const state of grandparentStateHistories) {
  console.log(state);
  console.log("-----");
}
```

```typescript
import { StateGraph, START, END, MemorySaver, Annotation } from "@langchain/langgraph";

const GraphAnnotation = Annotation.Root({
  foo: Annotation<string>
  bar: Annotation<string[]>({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  })
});

function nodeA(state: typeof GraphAnnotation.State) {
  return { foo: "a", bar: ["a"] };
}

function nodeB(state: typeof GraphAnnotation.State) {
  return { foo: "b", bar: ["b"] };
}

const workflow = new StateGraph(GraphAnnotation);
  .addNode("nodeA", nodeA)
  .addNode("nodeB", nodeB)
  .addEdge(START, "nodeA")
  .addEdge("nodeA", "nodeB")
  .addEdge("nodeB", END);

const checkpointer = new MemorySaver();
const graph = workflow.compile({ checkpointer });

const config = { configurable: { thread_id: "1" } };
await graph.invoke({ foo: "" }, config);
```

```typescript
// Get the latest state snapshot
const config = { configurable: { thread_id: "1" } };
const state = await graph.getState(config);

// Get a state snapshot for a specific checkpoint_id
const configWithCheckpoint = {
  configurable: {
    thread_id: "1",
    checkpoint_id: "1ef663ba-28fe-6528-8002-5a559208592c",
  },
};
const stateWithCheckpoint = await graph.getState(configWithCheckpoint);
```

```typescript
const config = { configurable: { thread_id: "1" } };
const history = await graph.getStateHistory(config);
```

```typescript
// { configurable: { thread_id: "1" } }  // valid config
// { configurable: { thread_id: "1", checkpoint_id: "0c62ca34-ac19-445d-bbb0-5b4984975b2a" } }  // also valid config

const config = { configurable: { thread_id: "1" } };
await graph.invoke(inputs, config);
```

```typescript
import { Annotation } from "@langchain/langgraph";

const GraphAnnotation = Annotation.Root({
  foo: Annotation<string>
  bar: Annotation<string[]>({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  })
});
```

```typescript
await graph.updateState(config, { foo: "2", bar: ["b"] });
```

```typescript
import { Annotation, StateGraph } from "@langchain/langgraph";

// The overall state of the graph
const OverallStateAnnotation = Annotation.Root({
  question: Annotation<string>,
  answer: Annotation<string>,
});

// This is what the node that generates the query will return
const QueryOutputAnnotation = Annotation.Root({
  query: Annotation<string>,
});

// This is what the node that retrieves the documents will return
const DocumentOutputAnnotation = Annotation.Root({
  docs: Annotation<string[]>,
});

// This is what the node that retrieves the documents will return
const GenerateOutputAnnotation = Annotation.Root({
  ...OverallStateAnnotation.spec,
  ...DocumentOutputAnnotation.spec,
});

// Node to generate query
const generateQuery = async (state: typeof OverallStateAnnotation.State) => {
  // Replace this with real logic
  return {
    query: state.question + " rephrased as a query!",
  };
};

// Node to retrieve documents
const retrieveDocuments = async (state: typeof QueryOutputAnnotation.State) => {
  // Replace this with real logic
  return {
    docs: [state.query, "some random document"],
  };
};

// Node to generate answer
const generate = async (state: typeof GenerateOutputAnnotation.State) => {
  return {
    answer: state.docs.concat([state.question]).join("\n\n"),
  };
};

const graph = new StateGraph(OverallStateAnnotation)
  .addNode("generate_query", generateQuery)
  .addNode("retrieve_documents", retrieveDocuments, {
    input: QueryOutputAnnotation,
  })
  .addNode("generate", generate, { input: GenerateOutputAnnotation })
  .addEdge("__start__", "generate_query")
  .addEdge("generate_query", "retrieve_documents")
  .addEdge("retrieve_documents", "generate")
  .compile();

await graph.invoke({
  question: "How are you?",
});
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";
```

```typescript
// process.env.LANGCHAIN_API_KEY = "ls...";
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "Respond in Format: LangGraphJS";
```

```typescript
import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
  }),
});
```

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const searchTool = tool(
  (_) => {
    // This is a placeholder, but don't tell the LLM that...
    return "67 degrees. Cloudy with a chance of rain.";
  },
  {
    name: "search",
    description: "Call to surf the web.",
    schema: z.object({
      query: z.string().describe("The query to use in your search."),
    }),
  }
);

const tools = [searchTool];
```

```typescript
import { ToolNode } from "@langchain/langgraph/prebuilt";

const toolNode = new ToolNode<typeof GraphState.State>(tools);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  temperature: 0,
  model: "gpt-4o",
});
```

```typescript
import { tool } from "@langchain/core/tools";

const Response = z.object({
  temperature: z.number().describe("the temperature"),
  other_notes: z.string().describe("any other notes about the weather"),
});

const finalResponseTool = tool(async () => "mocked value", {
  name: "Response",
  description: "Always respond to the user using this tool.",
  schema: Response,
});

const boundModel = model.bindTools([...tools, finalResponseTool]);
```

```typescript
import { AIMessage } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";

// Define the function that determines whether to continue or not
const route = (state: typeof GraphState.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  // If there is no function call, then we finish
  if (!lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
    return "__end__";
  }
  // Otherwise if there is, we need to check what type of function call it is
  if (lastMessage.tool_calls[0].name === "Response") {
    return "__end__";
  }
  // Otherwise we continue
  return "tools";
};

// Define the function that calls the model
const callModel = async (
  state: typeof GraphState.State,
  config?: RunnableConfig
) => {
  const { messages } = state;
  const response = await boundModel.invoke(messages, config);
  // We return an object, because this will get added to the existing list
  return { messages: [response] };
};
```

```typescript
import { StateGraph } from "@langchain/langgraph";

// Define a new graph
const workflow = new StateGraph(GraphState)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges(
    // First, we define the start node. We use `agent`.
    // This means these are the edges taken after the `agent` node is called.
    "agent",
    // Next, we pass in the function that will determine which node is called next.
    route,
    // We supply a map of possible response values to the conditional edge
    // to make it possible to draw a visualization of the graph.
    {
      __end__: "__end__",
      tools: "tools",
    }
  )
  // We now add a normal edge from `tools` to `agent`.
  // This means that after `tools` is called, `agent` node is called next.
  .addEdge("tools", "agent");

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const app = workflow.compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
import { HumanMessage, isAIMessage } from "@langchain/core/messages";

const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message.getType()}]: ${message.content}`;
  if (isAIMessage(message) && message?.tool_calls?.length) {
    const tool_calls = message?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

const inputs = {
  messages: [new HumanMessage("what is the weather in sf")],
};

const stream = await app.stream(inputs, { streamMode: "values" });

for await (const output of stream) {
  const { messages } = output;
  prettyPrint(messages.at(-1));
  console.log("\n---\n");
}
```

```typescript
import { concat } from "@langchain/core/utils/stream";

const eventStream = await app.streamEvents(inputs, { version: "v2" });

let aggregatedChunk;
for await (const { event, data } of eventStream) {
  if (event === "on_chat_model_stream") {
    const { chunk } = data;
    if (aggregatedChunk === undefined) {
      aggregatedChunk = chunk;
    } else {
      aggregatedChunk = concat(aggregatedChunk, chunk);
    }
    const currentToolCalls = aggregatedChunk.tool_calls;
    if (
      currentToolCalls.length === 0 ||
      currentToolCalls[0].name === "" ||
      !finalResponseTool.name.startsWith(currentToolCalls[0].name)
    ) {
      // No tool calls or a different tool call in the message,
      // so drop what's currently aggregated and start over
      aggregatedChunk = undefined;
    } else if (currentToolCalls[0].name === finalResponseTool.name) {
      // Now we're sure that this event is part of the final output!
      // Log the partially aggregated args.
      console.log(aggregatedChunk.tool_call_chunks[0].args);

      // You can also log the raw args instead:
      // console.log(chunk.tool_call_chunks);

      console.log("---");
    }
  }
}
// Final aggregated tool call
console.log(aggregatedChunk.tool_calls);
```

```typescript
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

```typescript
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_API_KEY = "YOUR_API_KEY";
```

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import {
  SystemMessage,
  HumanMessage,
  AIMessage,
  RemoveMessage,
} from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import {
  MessagesAnnotation,
  StateGraph,
  START,
  END,
  Annotation,
} from "@langchain/langgraph";
import { v4 as uuidv4 } from "uuid";

const memory = new MemorySaver();

// We will add a `summary` attribute (in addition to `messages` key,
// which MessagesAnnotation already has)
const GraphAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  summary: Annotation<string>({
    reducer: (_, action) => action,
    default: () => "",
  }),
});

// We will use this model for both the conversation and the summarization
const model = new ChatAnthropic({ model: "claude-3-haiku-20240307" });

// Define the logic to call the model
async function callModel(
  state: typeof GraphAnnotation.State
): Promise<Partial<typeof GraphAnnotation.State>> {
  // If a summary exists, we add this in as a system message
  const { summary } = state;
  let { messages } = state;
  if (summary) {
    const systemMessage = new SystemMessage({
      id: uuidv4(),
      content: `Summary of conversation earlier: ${summary}`,
    });
    messages = [systemMessage, ...messages];
  }
  const response = await model.invoke(messages);
  // We return an object, because this will get added to the existing state
  return { messages: [response] };
}

// We now define the logic for determining whether to end or summarize the conversation
function shouldContinue(
  state: typeof GraphAnnotation.State
): "summarize_conversation" | typeof END {
  const messages = state.messages;
  // If there are more than six messages, then we summarize the conversation
  if (messages.length > 6) {
    return "summarize_conversation";
  }
  // Otherwise we can just end
  return END;
}

async function summarizeConversation(
  state: typeof GraphAnnotation.State
): Promise<Partial<typeof GraphAnnotation.State>> {
  // First, we summarize the conversation
  const { summary, messages } = state;
  let summaryMessage: string;
  if (summary) {
    // If a summary already exists, we use a different system prompt
    // to summarize it than if one didn't
    summaryMessage =
      `This is summary of the conversation to date: ${summary}\n\n` +
      "Extend the summary by taking into account the new messages above:";
  } else {
    summaryMessage = "Create a summary of the conversation above:";
  }

  const allMessages = [
    ...messages,
    new HumanMessage({
      id: uuidv4(),
      content: summaryMessage,
    }),
  ];
  const response = await model.invoke(allMessages);
  // We now need to delete messages that we no longer want to show up
  // I will delete all but the last two messages, but you can change this
  const deleteMessages = messages
    .slice(0, -2)
    .map((m) => new RemoveMessage({ id: m.id }));
  if (typeof response.content !== "string") {
    throw new Error("Expected a string response from the model");
  }
  return { summary: response.content, messages: deleteMessages };
}

// Define a new graph
const workflow = new StateGraph(GraphAnnotation)
  // Define the conversation node and the summarize node
  .addNode("conversation", callModel)
  .addNode("summarize_conversation", summarizeConversation)
  // Set the entrypoint as conversation
  .addEdge(START, "conversation")
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `conversation`.
    // This means these are the edges taken after the `conversation` node is called.
    "conversation",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue
  )
  // We now add a normal edge from `summarize_conversation` to END.
  // This means that after `summarize_conversation` is called, we end.
  .addEdge("summarize_conversation", END);

// Finally, we compile it!
const app = workflow.compile({ checkpointer: memory });
```

```typescript
const printUpdate = (update: Record<string, any>) => {
  Object.keys(update).forEach((key) => {
    const value = update[key];

    if ("messages" in value && Array.isArray(value.messages)) {
      value.messages.forEach((msg) => {
        console.log(
          `\n================================ ${msg.getType()} Message =================================`
        );
        console.log(msg.content);
      });
    }
    if ("summary" in value && value.summary) {
      console.log(value.summary);
    }
  });
};
```

```typescript
import { HumanMessage } from "@langchain/core/messages";

const config = {
  configurable: { thread_id: "4" },
  streamMode: "updates" as const,
};

const inputMessage = new HumanMessage("hi! I'm bob");
console.log(inputMessage.content);
for await (const event of await app.stream(
  { messages: [inputMessage] },
  config
)) {
  printUpdate(event);
}

const inputMessage2 = new HumanMessage("What did I sat my name was?");
console.log(inputMessage2.content);
for await (const event of await app.stream(
  { messages: [inputMessage2] },
  config
)) {
  printUpdate(event);
}

const inputMessage3 = new HumanMessage("i like the celtics!");
console.log(inputMessage3.content);
for await (const event of await app.stream(
  { messages: [inputMessage3] },
  config
)) {
  printUpdate(event);
}
```

```typescript
const values = (await app.getState(config)).values;
console.log(values);
```

```typescript
const inputMessage4 = new HumanMessage("i like how much they win");
console.log(inputMessage4.content);
for await (const event of await app.stream(
  { messages: [inputMessage4] },
  config
)) {
  printUpdate(event);
}
```

```typescript
const values2 = (await app.getState(config)).values;
console.log(values2);
```

```typescript
const inputMessage5 = new HumanMessage("what's my name?");
console.log(inputMessage5.content);
for await (const event of await app.stream(
  { messages: [inputMessage5] },
  config
)) {
  printUpdate(event);
}
```

```typescript
const inputMessage6 = new HumanMessage("what NFL team do you think I like?");
console.log(inputMessage6.content);
for await (const event of await app.stream(
  { messages: [inputMessage6] },
  config
)) {
  printUpdate(event);
}
```

```typescript
const inputMessage7 = new HumanMessage("i like the patriots!");
console.log(inputMessage7.content);
for await (const event of await app.stream(
  { messages: [inputMessage7] },
  config
)) {
  printUpdate(event);
}
```

```typescript
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ChatAnthropic } from "@langchain/anthropic";

const getWeather = tool(
  async ({ city }) => {
    if (city === "nyc") {
      return "It might be cloudy in nyc";
    } else if (city === "sf") {
      return "It's always sunny in sf";
    } else {
      throw new Error("Unknown city.");
    }
  },
  {
    name: "get_weather",
    schema: z.object({
      city: z.enum(["nyc", "sf"]),
    }),
    description: "Use this to get weather information",
  }
);

const tools = [getWeather];

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
}).bindTools(tools);

// We add a tag that we'll be using later to filter outputs
const finalModel = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
}).withConfig({
  tags: ["final_node"],
});
```

```typescript
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";

const shouldContinue = async (state: typeof MessagesAnnotation.State) => {
  const messages = state.messages;
  const lastMessage: AIMessage = messages.at(-1);
  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user)
  return "final";
};

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const messages = state.messages;
  const response = await model.invoke(messages);
  // We return a list, because this will get added to the existing list
  return { messages: [response] };
};

const callFinalModel = async (state: typeof MessagesAnnotation.State) => {
  const messages = state.messages;
  const lastAIMessage = messages.at(-1);
  const response = await finalModel.invoke([
    new SystemMessage("Rewrite this in the voice of Al Roker"),
    new HumanMessage({ content: lastAIMessage.content }),
  ]);
  // MessagesAnnotation allows you to overwrite messages from the agent
  // by returning a message with the same id
  response.id = lastAIMessage.id;
  return { messages: [response] };
};

const toolNode = new ToolNode<typeof MessagesAnnotation.State>(tools);

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  // add a separate final node
  .addNode("final", callFinalModel)
  .addEdge("__start__", "agent")
  // Third parameter is optional and only here to draw a diagram of the graph
  .addConditionalEdges("agent", shouldContinue, {
    tools: "tools",
    final: "final",
  })
  .addEdge("tools", "agent")
  .addEdge("final", "__end__")
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
const inputs = { messages: [new HumanMessage("What's the weather in nyc?")] };

const eventStream = await graph.streamEvents(inputs, { version: "v2" });

for await (const { event, tags, data } of eventStream) {
  if (event === "on_chat_model_stream" && tags.includes("final_node")) {
    if (data.chunk.content) {
      // Empty content in the context of OpenAI or Anthropic usually means
      // that the model is asking for a tool to be invoked.
      // So we only print non-empty content
      console.log(data.chunk.content, "|");
    }
  }
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__..."
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT = "ReAct Agent: LangGraphJS";
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const model = new ChatOpenAI({
  model: "gpt-4o",
});

const getWeather = tool(
  (input) => {
    if (
      ["sf", "san francisco", "san francisco, ca"].includes(
        input.location.toLowerCase()
      )
    ) {
      return "It's 60 degrees and foggy.";
    } else {
      return "It's 90 degrees and sunny.";
    }
  },
  {
    name: "get_weather",
    description: "Call to get the current weather.",
    schema: z.object({
      location: z.string().describe("Location to get the weather for."),
    }),
  }
);

const agent = createReactAgent({ llm: model, tools: [getWeather] });
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await agent.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
let inputs = {
  messages: [{ role: "user", content: "what is the weather in SF?" }],
};

let stream = await agent.stream(inputs, {
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  } else if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  } else {
    console.log(msg);
  }
  console.log("-----\n");
}
```

```typescript
inputs = { messages: [{ role: "user", content: "who built you?" }] };

stream = await agent.stream(inputs, {
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  } else if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  } else {
    console.log(msg);
  }
  console.log("-----\n");
}
```

```typescript
import {
  StateGraph,
  MessagesAnnotation,
  LangGraphRunnableConfig,
} from "@langchain/langgraph";

const myNode = async (
  _state: typeof MessagesAnnotation.State,
  config: LangGraphRunnableConfig
) => {
  const chunks = [
    "Four",
    "score",
    "and",
    "seven",
    "years",
    "ago",
    "our",
    "fathers",
    "...",
  ];
  for (const chunk of chunks) {
    // write the chunk to be streamed using streamMode=custom
    // Only populated if one of the passed stream modes is "custom".
    config.writer?.(chunk);
  }
  return {
    messages: [
      {
        role: "assistant",
        content: chunks.join(" "),
      },
    ],
  };
};

const graph = new StateGraph(MessagesAnnotation)
  .addNode("model", myNode)
  .addEdge("__start__", "model")
  .compile();
```

```typescript
const inputs = [
  {
    role: "user",
    content: "What are you thinking about?",
  },
];

const stream = await graph.stream(
  { messages: inputs },
  { streamMode: "custom" }
);

for await (const chunk of stream) {
  console.log(chunk);
}
```

```typescript
const streamMultiple = await graph.stream(
  { messages: inputs },
  { streamMode: ["custom", "updates"] }
);

for await (const chunk of streamMultiple) {
  console.log(chunk);
}
```

```typescript
import { dispatchCustomEvent } from "@langchain/core/callbacks/dispatch";

const graphNode = async (_state: typeof MessagesAnnotation.State) => {
  const chunks = [
    "Four",
    "score",
    "and",
    "seven",
    "years",
    "ago",
    "our",
    "fathers",
    "...",
  ];
  for (const chunk of chunks) {
    await dispatchCustomEvent("my_custom_event", { chunk });
  }
  return {
    messages: [
      {
        role: "assistant",
        content: chunks.join(" "),
      },
    ],
  };
};

const graphWithDispatch = new StateGraph(MessagesAnnotation)
  .addNode("model", graphNode)
  .addEdge("__start__", "model")
  .compile();
```

```typescript
const eventStream = await graphWithDispatch.streamEvents(
  {
    messages: [
      {
        role: "user",
        content: "What are you thinking about?",
      },
    ],
  },
  {
    version: "v2",
  }
);

for await (const { event, name, data } of eventStream) {
  if (event === "on_custom_event" && name === "my_custom_event") {
    console.log(`${data.chunk}|`);
  }
}
```

```typescript
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

```typescript
import { ChatAnthropic } from "@langchain/anthropic";

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
});
```

```typescript
import type { BaseMessage, BaseMessageLike } from "@langchain/core/messages";
import {
  addMessages,
  entrypoint,
  task,
  getPreviousState,
  MemorySaver,
} from "@langchain/langgraph";

const callModel = task("callModel", async (messages: BaseMessageLike[]) => {
  const response = model.invoke(messages);
  return response;
});

const checkpointer = new MemorySaver();

const workflow = entrypoint(
  {
    name: "workflow",
    checkpointer,
  },
  async (inputs: BaseMessageLike[]) => {
    const previous = getPreviousState<BaseMessage>() ?? [];
    const messages = addMessages(previous, inputs);
    const response = await callModel(messages);
    return entrypoint.final({
      value: response,
      save: addMessages(messages, response),
    });
  }
);
```

```typescript
const config = {
  configurable: { thread_id: "1" },
  streamMode: "values" as const,
};
const inputMessage = { role: "user", content: "hi! I'm bob" };

const stream = await workflow.stream([inputMessage], config);

for await (const chunk of stream) {
  console.log("=".repeat(30), `${chunk.getType()} message`, "=".repeat(30));
  console.log(chunk.content);
}
```

```typescript
const followupStream = await workflow.stream(
  [{ role: "user", content: "what's my name?" }],
  config
);

for await (const chunk of followupStream) {
  console.log("=".repeat(30), `${chunk.getType()} message`, "=".repeat(30));
  console.log(chunk.content);
}
```

```typescript
const newStream = await workflow.stream(
  [{ role: "user", content: "what's my name?" }],
  {
    configurable: {
      thread_id: "2",
    },
    streamMode: "values",
  }
);

for await (const chunk of newStream) {
  console.log("=".repeat(30), `${chunk.getType()} message`, "=".repeat(30));
  console.log(chunk.content);
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__..."
// process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = "true";
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_PROJECT =
  "ReAct Agent with human-in-the-loop: LangGraphJS";
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";

const model = new ChatOpenAI({
  model: "gpt-4o",
});

const getWeather = tool(
  (input) => {
    if (["sf", "san francisco"].includes(input.location.toLowerCase())) {
      return "It's always sunny in sf";
    } else if (
      ["nyc", "new york city"].includes(input.location.toLowerCase())
    ) {
      return "It might be cloudy in nyc";
    } else {
      throw new Error("Unknown Location");
    }
  },
  {
    name: "get_weather",
    description: "Call to get the current weather in a given location.",
    schema: z.object({
      location: z.string().describe("Location to get the weather for."),
    }),
  }
);

// Here we only save in-memory
const memory = new MemorySaver();

const agent = createReactAgent({
  llm: model,
  tools: [getWeather],
  interruptBefore: ["tools"],
  checkpointSaver: memory,
});
```

```typescript
let inputs = {
  messages: [
    { role: "user", content: "what is the weather in SF california?" },
  ],
};
let config = { configurable: { thread_id: "1" } };

let stream = await agent.stream(inputs, {
  ...config,
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  }
  if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  }
  console.log("-----\n");
}
```

```typescript
const state = await agent.getState(config);
console.log(state.next);
```

```typescript
stream = await agent.stream(null, {
  ...config,
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  }
  if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  }
  console.log("-----\n");
}
```

```typescript
// First, lets get the current state
const currentState = await agent.getState(config);

// Let's now get the last message in the state
// This is the one with the tool calls that we want to update
let lastMessage = currentState.values.messages.at(-1);

// Let's now update the args for that tool call
lastMessage.tool_calls[0].args = { location: "San Francisco" };

// Let's now call `updateState` to pass in this message in the `messages` key
// This will get treated as any other update to the state
// It will get passed to the reducer function for the `messages` key
// That reducer function will use the ID of the message to update it
// It's important that it has the right ID! Otherwise it would get appended
// as a new message
await agent.updateState(config, { messages: lastMessage });
```

```typescript
stream = await agent.stream(null, {
  ...config,
  streamMode: "values",
});

for await (const { messages } of stream) {
  let msg = messages?.at(-1);
  if (msg?.content) {
    console.log(msg.content);
  }
  if (msg?.tool_calls?.length > 0) {
    console.log(msg.tool_calls);
  }
  console.log("-----\n");
}
```

```typescript
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});
```

```typescript
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { getContextVariable } from "@langchain/core/context";
import { LangGraphRunnableConfig } from "@langchain/langgraph";

const updateFavoritePets = tool(
  async (input, config: LangGraphRunnableConfig) => {
    // Some arguments are populated by the LLM; these are included in the schema below
    const { pets } = input;
    // Fetch a context variable named "currentState".
    // We must set this variable explicitly in each node that calls this tool.
    const currentState = getContextVariable("currentState");
    // Other information (such as a UserID) are most easily provided via the config
    // This is set when when invoking or streaming the graph
    const userId = config.configurable?.userId;
    // LangGraph's managed key-value store is also accessible from the config
    const store = config.store;
    await store.put([userId, "pets"], "names", pets);
    // Store the initial input message from the user as a note.
    // Using the same key will override previous values - you could
    // use something different if you wanted to store many interactions.
    await store.put(
      [userId, "pets"],
      "context",
      currentState.messages[0].content
    );

    return "update_favorite_pets called.";
  },
  {
    // The LLM "sees" the following schema:
    name: "update_favorite_pets",
    description: "add to the list of favorite pets.",
    schema: z.object({
      pets: z.array(z.string()),
    }),
  }
);
```

```typescript
import { zodToJsonSchema } from "zod-to-json-schema";

console.log(zodToJsonSchema(updateFavoritePets.schema));
```

```typescript
const getFavoritePets = tool(
  async (_, config: LangGraphRunnableConfig) => {
    const userId = config.configurable?.userId;
    // LangGraph's managed key-value store is also accessible via the config
    const store = config.store;
    const petNames = await store.get([userId, "pets"], "names");
    const context = await store.get([userId, "pets"], "context");
    return JSON.stringify({
      pets: petNames.value,
      context: context.value,
    });
  },
  {
    // The LLM "sees" the following schema:
    name: "get_favorite_pets",
    description: "retrieve the list of favorite pets for the given user.",
    schema: z.object({}),
  }
);
```

```typescript
import {
  END,
  START,
  StateGraph,
  MemorySaver,
  InMemoryStore,
} from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";

import { setContextVariable } from "@langchain/core/context";

const model = new ChatOpenAI({ model: "gpt-4o" });

const tools = [getFavoritePets, updateFavoritePets];

const routeMessage = (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages.at(-1) as AIMessage | undefined;
  // If no tools are called, we can finish (respond to the user)
  if (!lastMessage?.tool_calls?.length) {
    return END;
  }
  // Otherwise if there is, we continue and call the tools
  return "tools";
};

const callModel = async (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const modelWithTools = model.bindTools(tools);
  const responseMessage = await modelWithTools.invoke([
    {
      role: "system",
      content:
        "You are a personal assistant. Store any preferences the user tells you about.",
    },
    ...messages,
  ]);
  return { messages: [responseMessage] };
};

const toolNodeWithGraphState = async (state: typeof StateAnnotation.State) => {
  // We set a context variable before invoking the tool node and running our tool.
  setContextVariable("currentState", state);
  const toolNodeWithConfig = new ToolNode(tools);
  return toolNodeWithConfig.invoke(state);
};

const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNodeWithGraphState)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", routeMessage)
  .addEdge("tools", "agent");

const memory = new MemorySaver();
const store = new InMemoryStore();

const graph = workflow.compile({ checkpointer: memory, store: store });
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
let inputs = {
  messages: [
    {
      role: "user",
      content: "My favorite pet is a terrier. I saw a cute one on Twitter.",
    },
  ],
};
let config = {
  configurable: {
    thread_id: "1",
    userId: "a-user",
  },
};
let stream = await graph.stream(inputs, config);

for await (const chunk of stream) {
  for (const [node, values] of Object.entries(chunk)) {
    console.log(`Output from node: ${node}`);
    console.log("---");
    console.log(values);
    console.log("\n====\n");
  }
}
```

```typescript
inputs = {
  messages: [
    {
      role: "user",
      content:
        "What're my favorite pets and what did I say when I told you about them?",
    },
  ],
};
config = {
  configurable: {
    thread_id: "2", // New thread ID, so the conversation history isn't present.
    userId: "a-user",
  },
};

stream = await graph.stream(inputs, {
  ...config,
});

for await (const chunk of stream) {
  for (const [node, values] of Object.entries(chunk)) {
    console.log(`Output from node: ${node}`);
    console.log("---");
    console.log(values);
    console.log("\n====\n");
  }
}
```

```typescript
function generateTools(state: typeof StateAnnotation.State) {
  const updateFavoritePets = tool(
    async (input, config: LangGraphRunnableConfig) => {
      // Some arguments are populated by the LLM; these are included in the schema below
      const { pets } = input;
      // Others (such as a UserID) are best provided via the config
      // This is set when when invoking or streaming the graph
      const userId = config.configurable?.userId;
      // LangGraph's managed key-value store is also accessible via the config
      const store = config.store;
      await store.put([userId, "pets"], "names", pets);
      await store.put([userId, "pets"], "context", {
        content: state.messages[0].content,
      });

      return "update_favorite_pets called.";
    },
    {
      // The LLM "sees" the following schema:
      name: "update_favorite_pets",
      description: "add to the list of favorite pets.",
      schema: z.object({
        pets: z.array(z.string()),
      }),
    }
  );
  return [updateFavoritePets];
}
```

```typescript
const toolNodeWithClosure = async (state: typeof StateAnnotation.State) => {
  // We fetch the tools any time this node is reached to
  // form a closure and let it access the latest messages
  const tools = generateTools(state);
  const toolNodeWithConfig = new ToolNode(tools);
  return toolNodeWithConfig.invoke(state);
};
```

```typescript
function human(state: typeof MessagesAnnotation.State): Command {
  const userInput: string = interrupt("Ready for user input.");

  // Determine the active agent
  const activeAgent = ...;

  return new Command({
    update: {
      messages: [{
        role: "human",
        content: userInput,
      }]
    },
    goto: activeAgent,
  });
}

function agent(state: typeof MessagesAnnotation.State): Command {
  // The condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
  const goto = getNextAgent(...); // 'agent' / 'anotherAgent'

  if (goto) {
    return new Command({
      goto,
      update: { myStateKey: "myStateValue" }
    });
  }

  return new Command({ goto: "human" });
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "ls__...";
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
  MemorySaver,
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
    response: z
      .string()
      .describe(
        "A human readable response to the original question. Does not need to be a final response. Will be streamed back to the user."
      ),
    goto: z
      .enum(["finish", ...targetAgentNodes])
      .describe(
        "The next agent to call, or 'finish' if the user's query has been resolved. Must be one of the specified values."
      ),
  });
  return model
    .withStructuredOutput(outputSchema, { name: "Response" })
    .invoke(messages);
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

  const messages = [
    { role: "system", content: systemPrompt },
    ...state.messages,
  ] as BaseMessage[];
  const targetAgentNodes = ["sightseeingAdvisor", "hotelAdvisor"];
  const response = await callLlm(messages, targetAgentNodes);
  const aiMsg = {
    role: "ai",
    content: response.response,
    name: "travelAdvisor",
  };

  let goto = response.goto;
  if (goto === "finish") {
    goto = "human";
  }

  return new Command({ goto, update: { messages: [aiMsg] } });
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

  const messages = [
    { role: "system", content: systemPrompt },
    ...state.messages,
  ] as BaseMessage[];
  const targetAgentNodes = ["travelAdvisor", "hotelAdvisor"];
  const response = await callLlm(messages, targetAgentNodes);
  const aiMsg = {
    role: "ai",
    content: response.response,
    name: "sightseeingAdvisor",
  };

  let goto = response.goto;
  if (goto === "finish") {
    goto = "human";
  }

  return new Command({ goto, update: { messages: [aiMsg] } });
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

  const messages = [
    { role: "system", content: systemPrompt },
    ...state.messages,
  ] as BaseMessage[];
  const targetAgentNodes = ["travelAdvisor", "sightseeingAdvisor"];
  const response = await callLlm(messages, targetAgentNodes);
  const aiMsg = {
    role: "ai",
    content: response.response,
    name: "hotelAdvisor",
  };

  let goto = response.goto;
  if (goto === "finish") {
    goto = "human";
  }

  return new Command({ goto, update: { messages: [aiMsg] } });
}

function humanNode(state: typeof MessagesAnnotation.State): Command {
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
      messages: [
        {
          role: "human",
          content: userInput,
        },
      ],
    },
  });
}

const builder = new StateGraph(MessagesAnnotation)
  .addNode("travelAdvisor", travelAdvisor, {
    ends: ["sightseeingAdvisor", "hotelAdvisor"],
  })
  .addNode("sightseeingAdvisor", sightseeingAdvisor, {
    ends: ["human", "travelAdvisor", "hotelAdvisor"],
  })
  .addNode("hotelAdvisor", hotelAdvisor, {
    ends: ["human", "travelAdvisor", "sightseeingAdvisor"],
  })
  // This adds a node to collect human input, which will route
  // back to the active agent.
  .addNode("human", humanNode, {
    ends: ["hotelAdvisor", "sightseeingAdvisor", "travelAdvisor", "human"],
  })
  // We'll always start with a general travel advisor.
  .addEdge(START, "travelAdvisor");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
import { Command } from "@langchain/langgraph";
import { v4 as uuidv4 } from "uuid";

const threadConfig = {
  configurable: { thread_id: uuidv4() },
  streamMode: "values" as const,
};

const inputs = [
  // 1st round of conversation
  {
    messages: [
      { role: "user", content: "i wanna go somewhere warm in the caribbean" },
    ],
  },
  // Since we're using `interrupt`, we'll need to resume using the Command primitive.
  // 2nd round of conversation
  new Command({
    resume:
      "could you recommend a nice hotel in one of the areas and tell me which area it is.",
  }),
  // Third round of conversation
  new Command({
    resume: "could you recommend something to do near the hotel?",
  }),
];

let iter = 0;
for await (const userInput of inputs) {
  iter += 1;
  console.log(`\n--- Conversation Turn ${iter} ---\n`);
  console.log(`User: ${JSON.stringify(userInput)}\n`);

  for await (const update of await graph.stream(userInput, threadConfig)) {
    const lastMessage = update.messages?.at(-1);
    if (lastMessage && lastMessage.getType() === "ai") {
      console.log(`${lastMessage.name}: ${lastMessage.content}`);
    }
  }
}
```

```typescript
process.env.OPENAI_API_KEY = "YOUR_API_KEY";
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const getWeather = tool(
  async ({ location }) => {
    // This is a placeholder for the actual implementation
    const lowercaseLocation = location.toLowerCase();
    if (
      lowercaseLocation.includes("sf") ||
      lowercaseLocation.includes("san francisco")
    ) {
      return "It's sunny!";
    } else if (lowercaseLocation.includes("boston")) {
      return "It's rainy!";
    } else {
      return `I am not sure what the weather is in ${location}`;
    }
  },
  {
    name: "getWeather",
    schema: z.object({
      location: z.string().describe("Location to get the weather for"),
    }),
    description: "Call to get the weather from a specific location.",
  }
);

const tools = [getWeather];
```

```typescript
import {
  type BaseMessageLike,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { type ToolCall } from "@langchain/core/messages/tool";
import { task } from "@langchain/langgraph";

const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));

const callModel = task("callModel", async (messages: BaseMessageLike[]) => {
  const response = await model.bindTools(tools).invoke(messages);
  return response;
});

const callTool = task(
  "callTool",
  async (toolCall: ToolCall): Promise<AIMessage> => {
    const tool = toolsByName[toolCall.name];
    const observation = await tool.invoke(toolCall.args);
    return new ToolMessage({ content: observation, tool_call_id: toolCall.id });
    // Can also pass toolCall directly into the tool to return a ToolMessage
    // return tool.invoke(toolCall);
  }
);
```

```typescript
import { interrupt } from "@langchain/langgraph";

function reviewToolCall(toolCall: ToolCall): ToolCall | ToolMessage {
  // Interrupt for human review
  const humanReview = interrupt({
    question: "Is this correct?",
    tool_call: toolCall,
  });

  const { action, data } = humanReview;

  if (action === "continue") {
    return toolCall;
  } else if (action === "update") {
    return {
      ...toolCall,
      args: data,
    };
  } else if (action === "feedback") {
    return new ToolMessage({
      content: data,
      name: toolCall.name,
      tool_call_id: toolCall.id,
    });
  }
  throw new Error(`Unsupported review action: ${action}`);
}
```

```typescript
import {
  MemorySaver,
  addMessages,
  entrypoint,
  getPreviousState,
} from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const agent = entrypoint(
  {
    checkpointer,
    name: "agent",
  },
  async (messages: BaseMessageLike[]) => {
    const previous = getPreviousState<BaseMessageLike[]>() ?? [];
    let currentMessages = addMessages(previous, messages);
    let llmResponse = await callModel(currentMessages);
    while (true) {
      if (!llmResponse.tool_calls?.length) {
        break;
      }
      // Review tool calls
      const toolResults: ToolMessage[] = [];
      const toolCalls: ToolCall[] = [];

      for (let i = 0; i < llmResponse.tool_calls.length; i++) {
        const review = await reviewToolCall(llmResponse.tool_calls[i]);
        if (review instanceof ToolMessage) {
          toolResults.push(review);
        } else {
          // is a validated tool call
          toolCalls.push(review);
          if (review !== llmResponse.tool_calls[i]) {
            llmResponse.tool_calls[i] = review;
          }
        }
      }
      // Execute remaining tool calls
      const remainingToolResults = await Promise.all(
        toolCalls.map((toolCall) => callTool(toolCall))
      );

      // Append to message list
      currentMessages = addMessages(currentMessages, [
        llmResponse,
        ...toolResults,
        ...remainingToolResults,
      ]);

      // Call model again
      llmResponse = await callModel(currentMessages);
    }
    // Generate final response
    currentMessages = addMessages(currentMessages, llmResponse);
    return entrypoint.final({
      value: llmResponse,
      save: currentMessages,
    });
  }
);
```

```typescript
import { BaseMessage, isAIMessage } from "@langchain/core/messages";

const prettyPrintMessage = (message: BaseMessage) => {
  console.log("=".repeat(30), `${message.getType()} message`, "=".repeat(30));
  console.log(message.content);
  if (isAIMessage(message) && message.tool_calls?.length) {
    console.log(JSON.stringify(message.tool_calls, null, 2));
  }
};

const printStep = (step: Record<string, any>) => {
  if (step.__metadata__?.cached) {
    return;
  }
  for (const [taskName, result] of Object.entries(step)) {
    if (taskName === "agent") {
      continue; // just stream from tasks
    }

    console.log(`\n${taskName}:`);
    if (taskName === "__interrupt__" || taskName === "reviewToolCall") {
      console.log(JSON.stringify(result, null, 2));
    } else {
      prettyPrintMessage(result);
    }
  }
};
```

```typescript
const config = {
  configurable: {
    thread_id: "1",
  },
};

const userMessage = {
  role: "user",
  content: "What's the weather in san francisco?",
};
console.log(userMessage);

const stream = await agent.stream([userMessage], config);

for await (const step of stream) {
  printStep(step);
}
```

```typescript hl_lines="3 4 5 6 7"
import { Command } from "@langchain/langgraph";

const humanInput = new Command({
  resume: {
    action: "continue",
  },
});

const resumedStream = await agent.stream(humanInput, config);

for await (const step of resumedStream) {
  printStep(step);
}
```

```typescript
const config2 = {
  configurable: {
    thread_id: "2",
  },
};

const userMessage2 = {
  role: "user",
  content: "What's the weather in san francisco?",
};

console.log(userMessage2);

const stream2 = await agent.stream([userMessage2], config2);

for await (const step of stream2) {
  printStep(step);
}
```

```typescript hl_lines="1 2 3 4 5 6"
const humanInput2 = new Command({
  resume: {
    action: "update",
    data: { location: "SF, CA" },
  },
});

const resumedStream2 = await agent.stream(humanInput2, config2);

for await (const step of resumedStream2) {
  printStep(step);
}
```

```typescript
const config3 = {
  configurable: {
    thread_id: "3",
  },
};

const userMessage3 = {
  role: "user",
  content: "What's the weather in san francisco?",
};

console.log(userMessage3);

const stream3 = await agent.stream([userMessage3], config3);

for await (const step of stream3) {
  printStep(step);
}
```

```typescript hl_lines="1 2 3 4 5 6"
const humanInput3 = new Command({
  resume: {
    action: "feedback",
    data: "Please format as <City>, <State>.",
  },
});

const resumedStream3 = await agent.stream(humanInput3, config3);

for await (const step of resumedStream3) {
  printStep(step);
}
```

```typescript hl_lines="1 2 3 4 5"
const continueCommand = new Command({
  resume: {
    action: "continue",
  },
});

const continueStream = await agent.stream(continueCommand, config3);

for await (const step of continueStream) {
  printStep(step);
}
```

```typescript
import { BaseMessage } from "@langchain/core/messages";
import { Annotation } from "@langchain/langgraph";

const GraphAnnotation = Annotation.Root({
  // Define a 'messages' channel to store an array of BaseMessage objects
  messages: Annotation<BaseMessage[]>({
    // Reducer function: Combines the current state with new messages
    reducer: (currentState, updateValue) => currentState.concat(updateValue),
    // Default function: Initialize the channel with an empty array
    default: () => [],
  }),
});
```

```typescript
const QuestionAnswerAnnotation = Annotation.Root({
  question: Annotation<string>,
  answer: Annotation<string>,
});
```

```typescript
type QuestionAnswerAnnotationType = typeof QuestionAnswerAnnotation.State;
```

```typescript
type QuestionAnswerAnnotationType = {
  question: string;
  answer: string;
};
```

```typescript
const MergedAnnotation = Annotation.Root({
  ...QuestionAnswerAnnotation.spec,
  ...GraphAnnotation.spec,
});
```

```typescript
type MergedAnnotation = {
  messages: BaseMessage[];
  question: string;
  answer: string;
};
```

```typescript
import { StateGraph } from "@langchain/langgraph";

const workflow = new StateGraph(MergedAnnotation);
```

```typescript
import { StateGraph } from "@langchain/langgraph";

interface WorkflowChannelsState {
  messages: BaseMessage[];
  question: string;
  answer: string;
}

const workflowWithChannels = new StateGraph<WorkflowChannelsState>({
  channels: {
    messages: {
      reducer: (currentState, updateValue) => currentState.concat(updateValue),
      default: () => [],
    },
    question: null,
    answer: null,
  },
});
```

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import { BaseMessage, AIMessage } from "@langchain/core/messages";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { z } from "zod";

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

const memory = new MemorySaver();

const searchTool = tool(
  (_): string => {
    // This is a placeholder for the actual implementation
    // Don't let the LLM know this though 😊
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈.";
  },
  {
    name: "search",
    description: "Call to surf the web.",
    schema: z.object({
      query: z.string(),
    }),
  }
);

const tools = [searchTool];
const toolNode = new ToolNode<typeof AgentState.State>(tools);
const model = new ChatAnthropic({ model: "claude-3-haiku-20240307" });
const boundModel = model.bindTools(tools);

function shouldContinue(state: typeof AgentState.State): "action" | typeof END {
  const lastMessage = state.messages.at(-1);
  // If there is no function call, then we finish
  if (lastMessage && !(lastMessage as AIMessage).tool_calls?.length) {
    return END;
  }
  // Otherwise if there is, we continue
  return "action";
}

// Define the function that calls the model
async function callModel(state: typeof AgentState.State) {
  const response = await model.invoke(state.messages);
  // We return an object, because this will get merged with the existing state
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(AgentState)
  // Define the two nodes we will cycle between
  .addNode("agent", callModel)
  .addNode("action", toolNode)
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `agent`.
    // This means these are the edges taken after the `agent` node is called.
    "agent",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue
  )
  // We now add a normal edge from `action` to `agent`.
  // This means that after `action` is called, `agent` node is called next.
  .addEdge("action", "agent")
  // Set the entrypoint as `agent`
  // This means that this node is the first one called
  .addEdge(START, "agent");

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const app = workflow.compile({
  checkpointer: memory,
});
```

```typescript
import { HumanMessage } from "@langchain/core/messages";

const config = {
  configurable: { thread_id: "2" },
  streamMode: "values" as const,
};

const inputMessage = new HumanMessage("hi! I'm bob");
for await (const event of await app.stream(
  {
    messages: [inputMessage],
  },
  config
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}

console.log(
  "\n\n================================= END =================================\n\n"
);

const inputMessage2 = new HumanMessage("what's my name?");
for await (const event of await app.stream(
  {
    messages: [inputMessage2],
  },
  config
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (2) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import { BaseMessage, AIMessage } from "@langchain/core/messages";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { z } from "zod";

const MessageFilteringAgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

const messageFilteringMemory = new MemorySaver();

const messageFilteringSearchTool = tool(
  (_): string => {
    // This is a placeholder for the actual implementation
    // Don't let the LLM know this though 😊
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈.";
  },
  {
    name: "search",
    description: "Call to surf the web.",
    schema: z.object({
      query: z.string(),
    }),
  }
);

// We can re-use the same search tool as above as we don't need to change it for this example.
const messageFilteringTools = [messageFilteringSearchTool];
const messageFilteringToolNode = new ToolNode<
  typeof MessageFilteringAgentState.State
>(messageFilteringTools);
const messageFilteringModel = new ChatAnthropic({
  model: "claude-3-haiku-20240307",
});
const boundMessageFilteringModel = messageFilteringModel.bindTools(
  messageFilteringTools
);

async function shouldContinueMessageFiltering(
  state: typeof MessageFilteringAgentState.State
): Promise<"action" | typeof END> {
  const lastMessage = state.messages.at(-1);
  // If there is no function call, then we finish
  if (lastMessage && !(lastMessage as AIMessage).tool_calls?.length) {
    return END;
  }
  // Otherwise if there is, we continue
  return "action";
}

const filterMessages = (messages: BaseMessage[]): BaseMessage[] => {
  // This is very simple helper function which only ever uses the last message
  return messages.slice(-1);
};

// Define the function that calls the model
async function callModelMessageFiltering(
  state: typeof MessageFilteringAgentState.State
) {
  const response = await boundMessageFilteringModel.invoke(
    filterMessages(state.messages)
  );
  // We return an object, because this will get merged with the existing state
  return { messages: [response] };
}

// Define a new graph
const messageFilteringWorkflow = new StateGraph(MessageFilteringAgentState)
  // Define the two nodes we will cycle between
  .addNode("agent", callModelMessageFiltering)
  .addNode("action", messageFilteringToolNode)
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `agent`.
    // This means these are the edges taken after the `agent` node is called.
    "agent",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinueMessageFiltering
  )
  // We now add a normal edge from `action` to `agent`.
  // This means that after `action` is called, `agent` node is called next.
  .addEdge("action", "agent")
  // Set the entrypoint as `agent`
  // This means that this node is the first one called
  .addEdge(START, "agent");

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const messageFilteringApp = messageFilteringWorkflow.compile({
  checkpointer: messageFilteringMemory,
});
```

```typescript
import { HumanMessage } from "@langchain/core/messages";

const messageFilteringConfig = {
  configurable: { thread_id: "2" },
  streamMode: "values" as const,
};

const messageFilteringInput = new HumanMessage("hi! I'm bob");
for await (const event of await messageFilteringApp.stream(
  {
    messages: [messageFilteringInput],
  },
  messageFilteringConfig
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (1) =================================`
  );
  console.log(recentMsg.content);
}

console.log(
  "\n\n================================= END =================================\n\n"
);

const messageFilteringInput2 = new HumanMessage("what's my name?");
for await (const event of await messageFilteringApp.stream(
  {
    messages: [messageFilteringInput2],
  },
  messageFilteringConfig
)) {
  const recentMsg = event.messages.at(-1);
  console.log(
    `================================ ${recentMsg.getType()} Message (2) =================================`
  );
  console.log(recentMsg.content);
}
```

```typescript
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

```typescript
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatAnthropic } from "@langchain/anthropic";

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
  temperature: 0,
});

const getItems = tool(
  async (input, config) => {
    const template = ChatPromptTemplate.fromMessages([
      [
        "human",
        "Can you tell me what kind of items i might find in the following place: '{place}'. " +
          "List at least 3 such items separating them by a comma. And include a brief description of each item..",
      ],
    ]);

    const modelWithConfig = model.withConfig({
      runName: "Get Items LLM",
      tags: ["tool_llm"],
    });

    const chain = template.pipe(modelWithConfig);
    const result = await chain.invoke(input, config);
    return result.content;
  },
  {
    name: "get_items",
    description: "Use this tool to look up which items are in the given place.",
    schema: z.object({
      place: z.string().describe("The place to look up items for. E.g 'shelf'"),
    }),
  }
);
```

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const agent = createReactAgent({
  llm: model,
  tools: [getItems],
});
```

```typescript
let finalEvent;

for await (const event of agent.streamEvents(
  {
    messages: [
      [
        "human",
        "what items are on the shelf? You should call the get_items tool.",
      ],
    ],
  },
  {
    version: "v2",
  },
  {
    includeTags: ["tool_llm"],
  }
)) {
  if ("chunk" in event.data) {
    console.dir({
      type: event.data.chunk.getType(),
      content: event.data.chunk.content,
    });
  }
  finalEvent = event;
}
```

```typescript
const finalMessage = finalEvent?.data.output;
console.dir(
  {
    type: finalMessage.getType(),
    content: finalMessage.content,
    tool_calls: finalMessage.tool_calls,
  },
  { depth: null }
);
```

```typescript
import OpenAI from "openai";

const openaiClient = new OpenAI({});

const toolSchema: OpenAI.ChatCompletionTool = {
  type: "function",
  function: {
    name: "get_items",
    description: "Use this tool to look up which items are in the given place.",
    parameters: {
      type: "object",
      properties: {
        place: {
          type: "string",
        },
      },
      required: ["place"],
    },
  },
};
```

```typescript
import { dispatchCustomEvent } from "@langchain/core/callbacks/dispatch";
import { wrapOpenAI } from "langsmith/wrappers/openai";
import { Annotation } from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
  messages: Annotation<OpenAI.ChatCompletionMessageParam[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

// If using LangSmith, use "wrapOpenAI" on the whole client or
// "traceable" to wrap a single method for nicer tracing:
// https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code
const wrappedClient = wrapOpenAI(openaiClient);

const callModel = async (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const stream = await wrappedClient.chat.completions.create({
    messages,
    model: "gpt-4o-mini",
    tools: [toolSchema],
    stream: true,
  });
  let responseContent = "";
  let role: string = "assistant";
  let toolCallId: string | undefined;
  let toolCallName: string | undefined;
  let toolCallArgs = "";
  for await (const chunk of stream) {
    const delta = chunk.choices[0].delta;
    if (delta.role !== undefined) {
      role = delta.role;
    }
    if (delta.content) {
      responseContent += delta.content;
      await dispatchCustomEvent("streamed_token", {
        content: delta.content,
      });
    }
    if (delta.tool_calls !== undefined && delta.tool_calls.length > 0) {
      // note: for simplicity we're only handling a single tool call here
      const toolCall = delta.tool_calls[0];
      if (toolCall.function?.name !== undefined) {
        toolCallName = toolCall.function.name;
      }
      if (toolCall.id !== undefined) {
        toolCallId = toolCall.id;
      }
      await dispatchCustomEvent("streamed_tool_call_chunk", toolCall);
      toolCallArgs += toolCall.function?.arguments ?? "";
    }
  }
  let finalToolCalls;
  if (toolCallName !== undefined && toolCallId !== undefined) {
    finalToolCalls = [
      {
        id: toolCallId,
        function: {
          name: toolCallName,
          arguments: toolCallArgs,
        },
        type: "function" as const,
      },
    ];
  }

  const responseMessage = {
    role: role as any,
    content: responseContent,
    tool_calls: finalToolCalls,
  };
  return { messages: [responseMessage] };
};
```

```typescript
const getItems = async ({ place }: { place: string }) => {
  if (place.toLowerCase().includes("bed")) {
    // For under the bed
    return "socks, shoes and dust bunnies";
  } else if (place.toLowerCase().includes("shelf")) {
    // For 'shelf'
    return "books, pencils and pictures";
  } else {
    // if the agent decides to ask about a different place
    return "cat snacks";
  }
};

const callTools = async (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const mostRecentMessage = messages.at(-1);
  const toolCalls = (
    mostRecentMessage as OpenAI.ChatCompletionAssistantMessageParam
  ).tool_calls;
  if (toolCalls === undefined || toolCalls.length === 0) {
    throw new Error("No tool calls passed to node.");
  }
  const toolNameMap = {
    get_items: getItems,
  };
  const functionName = toolCalls[0].function.name;
  const functionArguments = JSON.parse(toolCalls[0].function.arguments);
  const response = await toolNameMap[functionName](functionArguments);
  const toolMessage = {
    tool_call_id: toolCalls[0].id,
    role: "tool" as const,
    name: functionName,
    content: response,
  };
  return { messages: [toolMessage] };
};
```

```typescript
import { StateGraph } from "@langchain/langgraph";
import OpenAI from "openai";

// We can reuse the same `GraphState` from above as it has not changed.
const shouldContinue = (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages.at(
    -1
  ) as OpenAI.ChatCompletionAssistantMessageParam;
  if (
    lastMessage?.tool_calls !== undefined &&
    lastMessage?.tool_calls.length > 0
  ) {
    return "tools";
  }
  return "__end__";
};

const graph = new StateGraph(StateAnnotation)
  .addNode("model", callModel)
  .addNode("tools", callTools)
  .addEdge("__start__", "model")
  .addConditionalEdges("model", shouldContinue, {
    tools: "tools",
    __end__: "__end__",
  })
  .addEdge("tools", "model")
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
const eventStream = await graph.streamEvents(
  { messages: [{ role: "user", content: "what's in the bedroom?" }] },
  { version: "v2" }
);

for await (const { event, name, data } of eventStream) {
  if (event === "on_custom_event") {
    console.log(name, data);
  }
}
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGCHAIN_API_KEY = "lsv2__...";
// process.env.ANTHROPIC_API_KEY = "your api key";
// process.env.LANGCHAIN_TRACING_V2 = "true";
// process.env.LANGCHAIN_PROJECT = "Cross-thread persistence: LangGraphJS";
```

```typescript
import { InMemoryStore } from "@langchain/langgraph";

const inMemoryStore = new InMemoryStore();
```

```typescript
import { v4 as uuidv4 } from "uuid";
import { ChatAnthropic } from "@langchain/anthropic";
import { BaseMessage } from "@langchain/core/messages";
import {
  Annotation,
  StateGraph,
  START,
  MemorySaver,
  LangGraphRunnableConfig,
  messagesStateReducer,
} from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
});

const model = new ChatAnthropic({ modelName: "claude-3-5-sonnet-20240620" });

// NOTE: we're passing the Store param to the node --
// this is the Store we compile the graph with
const callModel = async (
  state: typeof StateAnnotation.State,
  config: LangGraphRunnableConfig
): Promise<{ messages: any }> => {
  const store = config.store;
  if (!store) {
    if (!store) {
      throw new Error("store is required when compiling the graph");
    }
  }
  if (!config.configurable?.userId) {
    throw new Error("userId is required in the config");
  }
  const namespace = ["memories", config.configurable?.userId];
  const memories = await store.search(namespace);
  const info = memories.map((d) => d.value.data).join("\n");
  const systemMsg = `You are a helpful assistant talking to the user. User info: ${info}`;

  // Store new memories if the user asks the model to remember
  const lastMessage = state.messages.at(-1);
  if (
    typeof lastMessage.content === "string" &&
    lastMessage.content.toLowerCase().includes("remember")
  ) {
    await store.put(namespace, uuidv4(), { data: lastMessage.content });
  }

  const response = await model.invoke([
    { type: "system", content: systemMsg },
    ...state.messages,
  ]);
  return { messages: response };
};

const builder = new StateGraph(StateAnnotation)
  .addNode("call_model", callModel)
  .addEdge(START, "call_model");

// NOTE: we're passing the store object here when compiling the graph
const graph = builder.compile({
  checkpointer: new MemorySaver(),
  store: inMemoryStore,
});
// If you're using LangGraph Cloud or LangGraph Studio, you don't need to pass the store or checkpointer when compiling the graph, since it's done automatically.
```

```typescript
let config = { configurable: { thread_id: "1", userId: "1" } };
let inputMessage = { type: "user", content: "Hi! Remember: my name is Bob" };

for await (const chunk of await graph.stream(
  { messages: [inputMessage] },
  { ...config, streamMode: "values" }
)) {
  console.log(chunk.messages.at(-1));
}
```

```typescript
config = { configurable: { thread_id: "2", userId: "1" } };
inputMessage = { type: "user", content: "what is my name?" };

for await (const chunk of await graph.stream(
  { messages: [inputMessage] },
  { ...config, streamMode: "values" }
)) {
  console.log(chunk.messages.at(-1));
}
```

```typescript
const memories = await inMemoryStore.search(["memories", "1"]);
for (const memory of memories) {
  console.log(await memory.value);
}
```

```typescript
config = { configurable: { thread_id: "3", userId: "2" } };
inputMessage = { type: "user", content: "what is my name?" };

for await (const chunk of await graph.stream(
  { messages: [inputMessage] },
  { ...config, streamMode: "values" }
)) {
  console.log(chunk.messages.at(-1));
}
```

```typescript
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const responseFormat = z.object({
  // Respond to the user in this format
  mySpecialOutput: z.string(),
});

const graph = createReactAgent({
  llm: llm,
  tools: tools,
  // specify the schema for the structured output using `responseFormat` parameter
  responseFormat: responseFormat,
});
```

```typescript
// process.env.OPENAI_API_KEY = "sk_...";

// Optional, add tracing in LangSmith
// process.env.LANGSMITH_API_KEY = "ls__..."
process.env.LANGSMITH_TRACING = "true";
process.env.LANGSMITH_PROJECT = "ReAct Agent with system prompt: LangGraphJS";
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const weatherTool = tool(
  async (input): Promise<string> => {
    if (input.city === "nyc") {
      return "It might be cloudy in nyc";
    } else if (input.city === "sf") {
      return "It's always sunny in sf";
    } else {
      throw new Error("Unknown city");
    }
  },
  {
    name: "get_weather",
    description: "Use this to get weather information.",
    schema: z.object({
      city: z.enum(["nyc", "sf"]).describe("The city to get weather for"),
    }),
  }
);

const WeatherResponseSchema = z.object({
  conditions: z.string().describe("Weather conditions"),
});

const tools = [weatherTool];

const agent = createReactAgent({
  llm: new ChatOpenAI({ model: "gpt-4o", temperature: 0 }),
  tools: tools,
  responseFormat: WeatherResponseSchema,
});
```

```typescript
const response = await agent.invoke({
  messages: [
    {
      role: "user",
      content: "What's the weather in NYC?",
    },
  ],
});
```

```typescript
response.structuredResponse;
```

```typescript
const agent = createReactAgent({
  llm: new ChatOpenAI({ model: "gpt-4o", temperature: 0 }),
  tools: tools,
  responseFormat: {
    prompt: "Always return capitalized weather conditions",
    schema: WeatherResponseSchema,
  },
});

const response = await agent.invoke({
  messages: [
    {
      role: "user",
      content: "What's the weather in NYC?",
    },
  ],
});
```

```typescript
response.structuredResponse;
```

```typescript
process.env.OPENAI_API_KEY = "YOUR_API_KEY";
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

```typescript
import { InMemoryStore } from "@langchain/langgraph";
import { OpenAIEmbeddings } from "@langchain/openai";

const inMemoryStore = new InMemoryStore({
  index: {
    embeddings: new OpenAIEmbeddings({
      model: "text-embedding-3-small",
    }),
    dims: 1536,
  },
});
```

```typescript
import { v4 } from "uuid";
import { ChatAnthropic } from "@langchain/anthropic";
import {
  entrypoint,
  task,
  MemorySaver,
  addMessages,
  type BaseStore,
  getStore,
} from "@langchain/langgraph";
import type { BaseMessage, BaseMessageLike } from "@langchain/core/messages";

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
});

const callModel = task(
  "callModel",
  async (messages: BaseMessage[], memoryStore: BaseStore, userId: string) => {
    const namespace = ["memories", userId];
    const lastMessage = messages.at(-1);
    if (typeof lastMessage?.content !== "string") {
      throw new Error("Received non-string message content.");
    }
    const memories = await memoryStore.search(namespace, {
      query: lastMessage.content,
    });
    const info = memories.map((memory) => memory.value.data).join("\n");
    const systemMessage = `You are a helpful assistant talking to the user. User info: ${info}`;

    // Store new memories if the user asks the model to remember
    if (lastMessage.content.toLowerCase().includes("remember")) {
      // Hard-coded for demo
      const memory = `Username is Bob`;
      await memoryStore.put(namespace, v4(), { data: memory });
    }
    const response = await model.invoke([
      {
        role: "system",
        content: systemMessage,
      },
      ...messages,
    ]);
    return response;
  }
);

// NOTE: we're passing the store object here when creating a workflow via entrypoint()
const workflow = entrypoint(
  {
    checkpointer: new MemorySaver(),
    store: inMemoryStore,
    name: "workflow",
  },
  async (
    params: {
      messages: BaseMessageLike[];
      userId: string;
    },
    config
  ) => {
    const messages = addMessages([], params.messages);
    const response = await callModel(messages, config.store, params.userId);
    return entrypoint.final({
      value: response,
      save: addMessages(messages, response),
    });
  }
);
```

```typescript
const config = {
  configurable: {
    thread_id: "1",
  },
  streamMode: "values" as const,
};

const inputMessage = {
  role: "user",
  content: "Hi! Remember: my name is Bob",
};

const stream = await workflow.stream(
  { messages: [inputMessage], userId: "1" },
  config
);

for await (const chunk of stream) {
  console.log(chunk);
}
```

```typescript
const config2 = {
  configurable: {
    thread_id: "2",
  },
  streamMode: "values" as const,
};

const followupStream = await workflow.stream(
  {
    messages: [
      {
        role: "user",
        content: "what is my name?",
      },
    ],
    userId: "1",
  },
  config2
);

for await (const chunk of followupStream) {
  console.log(chunk);
}
```

```typescript
const memories = await inMemoryStore.search(["memories", "1"]);
for (const memory of memories) {
  console.log(memory.value);
}
```

```typescript
const config3 = {
  configurable: {
    thread_id: "3",
  },
  streamMode: "values" as const,
};

const otherUserStream = await workflow.stream(
  {
    messages: [
      {
        role: "user",
        content: "what is my name?",
      },
    ],
    userId: "2",
  },
  config3
);

for await (const chunk of otherUserStream) {
  console.log(chunk);
}
```

```typescript
process.env.OPENAI_API_KEY = "YOUR_API_KEY";
```

```typescript
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_API_KEY = "YOUR_API_KEY";
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import {
  MessagesAnnotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

const memory = new MemorySaver();

const search = tool(
  (_) => {
    // This is a placeholder for the actual implementation
    // Don't let the LLM know this though 😊
    return [
      "It's sunny in San Francisco, but you better look out if you're a Gemini 😈.",
    ];
  },
  {
    name: "search",
    description: "Call to surf the web.",
    schema: z.object({
      query: z.string(),
    }),
  }
);

const tools = [search];
const toolNode = new ToolNode<typeof MessagesAnnotation.State>(tools);
const model = new ChatOpenAI({ model: "gpt-4o" });
const boundModel = model.bindTools(tools);

function shouldContinue(
  state: typeof MessagesAnnotation.State
): "action" | typeof END {
  const lastMessage = state.messages.at(-1);
  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length
  ) {
    return "action";
  }
  // If there is no tool call, then we finish
  return END;
}

// Define the function that calls the model
async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await boundModel.invoke(state.messages);
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  // Define the two nodes we will cycle between
  .addNode("agent", callModel)
  .addNode("action", toolNode)
  // Set the entrypoint as `agent`
  // This means that this node is the first one called
  .addEdge(START, "agent")
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `agent`.
    // This means these are the edges taken after the `agent` node is called.
    "agent",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue
  )
  // We now add a normal edge from `tools` to `agent`.
  // This means that after `tools` is called, `agent` node is called next.
  .addEdge("action", "agent");

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const app = workflow.compile({ checkpointer: memory });
```

```typescript
import { HumanMessage } from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";

const config = {
  configurable: { thread_id: "2" },
  streamMode: "values" as const,
};
const inputMessage = new HumanMessage({
  id: uuidv4(),
  content: "hi! I'm bob",
});

for await (const event of await app.stream(
  { messages: [inputMessage] },
  config
)) {
  const lastMsg = event.messages.at(-1);
  console.dir(
    {
      type: lastMsg.getType(),
      content: lastMsg.content,
      tool_calls: lastMsg.tool_calls,
    },
    { depth: null }
  );
}

const inputMessage2 = new HumanMessage({
  id: uuidv4(),
  content: "What's my name?",
});
for await (const event of await app.stream(
  { messages: [inputMessage2] },
  config
)) {
  const lastMsg = event.messages.at(-1);
  console.dir(
    {
      type: lastMsg.getType(),
      content: lastMsg.content,
      tool_calls: lastMsg.tool_calls,
    },
    { depth: null }
  );
}
```

```typescript
const messages = (await app.getState(config)).values.messages;
console.dir(
  messages.map((msg) => ({
    id: msg.id,
    type: msg.getType(),
    content: msg.content,
    tool_calls: msg.tool_calls,
  })),
  { depth: null }
);
```

```typescript
import { RemoveMessage } from "@langchain/core/messages";

await app.updateState(config, {
  messages: new RemoveMessage({ id: messages[0].id }),
});
```

```typescript
const updatedMessages = (await app.getState(config)).values.messages;
console.dir(
  updatedMessages.map((msg) => ({
    id: msg.id,
    type: msg.getType(),
    content: msg.content,
    tool_calls: msg.tool_calls,
  })),
  { depth: null }
);
```

```typescript
import { RemoveMessage } from "@langchain/core/messages";
import { StateGraph, START, END } from "@langchain/langgraph";
import { MessagesAnnotation } from "@langchain/langgraph";

function deleteMessages(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  if (messages.length > 3) {
    return {
      messages: messages
        .slice(0, -3)
        .map((m) => new RemoveMessage({ id: m.id })),
    };
  }
  return {};
}

// We need to modify the logic to call deleteMessages rather than end right away
function shouldContinue2(
  state: typeof MessagesAnnotation.State
): "action" | "delete_messages" {
  const lastMessage = state.messages.at(-1);
  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length
  ) {
    return "action";
  }
  // Otherwise if there aren't, we finish
  return "delete_messages";
}

// Define a new graph
const workflow2 = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("action", toolNode)
  // This is our new node we're defining
  .addNode("delete_messages", deleteMessages)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue2)
  .addEdge("action", "agent")
  // This is the new edge we're adding: after we delete messages, we finish
  .addEdge("delete_messages", END);

const app2 = workflow2.compile({ checkpointer: memory });
```

```typescript
import { HumanMessage } from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";

const config2 = {
  configurable: { thread_id: "3" },
  streamMode: "values" as const,
};

const inputMessage3 = new HumanMessage({
  id: uuidv4(),
  content: "hi! I'm bob",
});

console.log("--- FIRST ITERATION ---\n");
for await (const event of await app2.stream(
  { messages: [inputMessage3] },
  config2
)) {
  console.log(
    event.messages.map((message) => [message.getType(), message.content])
  );
}

const inputMessage4 = new HumanMessage({
  id: uuidv4(),
  content: "what's my name?",
});

console.log("\n\n--- SECOND ITERATION ---\n");
for await (const event of await app2.stream(
  { messages: [inputMessage4] },
  config2
)) {
  console.log(
    event.messages.map((message) => [message.getType(), message.content]),
    "\n"
  );
}
```

```typescript
const messages3 = (await app.getState(config2)).values["messages"];
console.dir(
  messages3.map((msg) => ({
    id: msg.id,
    type: msg.getType(),
    content: msg.content,
    tool_calls: msg.tool_calls,
  })),
  { depth: null }
);
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, MessagesAnnotation } from "langgraph";

const model = new ChatOpenAI({ model: "gpt-4-turbo-preview" });

function callModel(state: typeof MessagesAnnotation.State) {
  const response = model.invoke(state.messages);
  return { messages: response };
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("callModel", callModel)
  .addEdge("start", "callModel")
  .addEdge("callModel", "end");
const app = workflow.compile();

const inputs = [{ role: "user", content: "hi!" }];

for await (const event of app.streamEvents({ messages: inputs })) {
  const kind = event.event;
  console.log(`${kind}: ${event.name}`);
}
```

```typescript
import { Annotation } from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
  myList: Annotation<any[]>({
    reducer: (
      existing: string[],
      updates: string[] | { type: string; from: number; to?: number }
    ) => {
      if (Array.isArray(updates)) {
        // Normal case, add to the history
        return [...existing, ...updates];
      } else if (typeof updates === "object" && updates.type === "keep") {
        // You get to decide what this looks like.
        // For example, you could simplify and just accept a string "DELETE"
        // and clear the entire list.
        return existing.slice(updates.from, updates.to);
      }
      // etc. We define how to interpret updates
      return existing;
    },
    default: () => [],
  }),
});

type State = typeof StateAnnotation.State;

function myNode(state: State) {
  return {
    // We return an update for the field "myList" saying to
    // keep only values from index -5 to the end (deleting the rest)
    myList: { type: "keep", from: -5, to: undefined },
  };
}
```

```typescript
import { RemoveMessage, AIMessage } from "@langchain/core/messages";
import { MessagesAnnotation } from "@langchain/langgraph";

type State = typeof MessagesAnnotation.State;

function myNode1(state: State) {
  // Add an AI message to the `messages` list in the state
  return { messages: [new AIMessage({ content: "Hi" })] };
}

function myNode2(state: State) {
  // Delete all but the last 2 messages from the `messages` list in the state
  const deleteMessages = state.messages
    .slice(0, -2)
    .map((m) => new RemoveMessage({ id: m.id }));
  return { messages: deleteMessages };
}
```

```typescript
import { MessagesAnnotation, Annotation } from "@langchain/langgraph";

const MyGraphAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  summary: Annotation<string>,
});
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, RemoveMessage } from "@langchain/core/messages";

type State = typeof MyGraphAnnotation.State;

async function summarizeConversation(state: State) {
  // First, we get any existing summary
  const summary = state.summary || "";

  // Create our summarization prompt
  let summaryMessage: string;
  if (summary) {
    // A summary already exists
    summaryMessage =
      `This is a summary of the conversation to date: ${summary}\n\n` +
      "Extend the summary by taking into account the new messages above:";
  } else {
    summaryMessage = "Create a summary of the conversation above:";
  }

  // Add prompt to our history
  const messages = [
    ...state.messages,
    new HumanMessage({ content: summaryMessage }),
  ];

  // Assuming you have a ChatOpenAI model instance
  const model = new ChatOpenAI();
  const response = await model.invoke(messages);

  // Delete all but the 2 most recent messages
  const deleteMessages = state.messages
    .slice(0, -2)
    .map((m) => new RemoveMessage({ id: m.id }));

  return {
    summary: response.content,
    messages: deleteMessages,
  };
}
```

```typescript
import { trimMessages } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

trimMessages(messages, {
  // Keep the last <= n_count tokens of the messages.
  strategy: "last",
  // Remember to adjust based on your model
  // or else pass a custom token_encoder
  tokenCounter: new ChatOpenAI({ modelName: "gpt-4" }),
  // Remember to adjust based on the desired conversation
  // length
  maxTokens: 45,
  // Most chat models expect that chat history starts with either:
  // (1) a HumanMessage or
  // (2) a SystemMessage followed by a HumanMessage
  startOn: "human",
  // Most chat models expect that chat history ends with either:
  // (1) a HumanMessage or
  // (2) a ToolMessage
  endOn: ["human", "tool"],
  // Usually, we want to keep the SystemMessage
  // if it's present in the original history.
  // The SystemMessage has special instructions for the model.
  includeSystem: true,
});
```

```typescript
import { InMemoryStore } from "@langchain/langgraph";

// InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
const store = new InMemoryStore();
const userId = "my-user";
const applicationContext = "chitchat";
const namespace = [userId, applicationContext];
await store.put(namespace, "a-memory", {
  rules: [
    "User likes short, direct language",
    "User only speaks English & TypeScript",
  ],
  "my-key": "my-value",
});
// get the "memory" by ID
const item = await store.get(namespace, "a-memory");
// list "memories" within this namespace, filtering on content equivalence
const items = await store.search(namespace, {
  filter: { "my-key": "my-value" },
});
```

```typescript
import { BaseStore } from "@langchain/langgraph/store";
import { State } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

// Node that *uses* the instructions
const callModel = async (state: State, store: BaseStore) => {
  const namespace = ["agent_instructions"];
  const instructions = await store.get(namespace, "agent_a");
  // Application logic
  const prompt = promptTemplate.format({
    instructions: instructions[0].value.instructions,
  });
  // ... rest of the logic
};

// Node that updates instructions
const updateInstructions = async (state: State, store: BaseStore) => {
  const namespace = ["instructions"];
  const currentInstructions = await store.search(namespace);
  // Memory logic
  const prompt = promptTemplate.format({
    instructions: currentInstructions[0].value.instructions,
    conversation: state.messages,
  });
  const llm = new ChatOpenAI();
  const output = await llm.invoke(prompt);
  const newInstructions = output.content; // Assuming the LLM returns the new instructions
  await store.put(["agent_instructions"], "agent_a", {
    instructions: newInstructions,
  });
  // ... rest of the logic
};
```

```typescript
import { task, entrypoint, interrupt, MemorySaver } from "@langchain/langgraph";

const writeEssay = task("write_essay", (topic: string): string => {
  // A placeholder for a long-running task.
  return `An essay about topic: ${topic}`;
});

const workflow = entrypoint(
  { checkpointer: new MemorySaver(), name: "workflow" },
  async (topic: string) => {
    const essay = await writeEssay(topic);
    const isApproved = interrupt({
      // Any json-serializable payload provided to interrupt as argument.
      // It will be surfaced on the client side as an Interrupt when streaming data
      // from the workflow.
      essay, // The essay we want reviewed.
      // We can add any additional information that we need.
      // For example, introduce a key called "action" with some instructions.
      action: "Please approve/reject the essay",
    });

    return {
      essay, // The essay that was generated
      isApproved, // Response from HIL
    };
  }
);
```

```typescript
import {
  task,
  entrypoint,
  interrupt,
  MemorySaver,
  Command,
} from "@langchain/langgraph";

const writeEssay = task("write_essay", (topic: string): string => {
  return `An essay about topic: ${topic}`;
});

const workflow = entrypoint(
  { checkpointer: new MemorySaver(), name: "workflow" },
  async (topic: string) => {
    const essay = await writeEssay(topic);
    const isApproved = interrupt({
      essay, // The essay we want reviewed.
      action: "Please approve/reject the essay",
    });

    return {
      essay,
      isApproved,
    };
  }
);

const threadId = crypto.randomUUID();

const config = {
  configurable: {
    thread_id: threadId,
  },
};

for await (const item of await workflow.stream("cat", config)) {
  console.log(item);
}
```

```typescript
{
  write_essay: "An essay about topic: cat";
}
{
  __interrupt__: [
    {
      value: {
        essay: "An essay about topic: cat",
        action: "Please approve/reject the essay",
      },
      resumable: true,
      ns: ["workflow:f7b8508b-21c0-8b4c-5958-4e8de74d2684"],
      when: "during",
    },
  ];
}
```

```typescript
// Get review from a user (e.g., via a UI)
// In this case, we're using a bool, but this can be any json-serializable value.
const humanReview = true;

for await (const item of await workflow.stream(
  new Command({ resume: humanReview }),
  config
)) {
  console.log(item);
}
```

```typescript
{ workflow: { essay: 'An essay about topic: cat', isApproved: true } }
```

```typescript
import { entrypoint, MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (someInput: Record<string, any>): Promise<number> => {
    // some logic that may involve long-running tasks like API calls,
    // and may be interrupted for human-in-the-loop.
    return result;
  }
);
```

```typescript
import {
  entrypoint,
  getPreviousState,
  BaseStore,
  InMemoryStore,
} from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";

const inMemoryStore = new InMemoryStore(...);  // An instance of InMemoryStore for long-term memory

const myWorkflow = entrypoint(
  {
    checkpointer,  // Specify the checkpointer
    store: inMemoryStore,  // Specify the store
    name: "myWorkflow",
  },
  async (someInput: Record<string, any>) => {
    const previous = getPreviousState<any>(); // For short-term memory
    // Rest of workflow logic...
  }
);
```

```typescript
const config = {
  configurable: {
    thread_id: "some_thread_id",
  },
};
await myWorkflow.invoke(someInput, config); // Wait for the result
```

```typescript
const config = {
  configurable: {
    thread_id: "some_thread_id",
  },
};

for await (const chunk of await myWorkflow.stream(someInput, config)) {
  console.log(chunk);
}
```

```typescript
import { Command } from "@langchain/langgraph";

const config = {
  configurable: {
    thread_id: "some_thread_id",
  },
};

await myWorkflow.invoke(new Command({ resume: someResumeValue }), config);
```

```typescript
import { Command } from "@langchain/langgraph";

const config = {
  configurable: {
    thread_id: "some_thread_id",
  },
};

const stream = await myWorkflow.stream(
  new Command({ resume: someResumeValue }),
  config
);

for await (const chunk of stream) {
  console.log(chunk);
}
```

```typescript
const config = {
  configurable: {
    thread_id: "some_thread_id",
  },
};

await myWorkflow.invoke(null, config);
```

```typescript
const config = {
  configurable: {
    thread_id: "some_thread_id",
  },
};

for await (const chunk of await myWorkflow.stream(null, config)) {
  console.log(chunk);
}
```

```typescript
const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (number: number) => {
    const previous = getPreviousState<number>();
    return number + (previous ?? 0);
  }
);

const config = {
  configurable: {
    thread_id: "some_thread_id",
  },
};

await myWorkflow.invoke(1, config); // 1 (previous was undefined)
await myWorkflow.invoke(2, config); // 3 (previous was 1 from the previous invocation)
```

```typescript
const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (number: number) => {
    const previous = getPreviousState<number>();
    // This will return the previous value to the caller, saving
    // 2 * number to the checkpoint, which will be used in the next invocation
    // for the previous state
    return entrypoint.final({
      value: previous ?? 0,
      save: 2 * number,
    });
  }
);

const config = {
  configurable: {
    thread_id: "1",
  },
};

await myWorkflow.invoke(3, config); // 0 (previous was undefined)
await myWorkflow.invoke(1, config); // 6 (previous was 3 * 2 from the previous invocation)
```

```typescript
import { task } from "@langchain/langgraph";

const slowComputation = task({"slowComputation", async (inputValue: any) => {
  // Simulate a long-running operation
  ...
  return result;
});
```

```typescript
const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (someInput: number) => {
    return await slowComputation(someInput);
  }
);
```

```typescript
const slowComputation = task(
  {
    name: "slowComputation",
    // only attempt to run this task once before giving up
    retry: { maxAttempts: 1 },
  },
  async (inputValue: any) => {
    // A long-running operation that may fail
    return result;
  }
);
```

```typescript hl_lines="6"
const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (inputs: Record<string, any>) => {
    // This code will be executed a second time when resuming the workflow.
    // Which is likely not what you want.
    await fs.writeFile("output.txt", "Side effect executed");
    const value = interrupt("question");
    return value;
  }
);
```

```typescript hl_lines="3"
import { task } from "@langchain/langgraph";

const writeToFile = task("writeToFile", async () => {
  await fs.writeFile("output.txt", "Side effect executed");
});

const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (inputs: Record<string, any>) => {
    // The side effect is now encapsulated in a task.
    await writeToFile();
    const value = interrupt("question");
    return value;
  }
);
```

```typescript hl_lines="4"
const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (inputs: { t0: number }) => {
    const t1 = Date.now();

    const deltaT = t1 - inputs.t0;

    if (deltaT > 1000) {
      const result = await slowTask(1);
      const value = interrupt("question");
      return { result, value };
    } else {
      const result = await slowTask(2);
      const value = interrupt("question");
      return { result, value };
    }
  }
);
```

```typescript hl_lines="3 8"
import { task } from "@langchain/langgraph";

const getTime = task("getTime", () => Date.now());

const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (inputs: { t0: number }) => {
    const t1 = await getTime();

    const deltaT = t1 - inputs.t0;

    if (deltaT > 1000) {
      const result = await slowTask(1);
      const value = interrupt("question");
      return { result, value };
    } else {
      const result = await slowTask(2);
      const value = interrupt("question");
      return { result, value };
    }
  }
);
```

```typescript
const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (inputs: { value: number; anotherValue: number }) => {
    const value = inputs.value;
    const anotherValue = inputs.anotherValue;
    ...
  }
);

await myWorkflow.invoke([{ value: 1, anotherValue: 2 }]);
```

```typescript
const addOne = task("addOne", (number: number) => number + 1);

const graph = entrypoint(
  { checkpointer, name: "graph" },
  async (numbers: number[]) => {
    return await Promise.all(numbers.map(addOne));
  }
);
```

```typescript
import { entrypoint, StateGraph } from "@langchain/langgraph";

const builder = new StateGraph();
...
const someGraph = builder.compile();

const someWorkflow = entrypoint(
  { name: "someWorkflow" },
  async (someInput: Record<string, any>) => {
    // Call a graph defined using the graph API
    const result1 = await someGraph.invoke(...);
    // Call another graph defined using the graph API
    const result2 = await anotherGraph.invoke(...);
    return {
      result1,
      result2,
    };
  }
);
```

```typescript
const someOtherWorkflow = entrypoint(
  { name: "someOtherWorkflow" }, // Will automatically use the checkpointer from the parent entrypoint
  async (inputs: { value: number }) => {
    return inputs.value;
  }
);

const myWorkflow = entrypoint(
  { checkpointer, name: "myWorkflow" },
  async (inputs: Record<string, any>) => {
    const value = await someOtherWorkflow.invoke([{ value: 1 }]);
    return value;
  }
);
```

```typescript
import {
  entrypoint,
  task,
  MemorySaver,
  LangGraphRunnableConfig,
} from "@langchain/langgraph";

const addOne = task("addOne", (x: number) => x + 1);

const addTwo = task("addTwo", (x: number) => x + 2);

const checkpointer = new MemorySaver();

const main = entrypoint(
  { checkpointer, name: "main" },
  async (inputs: { number: number }, config: LangGraphRunnableConfig) => {
    config.writer?.("hello"); // Write some data to the `custom` stream
    await addOne(inputs.number); // Will write data to the `updates` stream
    config.writer?.("world"); // Write some more data to the `custom` stream
    await addTwo(inputs.number); // Will write data to the `updates` stream
    return 5;
  }
);

const config = {
  configurable: {
    thread_id: "1",
  },
};

const stream = await main.stream(
  { number: 1 },
  { streamMode: ["custom", "updates"], ...config }
);

for await (const chunk of stream) {
  console.log(chunk);
}
```

```typescript
["updates", { addOne: 2 }][("updates", { addTwo: 3 })][("custom", "hello")][
  ("custom", "world")
][("updates", { main: 5 })];
```

```typescript
import { entrypoint, task, MemorySaver } from "@langchain/langgraph";

// Global variable to track the number of attempts
let attempts = 0;

const getInfo = task("getInfo", () => {
  /*
   * Simulates a task that fails once before succeeding.
   * Throws an error on the first attempt, then returns "OK" on subsequent tries.
   */
  attempts += 1;

  if (attempts < 2) {
    throw new Error("Failure"); // Simulate a failure on the first attempt
  }
  return "OK";
});

// Initialize an in-memory checkpointer for persistence
const checkpointer = new MemorySaver();

const slowTask = task("slowTask", async () => {
  /*
   * Simulates a slow-running task by introducing a 1-second delay.
   */
  await new Promise((resolve) => setTimeout(resolve, 1000));
  return "Ran slow task.";
});

const main = entrypoint(
  { checkpointer, name: "main" },
  async (inputs: Record<string, any>) => {
    /*
     * Main workflow function that runs the slowTask and getInfo tasks sequentially.
     *
     * Parameters:
     * - inputs: Record<string, any> containing workflow input values.
     *
     * The workflow first executes `slowTask` and then attempts to execute `getInfo`,
     * which will fail on the first invocation.
     */
    const slowTaskResult = await slowTask(); // Blocking call to slowTask
    await getInfo(); // Error will be thrown here on the first attempt
    return slowTaskResult;
  }
);

// Workflow execution configuration with a unique thread identifier
const config = {
  configurable: {
    thread_id: "1", // Unique identifier to track workflow execution
  },
};

// This invocation will take ~1 second due to the slowTask execution
try {
  // First invocation will throw an error due to the `getInfo` task failing
  await main.invoke({ anyInput: "foobar" }, config);
} catch (err) {
  // Handle the failure gracefully
}
```

```typescript
await main.invoke(null, config);
```

```typescript
"Ran slow task.";
```

```typescript
// Compile our graph with a checkpointer and a breakpoint before "step_for_human_in_the_loop"
const graph = builder.compile({
  checkpointer,
  interruptBefore: ["step_for_human_in_the_loop"],
});

// Run the graph up to the breakpoint
const threadConfig = {
  configurable: { thread_id: "1" },
  streamMode: "values" as const,
};
for await (const event of await graph.stream(inputs, threadConfig)) {
  console.log(event);
}

// Perform some action that requires human in the loop

// Continue the graph execution from the current checkpoint
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
function myNode(
  state: typeof GraphAnnotation.State
): typeof GraphAnnotation.State {
  if (state.input.length > 5) {
    throw new NodeInterrupt(
      `Received input that is longer than 5 characters: ${state["input"]}`
    );
  }
  return state;
}
```

```typescript
// Attempt to continue the graph execution with no change to state after we hit the dynamic breakpoint
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
// Update the state to pass the dynamic breakpoint
await graph.updateState(threadConfig, { input: "foo" });
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
// This update will skip the node `myNode` altogether
await graph.updateState(threadConfig, null, "myNode");
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
// Compile our graph with a checkpointer and a breakpoint before the step to approve
const graph = builder.compile({ checkpointer, interruptBefore: ["node_2"] });

// Run the graph up to the breakpoint
for await (const event of await graph.stream(inputs, threadConfig)) {
  console.log(event);
}

// ... Get human approval ...

// If approved, continue the graph execution from the last saved checkpoint
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
// Compile our graph with a checkpointer and a breakpoint before the step to review
const graph = builder.compile({ checkpointer, interruptBefore: ["node_2"] });

// Run the graph up to the breakpoint
for await (const event of await graph.stream(inputs, threadConfig)) {
  console.log(event);
}

// Review the state, decide to edit it, and create a forked checkpoint with the new state
await graph.updateState(threadConfig, { state: "new state" });

// Continue the graph execution from the forked checkpoint
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
// Compile our graph with a checkpointer and a breakpoint before the step to collect human input
const graph = builder.compile({
  checkpointer,
  interruptBefore: ["human_input"],
});

// Run the graph up to the breakpoint
for await (const event of await graph.stream(inputs, threadConfig)) {
  console.log(event);
}

// Update the state with the user input as if it was the human_input node
await graph.updateState(threadConfig, { user_input: userInput }, "human_input");

// Continue the graph execution from the checkpoint created by the human_input node
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
// Compile our graph with a checkpointer and a breakpoint before the step to review the tool call from the LLM
const graph = builder.compile({
  checkpointer,
  interruptBefore: ["human_review"],
});

// Run the graph up to the breakpoint
for await (const event of await graph.stream(inputs, threadConfig)) {
  console.log(event);
}

// Review the tool call and update it, if needed, as the human_review node
await graph.updateState(
  threadConfig,
  { tool_call: "updated tool call" },
  "human_review"
);

// Otherwise, approve the tool call and proceed with the graph execution with no edits

// Continue the graph execution from either:
// (1) the forked checkpoint created by human_review or
// (2) the checkpoint saved when the tool call was originally made (no edits in human_review)
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
const threadConfig = { configurable: { thread_id: "1" } };
for await (const event of await graph.stream(null, threadConfig)) {
  console.log(event);
}
```

```typescript
const allCheckpoints = [];
for await (const state of app.getStateHistory(threadConfig)) {
  allCheckpoints.push(state);
}
```

```typescript
const config = {
  configurable: { thread_id: "1", checkpoint_id: "xxx" },
  streamMode: "values" as const,
};
for await (const event of await graph.stream(null, config)) {
  console.log(event);
}
```

```typescript
const config = { configurable: { thread_id: "1", checkpoint_id: "xxx" } };
await graph.updateState(config, { state: "updated state" });
```

```typescript
const config = {
  configurable: { thread_id: "1", checkpoint_id: "xxx-fork" },
  streamMode: "values" as const,
};
for await (const event of await graph.stream(null, config)) {
  console.log(event);
}
```

```typescript
import { interrupt } from "@langchain/langgraph";

function humanNode(state: typeof GraphAnnotation.State) {
  const value = interrupt(
    // Any JSON serializable value to surface to the human.
    // For example, a question or a piece of text or a set of keys in the state
    {
      text_to_revise: state.some_text,
    }
  );
  // Update the state with the human's input or route the graph based on the input
  return {
    some_text: value,
  };
}

const graph = workflow.compile({
  checkpointer, // Required for `interrupt` to work
});

// Run the graph until the interrupt
const threadConfig = { configurable: { thread_id: "some_id" } };
await graph.invoke(someInput, threadConfig);

// Resume the graph with the human's input
await graph.invoke(new Command({ resume: valueFromHuman }), threadConfig);
```

```typescript
{
  some_text: "Edited text";
}
```

```typescript
import {
  MemorySaver,
  Annotation,
  interrupt,
  Command,
  StateGraph,
} from "@langchain/langgraph";

// Define the graph state
const StateAnnotation = Annotation.Root({
  some_text: Annotation<string>(),
});

function humanNode(state: typeof StateAnnotation.State) {
  const value = interrupt(
    // Any JSON serializable value to surface to the human.
    // For example, a question or a piece of text or a set of keys in the state
    {
      text_to_revise: state.some_text,
    }
  );
  return {
    // Update the state with the human's input
    some_text: value,
  };
}

// Build the graph
const workflow = new StateGraph(StateAnnotation)
  // Add the human-node to the graph
  .addNode("human_node", humanNode)
  .addEdge("__start__", "human_node");

// A checkpointer is required for `interrupt` to work.
const checkpointer = new MemorySaver();
const graph = workflow.compile({
  checkpointer,
});

// Using stream() to directly surface the `__interrupt__` information.
for await (const chunk of await graph.stream(
  { some_text: "Original text" },
  threadConfig
)) {
  console.log(chunk);
}

// Resume using Command
for await (const chunk of await graph.stream(
  new Command({ resume: "Edited text" }),
  threadConfig
)) {
  console.log(chunk);
}
```

```typescript
{
  __interrupt__: [
    {
      value: { question: "Please revise the text", some_text: "Original text" },
      resumable: true,
      ns: ["human_node:10fe492f-3688-c8c6-0d0a-ec61a43fecd6"],
      when: "during",
    },
  ];
}
{
  human_node: {
    some_text: "Edited text";
  }
}
```

```typescript
import { interrupt, Command } from "@langchain/langgraph";

function humanApproval(state: typeof GraphAnnotation.State): Command {
  const isApproved = interrupt({
    question: "Is this correct?",
    // Surface the output that should be
    // reviewed and approved by the human.
    llm_output: state.llm_output,
  });

  if (isApproved) {
    return new Command({ goto: "some_node" });
  } else {
    return new Command({ goto: "another_node" });
  }
}

// Add the node to the graph in an appropriate location
// and connect it to the relevant nodes.
graphBuilder.addNode("human_approval", humanApproval);

const graph = graphBuilder.compile({ checkpointer });

// After running the graph and hitting the interrupt, the graph will pause.
// Resume it with either an approval or rejection.
const threadConfig = { configurable: { thread_id: "some_id" } };
await graph.invoke(new Command({ resume: true }), threadConfig);
```

```typescript
import { interrupt } from "@langchain/langgraph";

function humanEditing(state: typeof GraphAnnotation.State): Command {
  const result = interrupt({
    // Interrupt information to surface to the client.
    // Can be any JSON serializable value.
    task: "Review the output from the LLM and make any necessary edits.",
    llm_generated_summary: state.llm_generated_summary,
  });

  // Update the state with the edited text
  return {
    llm_generated_summary: result.edited_text,
  };
}

// Add the node to the graph in an appropriate location
// and connect it to the relevant nodes.
graphBuilder.addNode("human_editing", humanEditing);

const graph = graphBuilder.compile({ checkpointer });

// After running the graph and hitting the interrupt, the graph will pause.
// Resume it with the edited text.
const threadConfig = { configurable: { thread_id: "some_id" } };
await graph.invoke(
  new Command({ resume: { edited_text: "The edited text" } }),
  threadConfig
);
```

```typescript
import { interrupt, Command } from "@langchain/langgraph";

function humanReviewNode(state: typeof GraphAnnotation.State): Command {
  // This is the value we'll be providing via Command.resume(<human_review>)
  const humanReview = interrupt({
    question: "Is this correct?",
    // Surface tool calls for review
    tool_call: toolCall,
  });

  const [reviewAction, reviewData] = humanReview;

  // Approve the tool call and continue
  if (reviewAction === "continue") {
    return new Command({ goto: "run_tool" });
  }
  // Modify the tool call manually and then continue
  else if (reviewAction === "update") {
    const updatedMsg = getUpdatedMsg(reviewData);
    // Remember that to modify an existing message you will need
    // to pass the message with a matching ID.
    return new Command({
      goto: "run_tool",
      update: { messages: [updatedMsg] },
    });
  }
  // Give natural language feedback, and then pass that back to the agent
  else if (reviewAction === "feedback") {
    const feedbackMsg = getFeedbackMsg(reviewData);
    return new Command({
      goto: "call_llm",
      update: { messages: [feedbackMsg] },
    });
  }
}
```

```typescript
import { interrupt } from "@langchain/langgraph";

function humanInput(state: typeof GraphAnnotation.State) {
  const humanMessage = interrupt("human_input");

  return {
    messages: [
      {
        role: "human",
        content: humanMessage,
      },
    ],
  };
}

function agent(state: typeof GraphAnnotation.State) {
  // Agent logic
  // ...
}

graphBuilder.addNode("human_input", humanInput);
graphBuilder.addEdge("human_input", "agent");

const graph = graphBuilder.compile({ checkpointer });

// After running the graph and hitting the interrupt, the graph will pause.
// Resume it with the human's input.
await graph.invoke(new Command({ resume: "hello!" }), threadConfig);
```

```typescript
import { interrupt, Command, MessagesAnnotation } from "@langchain/langgraph";

function humanNode(state: typeof MessagesAnnotation.State): Command {
  /**
   * A node for collecting user input.
   */
  const userInput = interrupt("Ready for user input.");

  // Determine the **active agent** from the state, so
  // we can route to the correct agent after collecting input.
  // For example, add a field to the state or use the last active agent.
  // or fill in `name` attribute of AI messages generated by the agents.
  const activeAgent = ...;

  return new Command({
    goto: activeAgent,
    update: {
      messages: [{
        role: "human",
        content: userInput,
      }]
    }
  });
}
```

```typescript
import { interrupt } from "@langchain/langgraph";

function humanNode(state: typeof GraphAnnotation.State) {
  /**
   * Human node with validation.
   */
  let question = "What is your age?";

  while (true) {
    const answer = interrupt(question);

    // Validate answer, if the answer isn't valid ask for input again.
    if (typeof answer !== "number" || answer < 0) {
      question = `'${answer}' is not a valid age. What is your age?`;
      continue;
    } else {
      // If the answer is valid, we can proceed.
      break;
    }
  }

  console.log(`The human in the loop is ${answer} years old.`);

  return {
    age: answer,
  };
}
```

```typescript
// Resume graph execution with the user's input.
await graph.invoke(new Command({ resume: { age: "25" } }), threadConfig);
```

```typescript
// Update the graph state and resume.
// You must provide a `resume` value if using an `interrupt`.
await graph.invoke(
  new Command({ resume: "Let's go!!!", update: { foo: "bar" } }),
  threadConfig
);
```

```typescript
// Run the graph up to the interrupt
const result = await graph.invoke(inputs, threadConfig);

// Get the graph state to get interrupt information.
const state = await graph.getState(threadConfig);

// Print the state values
console.log(state.values);

// Print the pending tasks
console.log(state.tasks);

// Resume the graph with the user's input.
await graph.invoke(new Command({ resume: { age: "25" } }), threadConfig);
```

```typescript
{
  foo: "bar";
} // State values

[
  {
    id: "5d8ffc92-8011-0c9b-8b59-9d3545b7e553",
    name: "node_foo",
    path: ["__pregel_pull", "node_foo"],
    error: null,
    interrupts: [
      {
        value: "value_in_interrupt",
        resumable: true,
        ns: ["node_foo:5d8ffc92-8011-0c9b-8b59-9d3545b7e553"],
        when: "during",
      },
    ],
    state: null,
    result: null,
  },
]; // Pending tasks. interrupts
```

```typescript
let counter = 0;

function node(state: State) {
  // All the code from the beginning of the node to the interrupt will be re-executed
  // when the graph resumes.
  counter += 1;

  console.log(`> Entered the node: ${counter} # of times`);

  // Pause the graph and wait for user input.
  const answer = interrupt();

  console.log("The value of counter is:", counter);
  // ...
}
```

```typescript
> Entered the node: 2 # of times
The value of counter is: 2
```

```typescript
import { interrupt } from "@langchain/langgraph";

function humanNode(state: typeof GraphAnnotation.State) {
  /**
   * Human node with validation.
   */
  apiCall(); // This code will be re-executed when the node is resumed.

  const answer = interrupt(question);
}
```

```typescript
import { interrupt } from "@langchain/langgraph";

function humanNode(state: typeof GraphAnnotation.State) {
  /**
   * Human node with validation.
   */

  const answer = interrupt(question);

  apiCall(answer); // OK as it's after the interrupt
}
```

```typescript
import { interrupt } from "@langchain/langgraph";

function humanNode(state: typeof GraphAnnotation.State) {
  /**
   * Human node with validation.
   */

  const answer = interrupt(question);

  return {
    answer,
  };
}

function apiCallNode(state: typeof GraphAnnotation.State) {
  apiCall(); // OK as it's in a separate node
}
```

```typescript
async function nodeInParentGraph(state: typeof GraphAnnotation.State) {
    someCode();  // <-- This will re-execute when the subgraph is resumed.
    // Invoke a subgraph as a function.
    // The subgraph contains an `interrupt` call.
    const subgraphResult = await subgraph.invoke(someInput);
    ...
}
```

```typescript
import {
  StateGraph,
  START,
  interrupt,
  Command,
  MemorySaver,
  Annotation,
} from "@langchain/langgraph";

const GraphAnnotation = Annotation.Root({
  stateCounter: Annotation<number>({
    reducer: (a, b) => a + b,
    default: () => 0,
  }),
});

let counterNodeInSubgraph = 0;

function nodeInSubgraph(state: typeof GraphAnnotation.State) {
  counterNodeInSubgraph += 1; // This code will **NOT** run again!
  console.log(
    `Entered 'nodeInSubgraph' a total of ${counterNodeInSubgraph} times`
  );
  return {};
}

let counterHumanNode = 0;

async function humanNode(state: typeof GraphAnnotation.State) {
  counterHumanNode += 1; // This code will run again!
  console.log(
    `Entered humanNode in sub-graph a total of ${counterHumanNode} times`
  );
  const answer = await interrupt("what is your name?");
  console.log(`Got an answer of ${answer}`);
  return {};
}

const checkpointer = new MemorySaver();

const subgraphBuilder = new StateGraph(GraphAnnotation)
  .addNode("some_node", nodeInSubgraph)
  .addNode("human_node", humanNode)
  .addEdge(START, "some_node")
  .addEdge("some_node", "human_node");
const subgraph = subgraphBuilder.compile({ checkpointer });

let counterParentNode = 0;

async function parentNode(state: typeof GraphAnnotation.State) {
  counterParentNode += 1; // This code will run again on resuming!
  console.log(`Entered 'parentNode' a total of ${counterParentNode} times`);

  // Please note that we're intentionally incrementing the state counter
  // in the graph state as well to demonstrate that the subgraph update
  // of the same key will not conflict with the parent graph (until
  const subgraphState = await subgraph.invoke(state);
  return subgraphState;
}

const builder = new StateGraph(GraphAnnotation)
  .addNode("parent_node", parentNode)
  .addEdge(START, "parent_node");

// A checkpointer must be enabled for interrupts to work!
const graph = builder.compile({ checkpointer });

const config = {
  configurable: {
    thread_id: crypto.randomUUID(),
  },
};

for await (const chunk of await graph.stream({ stateCounter: 1 }, config)) {
  console.log(chunk);
}

console.log("--- Resuming ---");

for await (const chunk of await graph.stream(
  new Command({ resume: "35" }),
  config
)) {
  console.log(chunk);
}
```

```typescript
--- First invocation ---
In parent node: { foo: 'bar' }
Entered 'parentNode' a total of 1 times
Entered 'nodeInSubgraph' a total of 1 times
Entered humanNode in sub-graph a total of 1 times
{ __interrupt__: [{ value: 'what is your name?', resumable: true, ns: ['parent_node:0b23d72f-aaba-0329-1a59-ca4f3c8bad3b', 'human_node:25df717c-cb80-57b0-7410-44e20aac8f3c'], when: 'during' }] }

--- Resuming ---
In parent node: { foo: 'bar' }
Entered 'parentNode' a total of 2 times
Entered humanNode in sub-graph a total of 2 times
Got an answer of 35
{ parent_node: null }
```

```typescript
import { v4 as uuidv4 } from "uuid";
import {
  StateGraph,
  MemorySaver,
  START,
  interrupt,
  Command,
  Annotation
} from "@langchain/langgraph";

const GraphAnnotation = Annotation.Root({
  name: Annotation<string>(),
  age: Annotation<string>()
});

function humanNode(state: typeof GraphAnnotation.State) {
  let name;
  if (!state.name) {
    name = interrupt("what is your name?");
  } else {
    name = "N/A";
  }

  let age;
  if (!state.age) {
    age = interrupt("what is your age?");
  } else {
    age = "N/A";
  }

  console.log(`Name: ${name}. Age: ${age}`);

  return {
    age,
    name,
  };
}

const builder = new StateGraph(GraphAnnotation)
  .addNode("human_node", humanNode);
  .addEdge(START, "human_node");

// A checkpointer must be enabled for interrupts to work!
const checkpointer = new MemorySaver();

const graph = builder.compile({ checkpointer });

const config = {
  configurable: {
    thread_id: uuidv4(),
  }
};

for await (const chunk of await graph.stream({ age: undefined, name: undefined }, config)) {
  console.log(chunk);
}

for await (const chunk of await graph.stream(
  Command({ resume: "John", update: { name: "foo" } }),
  config
)) {
  console.log(chunk);
}
```

```typescript
{ __interrupt__: [{
  value: 'what is your name?',
  resumable: true,
  ns: ['human_node:3a007ef9-c30d-c357-1ec1-86a1a70d8fba'],
  when: 'during'
}]}
Name: N/A. Age: John
{ human_node: { age: 'John', name: 'N/A' } }
```

```typescript
const graph = graphBuilder.compile(...);
```

```typescript
import { StateGraph, Annotation } from "@langchain/langgraph";

const State = Annotation.Root({
  foo: Annotation<number>,
  bar: Annotation<string[]>,
});

const graphBuilder = new StateGraph(State);
```

```typescript
import { StateGraph, Annotation } from "@langchain/langgraph";

const State = Annotation.Root({
  foo: Annotation<number>,
  bar: Annotation<string[]>({
    reducer: (state: string[], update: string[]) => state.concat(update),
    default: () => [],
  }),
});

const graphBuilder = new StateGraph(State);
```

```typescript
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";

const graph = new StateGraph(MessagesAnnotation)
  .addNode(...)
  ...
```

```typescript
import { BaseMessage } from "@langchain/core/messages";
import { Annotation, StateGraph, messagesStateReducer } from "@langchain/langgraph";

export const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
});

const graph = new StateGraph(StateAnnotation)
  .addNode(...)
  ...
```

```typescript
import { Annotation, MessagesAnnotation } from "@langchain/langgraph";

const StateWithDocuments = Annotation.Root({
  ...MessagesAnnotation.spec, // Spread in the messages state
  documents: Annotation<string[]>,
});
```

```typescript
import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph, Annotation } from "@langchain/langgraph";

const GraphAnnotation = Annotation.Root({
  input: Annotation<string>,
  results: Annotation<string>,
});

// The state type can be extracted using `typeof <annotation variable name>.State`
const myNode = (state: typeof GraphAnnotation.State, config?: RunnableConfig) => {
  console.log("In node: ", config.configurable?.user_id);
  return {
    results: `Hello, ${state.input}!`
  };
};

// The second argument is optional
const myOtherNode = (state: typeof GraphAnnotation.State) => {
  return state;
};

const builder = new StateGraph(GraphAnnotation)
  .addNode("myNode", myNode)
  .addNode("myOtherNode", myOtherNode)
  ...
```

```typescript
import { START } from "@langchain/langgraph";

graph.addEdge(START, "nodeA");
```

```typescript
import { END } from "@langchain/langgraph";

graph.addEdge("nodeA", END);
```

```typescript
graph.addEdge("nodeA", "nodeB");
```

```typescript
graph.addConditionalEdges("nodeA", routingFunction);
```

```typescript
graph.addConditionalEdges("nodeA", routingFunction, {
  true: "nodeB",
  false: "nodeC",
});
```

```typescript
import { START } from "@langchain/langgraph";

graph.addEdge(START, "nodeA");
```

```typescript
import { START } from "@langchain/langgraph";

graph.addConditionalEdges(START, routingFunction);
```

```typescript
graph.addConditionalEdges(START, routingFunction, {
  true: "nodeB",
  false: "nodeC",
});
```

```typescript
const continueToJokes = (state: { subjects: string[] }) => {
  return state.subjects.map(
    (subject) => new Send("generate_joke", { subject })
  );
};

graph.addConditionalEdges("nodeA", continueToJokes);
```

```typescript
const config = { configurable: { llm: "anthropic" } };

await graph.invoke(inputs, config);
```

```typescript
const nodeA = (state, config) => {
  const llmType = config?.configurable?.llm;
  let llm: BaseChatModel;
  if (llmType) {
    const llm = getLlm(llmType);
  }
  ...
};

```

```typescript
const config = { configurable: { thread_id: "foo" } };

// Initial run of graph
await graph.invoke(inputs, config);

// Let's assume it hit a breakpoint somewhere, you can then resume by passing in None
await graph.invoke(null, config);
```

```typescript
function myNode(
  state: typeof GraphAnnotation.State
): typeof GraphAnnotation.State {
  if (state.input.length > 5) {
    throw new NodeInterrupt(
      `Received input that is longer than 5 characters: ${state.input}`
    );
  }

  return state;
}
```
