---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Evals

To evaluate your agent's performance you can use `LangSmith` [evaluations](https://docs.smith.langchain.com/evaluation). You would need to first define an evaluator function to judge the results from an agent, such as final outputs or trajectory. Depending on your evaluation technique, this may or may not involve a reference output:

:::python

```python
def evaluator(*, outputs: dict, reference_outputs: dict):
    # compare agent outputs against reference outputs
    output_messages = outputs["messages"]
    reference_messages = reference_outputs["messages"]
    score = compare_messages(output_messages, reference_messages)
    return {"key": "evaluator_score", "score": score}
```

:::

:::js

```typescript
type EvaluatorParams = {
  outputs: Record<string, any>;
  referenceOutputs: Record<string, any>;
};

function evaluator({ outputs, referenceOutputs }: EvaluatorParams) {
  // compare agent outputs against reference outputs
  const outputMessages = outputs.messages;
  const referenceMessages = referenceOutputs.messages;
  const score = compareMessages(outputMessages, referenceMessages);
  return { key: "evaluator_score", score: score };
}
```

:::

To get started, you can use prebuilt evaluators from `AgentEvals` package:

:::python

```bash
pip install -U agentevals
```

:::

:::js

```bash
npm install agentevals
```

:::

## Create evaluator

A common way to evaluate agent performance is by comparing its trajectory (the order in which it calls its tools) against a reference trajectory:

:::python

```python
import json
# highlight-next-line
from agentevals.trajectory.match import create_trajectory_match_evaluator

outputs = [
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "san francisco"}),
                }
            },
            {
                "function": {
                    "name": "get_directions",
                    "arguments": json.dumps({"destination": "presidio"}),
                }
            }
        ],
    }
]
reference_outputs = [
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "san francisco"}),
                }
            },
        ],
    }
]

# Create the evaluator
evaluator = create_trajectory_match_evaluator(
    # highlight-next-line
    trajectory_match_mode="superset",  # (1)!
)

# Run the evaluator
result = evaluator(
    outputs=outputs, reference_outputs=reference_outputs
)
```

:::

:::js

```typescript
import { createTrajectoryMatchEvaluator } from "agentevals/trajectory/match";

const outputs = [
  {
    role: "assistant",
    tool_calls: [
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "san francisco" }),
        },
      },
      {
        function: {
          name: "get_directions",
          arguments: JSON.stringify({ destination: "presidio" }),
        },
      },
    ],
  },
];

const referenceOutputs = [
  {
    role: "assistant",
    tool_calls: [
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "san francisco" }),
        },
      },
    ],
  },
];

// Create the evaluator
const evaluator = createTrajectoryMatchEvaluator({
  // Specify how the trajectories will be compared. `superset` will accept output trajectory as valid if it's a superset of the reference one. Other options include: strict, unordered and subset
  trajectoryMatchMode: "superset", // (1)!
});

// Run the evaluator
const result = evaluator({
  outputs: outputs,
  referenceOutputs: referenceOutputs,
});
```

:::

1. Specify how the trajectories will be compared. `superset` will accept output trajectory as valid if it's a superset of the reference one. Other options include: [strict](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#strict-match), [unordered](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#unordered-match) and [subset](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#subset-and-superset-match)

As a next step, learn more about how to [customize trajectory match evaluator](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#agent-trajectory-match).

### LLM-as-a-judge

You can use LLM-as-a-judge evaluator that uses an LLM to compare the trajectory against the reference outputs and output a score:

:::python

```python
import json
from agentevals.trajectory.llm import (
    # highlight-next-line
    create_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
)

evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    model="openai:o3-mini"
)
```

:::

:::js

```typescript
import {
  createTrajectoryLlmAsJudge,
  TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
} from "agentevals/trajectory/llm";

const evaluator = createTrajectoryLlmAsJudge({
  prompt: TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
  model: "openai:o3-mini",
});
```

:::

## Run evaluator

To run an evaluator, you will first need to create a [LangSmith dataset](https://docs.smith.langchain.com/evaluation/concepts#datasets). To use the prebuilt AgentEvals evaluators, you will need a dataset with the following schema:

- **input**: `{"messages": [...]}` input messages to call the agent with.
- **output**: `{"messages": [...]}` expected message history in the agent output. For trajectory evaluation, you can choose to keep only assistant messages.

:::python

```python
from langsmith import Client
from langgraph.prebuilt import create_react_agent
from agentevals.trajectory.match import create_trajectory_match_evaluator

client = Client()
agent = create_react_agent(...)
evaluator = create_trajectory_match_evaluator(...)

experiment_results = client.evaluate(
    lambda inputs: agent.invoke(inputs),
    # replace with your dataset name
    data="<Name of your dataset>",
    evaluators=[evaluator]
)
```

:::

:::js

```typescript
import { Client } from "langsmith";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { createTrajectoryMatchEvaluator } from "agentevals/trajectory/match";

const client = new Client();
const agent = createReactAgent({...});
const evaluator = createTrajectoryMatchEvaluator({...});

const experimentResults = await client.evaluate(
    (inputs) => agent.invoke(inputs),
    // replace with your dataset name
    { data: "<Name of your dataset>" },
    { evaluators: [evaluator] }
);
```

:::
