---
search:
  boost: 2
tags:
  - human-in-the-loop
  - hil
  - overview
hide:
  - tags
---

# Human-in-the-loop

LangGraph supports interactive workflows through persistent execution, enabling humans to inspect, intervene, and steer execution at any point. This includes both application-facing workflows—such as human-in-the-loop (HIL) review—and development-facing tools like breakpoints and time travel.

## Core capabilities

### Human-in-the-loop (HIL)

Human-in-the-loop allows human intervention during graph execution. Using the `interrupt` function, the graph can pause at any node, surface data to a human, and later resume with their input. This is critical for:

* Reviewing or editing LLM-generated outputs
* Approving or modifying tool invocations
* Providing additional context in multi-turn conversations

Because graph state is persisted via LangGraph’s checkpointing system, execution can pause **indefinitely** until human input is provided.

### Breakpoints

Breakpoints enable step-by-step inspection of a graph’s execution. They pause the graph before designated nodes, allowing developers to inspect inputs and state before continuing. Like HIL, breakpoints rely on persistent checkpoints and do not require real-time resumption.

Breakpoints are primarily used during development and debugging to:

* Step through logic incrementally
* Verify intermediate state transformations
* Troubleshoot unexpected behavior

### Time travel

LangGraph supports "time travel" by allowing you to resume execution from any prior checkpoint. This creates a **new fork** of execution, enabling safe experimentation.

Common use cases:

* Debug past errors by replaying earlier states
* Modify state to explore alternative paths
* Analyze model behavior and decision-making

Time travel complements HIL and breakpoints by offering a powerful way to inspect and test non-deterministic workflows, especially those involving LLMs or agents.

## Summary

| Capability        | Purpose                           | Target User      | Resumable? | Requires Real-Time? |
|-------------------|-----------------------------------|------------------|------------|---------------------|
| HIL (`interrupt`) | Human reviews, approvals, edits   | Application user | ✅ Yes      | ❌ No                |
| Breakpoints       | Debugging, step-through execution | Developer        | ✅ Yes      | ❌ No                |
| Time travel       | Debugging, branching alternatives | Developer        | ✅ Yes      | ❌ No                |

All three features rely on **LangGraph’s persistence layer**, which captures graph state after each step and enables precise, controlled resumption.