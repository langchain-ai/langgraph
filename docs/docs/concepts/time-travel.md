---
search:
  boost: 2
---

# Time Travel ⏱️

When working with non-deterministic systems that make model-based decisions (e.g., agents powered by LLMs), it can be useful to examine their decision-making process in detail:

1. 🤔 **Understand reasoning**: Analyze the steps that led to a successful result.
2. 🐞 **Debug mistakes**: Identify where and why errors occurred.
3. 🔍 **Explore alternatives**: Test different paths to uncover better solutions.

LangGraph provides [time travel functionality](../how-tos/human_in_the_loop/time-travel.md) to support these use cases. Specifically, you can resume execution from a prior checkpoint — either replaying the same state or modifying it to explore alternatives. In all cases, resuming past execution produces a new fork in the history.

!!! tip

    For information on how to use time travel, see [Use time travel](../how-tos/human_in_the_loop/time-travel.md) and [Time travel using Server API](../cloud/how-tos/human_in_the_loop_time_travel.md).
