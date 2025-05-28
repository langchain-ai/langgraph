---
search:
  boost: 2
---

# Breakpoints

Breakpoints pause graph execution at defined points and let you step through each stage. They use LangGraph's [**persistence layer**](./persistence.md), which saves the graph state after each step.

With breakpoints, you can inspect the graph's state and node inputs at any point. Execution pauses **indefinitely** until you resume, as the checkpointer preserves the state.

<figure markdown="1">
![image](img/breakpoints.png){: style="max-height:400px"}
<figcaption>An example graph consisting of 3 sequential steps with a breakpoint before step_3. </figcaption> </figure>
