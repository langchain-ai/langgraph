# `langgraph`

## Get started

`pip install langgraph`

## Overview

LangGraph is an alpha-stage library for building stateful, multi-actor applications with LLMs. It extends the [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) with the ability to coordinate multiple chains (or actors) across multiple steps of computation. It is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/).

Some of the use cases are:

- Recursive/iterative LLM chains
- LLM chains with persistent state/memory
- LLM agents
- Multi-agent simulations
- ...and more!

## How it works

### Channels

Channels are used to communicate between chains. Each channel has a value type, an update type, and an update function – which takes a sequence of updates and modifies the stored value. Channels can be used to send data from one chain to another, or to send data from a chain to itself in a future step. LangGraph provides a number of built-in channels:

#### Basic channels: LastValue and Topic

- `LastValue`: The default channel, stores the last value sent to the channel, useful for input and output values, or for sending data from one step to the next
- `Topic`: A configurable PubSub Topic, useful for sending multiple values between chains, or for accumulating output. Can be configured to deduplicate values, and/or to accummulate values over the course of multiple steps.

#### Advanced channels: Context and BinaryOperatorAggregate

- `Context`: exposes the value of a context manager, managing its lifecycle. Useful for accessing external resources that require setup and/or teardown. eg. `client = Context(httpx.Client)`
- `BinaryOperatorAggregate`: stores a persistent value, updated by applying a binary operator to the current value and each update sent to the channel, useful for computing aggregates over multiple steps. eg. `total = BinaryOperatorAggregate(int, operator.add)`

### Chains

Chains are LCEL Runnables which subscribe to one or more channels, and write to one or more channels. Any valid LCEL expression can be used as a chain. Chains can be combined into a Pregel application, which coordinates the execution of the chains across multiple steps.

### Pregel

Pregel combines multiple chains (or actors) into a single application. It coordinates the execution of the chains across multiple steps, following the Pregel/Bulk Synchronous Parallel model. Each step consists of three phases:

- **Plan**: Determine which chains to execute in this step, ie. the chains that subscribe to channels updated in the previous step (or, in the first step, chains that subscribe to input channels)
- **Execution**: Execute those chains in parallel, until all complete, or one fails, or a timeout is reached. Any channel updates are invisible to other chains until the next step.
- **Update**: Update the channels with the values written by the chains in this step.

Repeat until no chains are planned for execution, or a maximum number of steps is reached.

## Example

```python
from langgraph import Channel, Pregel

grow_value = (
    Channel.subscribe_to("value")
    | (lambda x: x + x)
    | Channel.write_to(value=lambda x: x if len(x) < 10 else None)
)

app = Pregel(
    chains={"grow_value": grow_value},
    input="value",
    output="value",
)

assert app.invoke("a") == "aaaaaaaa"
```

Check `examples` for more examples.

## Near-term Roadmap

- [x] Iterate on API
  - [x] do we want api to receive output from multiple channels in invoke()
  - [x] do we want api to send input to multiple channels in invoke()
  - [x] Finish updating tests to new API
- [x] Implement input_schema and output_schema in Pregel
- [ ] More tests
  - [x] Test different input and output types (str, str sequence)
  - [x] Add tests for Stream, UniqueInbox
  - [ ] Add tests for subscribe_to_each().join()
- [x] Add optional debug logging
- [ ] Add an optional Diff value for Channels that implements `__add__`, returned by update(), yielded by Pregel for output channels. Add replacing_keys set to AddableDict. use an addabledict for yielding values. channels that dont implement it get marked with replacing_keys
- [x] Implement checkpointing
  - [x] Save checkpoints at end of each step/run
  - [x] Load checkpoint at start of invocation
  - [x] API to specify storage backend and save key
  - [x] Tests
- [ ] Add more examples
  - [ ] multi agent simulation
  - [ ] human in the loop
  - [ ] combine documents
  - [ ] agent executor (add current v total iterations info to read/write steps to enable doing a final update at the end)
  - [ ] run over dataset
- [ ] Fault tolerance
  - [ ] Expose a unique id to each step, hash of (app, chain, checkpoint) (include input updates for first step)
  - [ ] Retry individual processes in a step
  - [ ] Retry entire step?
- [ ] Pregel.stream_log to contain additional keys specific to Pregel
  - [ ] tasks: inputs of each chain in each step, keyed by {name}:{step}
  - [ ] task_results: same as above but outputs
  - [ ] channels: channel values at end of each step, keyed by {name}:{step}
