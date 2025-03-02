# Python-Java Implementation Mapping

This document records the mapping between Python and Java implementations of LangGraph, highlighting any deliberate differences and their rationale.

## Core Components

### Channels

| Component | Python Path | Java Path | Deviations |
|-----------|-------------|-----------|------------|
| BaseChannel | langgraph/channels/base.py | com.langgraph.channels.BaseChannel | Java uses interface with default methods instead of Python's abstract base class. Channel returns null or empty values when uninitialized, rather than throwing exceptions. |
| AbstractChannel | langgraph/channels/base.py | com.langgraph.channels.AbstractChannel | Java implementation provides default functionality shared by channel implementations. Added Python compatibility for uninitialized channels. |
| TopicChannel | langgraph/channels/topic_channel.py | com.langgraph.channels.TopicChannel | Java implementation preserves Python's multi-value behavior while using Java collections. Returns empty list for uninitialized channels. |
| LastValue | langgraph/channels/last_value.py | com.langgraph.channels.LastValue | Returns null for uninitialized channels to match Python behavior. |
| EphemeralValue | langgraph/channels/ephemeral_value.py | com.langgraph.channels.EphemeralValue | Returns null for uninitialized channels to match Python behavior. |
| Channels (utility) | langgraph/channels/__init__.py | com.langgraph.channels.Channels | Java uses utility class with static methods instead of module-level functions. |

### Pregel Algorithm

| Component | Python Path | Java Path | Deviations |
|-----------|-------------|-----------|------------|
| PregelNode | langgraph/pregel/algorithm.py | com.langgraph.pregel.PregelNode | Java exposes these concepts with clearer naming: 'channels' (input channels to read from) and 'triggerChannels' (channels that trigger execution). Java now supports multiple trigger channels like Python. |
| Pregel | langgraph/pregel/pregel.py | com.langgraph.pregel.Pregel | Java uses Builder pattern instead of Python's initialization parameters. Functionally equivalent. |
| PregelLoop | langgraph/pregel/pregel_loop.py | com.langgraph.pregel.execute.PregelLoop | Implementation follows Java conventions with robust cycle detection. Ensures runs complete when possible by executing a final validation step before throwing recursion errors. |
| Runner Functions | langgraph/pregel/runner.py | com.langgraph.pregel.execute.SuperstepManager | Python's functional approach mapped to Java's object-oriented design. |
| Algorithm Functions | langgraph/pregel/algo.py | Various Java classes | Python's functional approach distributed across several Java classes according to responsibility. |
| TaskPlanner | langgraph/pregel/algo.py | com.langgraph.pregel.task.TaskPlanner | Java implementation now matches Python: only nodes with the input channel as a trigger execute on first run. See CHANNEL_INITIALIZATION.md for details. |

### Checkpoint

| Component | Python Path | Java Path | Deviations |
|-----------|-------------|-----------|------------|
| BaseCheckpointSaver | langgraph/checkpoint/base.py | com.langgraph.checkpoint.base.BaseCheckpointSaver | Java uses interfaces rather than abstract classes where appropriate. |
| MemoryCheckpointSaver | langgraph/checkpoint/memory.py | com.langgraph.checkpoint.base.memory.MemoryCheckpointSaver | Java implementation uses more type safety but maintains same functionality. |
| Serializer | langgraph/checkpoint/serde.py | com.langgraph.checkpoint.serde.Serializer | Java uses interface with specific implementations for different serialization approaches. |

## Method-Level Mappings

### PregelLoop (Python: langgraph/pregel/loop.py, Java: com.langgraph.pregel.execute.PregelLoop)

| Python Method | Java Method | Deviations |
|---------------|-------------|------------|
| `__init__` | Constructor + Builder pattern | Java uses Builder pattern for more flexible initialization. |
| `tick` | `execute` | Same core functionality, but with improved recursion detection that matches Python behavior while being more resilient. Java executes a final validation step before throwing recursion errors to ensure runs complete when possible. |
| `_first` | `initializeWithInput` | Similar initialization logic but with Java-specific patterns. |
| `stream` | `stream` | Both handle streaming with similar semantics but with improved robustness in Java. Stream mode includes more validation to prevent false recursion errors. |
| `_put_checkpoint` | `createCheckpoint` | Similar checkpoint creation but with Java-specific implementation. |

### Runner Functions (Python: langgraph/pregel/runner.py)

| Python Function | Java Method | Deviations |
|-----------------|-------------|------------|
| `commit` | `SuperstepManager.commit` | Java implementation encapsulates in object instead of standalone function. |
| `tick` | `SuperstepManager.tick` | Same core functionality but adapted to Java's object-oriented paradigm. |

### Algorithm Functions (Python: langgraph/pregel/algo.py)

| Python Function | Java Method | Deviations |
|-----------------|-------------|------------|
| `prepare_next_tasks` | `TaskPlanner.planTasks` | Java implementation encapsulates in object instead of standalone function. |
| `prepare_single_task` | `TaskPlanner.planSingleTask` | Same approach but with stronger typing in Java. |
| `apply_writes` | Multiple methods in ChannelRegistry | Java distributes responsibility across specialized classes. |

## Implementation Notes

### General Patterns
- Java uses more explicit type information compared to Python
- Builder pattern is used in Java where Python uses parameter initialization
- Java collections (List, Map) replace Python collections (list, dict)
- Java follows standard exception hierarchy rather than Python's exception model
- Python's functional approach is often translated to Java's object-oriented design using objects with state
- Uninitialized channels in Java return null or empty collections rather than throwing exceptions
- Nodes in Java follow Python's behavior: only nodes with input channel as a trigger run in the first superstep
- Both implementations handle uninitialized channels gracefully without requiring manual initialization

### Missing Features (To Be Implemented)
- Some stream modes are not yet fully implemented in Java
- Advanced graph features are still under development in Java
- Some error handling cases need refinement to match Python semantics fully

## When Adding New Components
When adding new Java classes that correspond to Python implementations:
1. Add an entry to this document
2. Document any deviations and justify according to allowed reasons:
   - Different public interfaces to match Java developer expectations
   - Different implementation details to match Java stdlib/patterns
   - Not yet fully implemented Python behavior
3. Never introduce deviations just to take shortcuts or change behavior