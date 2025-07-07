[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / MessageMetadata

# Type Alias: MessageMetadata\<StateType\>

> **MessageMetadata**\<`StateType`\>: `object`

Defined in: [react/stream.tsx:169](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L169)

## Type Parameters

• **StateType** *extends* `Record`\<`string`, `unknown`\>

## Type declaration

### branch

> **branch**: `string` \| `undefined`

The branch of the message.

### branchOptions

> **branchOptions**: `string`[] \| `undefined`

The list of branches this message is part of.
This is useful for displaying branching controls.

### firstSeenState

> **firstSeenState**: `ThreadState`\<`StateType`\> \| `undefined`

The first thread state the message was seen in.

### messageId

> **messageId**: `string`

The ID of the message used.
