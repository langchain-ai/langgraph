[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / UseStreamOptions

# Interface: UseStreamOptions\<StateType, Bag\>

Defined in: [react/stream.tsx:413](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L413)

## Type Parameters

• **StateType** *extends* `Record`\<`string`, `unknown`\> = `Record`\<`string`, `unknown`\>

• **Bag** *extends* `BagTemplate` = `BagTemplate`

## Properties

### apiKey?

> `optional` **apiKey**: `string`

Defined in: [react/stream.tsx:435](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L435)

The API key to use.

***

### apiUrl?

> `optional` **apiUrl**: `string`

Defined in: [react/stream.tsx:430](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L430)

The URL of the API to use.

***

### assistantId

> **assistantId**: `string`

Defined in: [react/stream.tsx:420](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L420)

The ID of the assistant to use.

***

### callerOptions?

> `optional` **callerOptions**: `AsyncCallerParams`

Defined in: [react/stream.tsx:440](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L440)

Custom call options, such as custom fetch implementation.

***

### client?

> `optional` **client**: `Client`\<`DefaultValues`, `DefaultValues`, `unknown`\>

Defined in: [react/stream.tsx:425](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L425)

Client used to send requests.

***

### defaultHeaders?

> `optional` **defaultHeaders**: `Record`\<`string`, `HeaderValue`\>

Defined in: [react/stream.tsx:445](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L445)

Default headers to send with requests.

***

### messagesKey?

> `optional` **messagesKey**: `string`

Defined in: [react/stream.tsx:453](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L453)

Specify the key within the state that contains messages.
Defaults to "messages".

#### Default

```ts
"messages"
```

***

### onCreated()?

> `optional` **onCreated**: (`run`) => `void`

Defined in: [react/stream.tsx:471](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L471)

Callback that is called when a new stream is created.

#### Parameters

##### run

`RunCallbackMeta`

#### Returns

`void`

***

### onCustomEvent()?

> `optional` **onCustomEvent**: (`data`, `options`) => `void`

Defined in: [react/stream.tsx:483](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L483)

Callback that is called when a custom event is received.

#### Parameters

##### data

`GetCustomEventType`\<`Bag`\>

##### options

###### mutate

(`update`) => `void`

#### Returns

`void`

***

### onDebugEvent()?

> `optional` **onDebugEvent**: (`data`) => `void`

Defined in: [react/stream.tsx:507](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L507)

**`Internal`**

Callback that is called when a debug event is received.
 This API is experimental and subject to change.

#### Parameters

##### data

`unknown`

#### Returns

`void`

***

### onError()?

> `optional` **onError**: (`error`, `run`) => `void`

Defined in: [react/stream.tsx:458](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L458)

Callback that is called when an error occurs.

#### Parameters

##### error

`unknown`

##### run

`undefined` | `RunCallbackMeta`

#### Returns

`void`

***

### onFinish()?

> `optional` **onFinish**: (`state`, `run`) => `void`

Defined in: [react/stream.tsx:463](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L463)

Callback that is called when the stream is finished.

#### Parameters

##### state

`ThreadState`\<`StateType`\>

##### run

`undefined` | `RunCallbackMeta`

#### Returns

`void`

***

### onLangChainEvent()?

> `optional` **onLangChainEvent**: (`data`) => `void`

Defined in: [react/stream.tsx:501](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L501)

Callback that is called when a LangChain event is received.

#### Parameters

##### data

###### data

`unknown`

###### event

`string` & `object` \| `"on_tool_start"` \| `"on_tool_stream"` \| `"on_tool_end"` \| `"on_chat_model_start"` \| `"on_chat_model_stream"` \| `"on_chat_model_end"` \| `"on_llm_start"` \| `"on_llm_stream"` \| `"on_llm_end"` \| `"on_chain_start"` \| `"on_chain_stream"` \| `"on_chain_end"` \| `"on_retriever_start"` \| `"on_retriever_stream"` \| `"on_retriever_end"` \| `"on_prompt_start"` \| `"on_prompt_stream"` \| `"on_prompt_end"`

###### metadata

`Record`\<`string`, `unknown`\>

###### name

`string`

###### parent_ids

`string`[]

###### run_id

`string`

###### tags

`string`[]

#### Returns

`void`

#### See

https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_events/#stream-graph-in-events-mode for more details.

***

### onMetadataEvent()?

> `optional` **onMetadataEvent**: (`data`) => `void`

Defined in: [react/stream.tsx:495](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L495)

Callback that is called when a metadata event is received.

#### Parameters

##### data

###### run_id

`string`

###### thread_id

`string`

#### Returns

`void`

***

### onThreadId()?

> `optional` **onThreadId**: (`threadId`) => `void`

Defined in: [react/stream.tsx:517](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L517)

Callback that is called when the thread ID is updated (ie when a new thread is created).

#### Parameters

##### threadId

`string`

#### Returns

`void`

***

### onUpdateEvent()?

> `optional` **onUpdateEvent**: (`data`) => `void`

Defined in: [react/stream.tsx:476](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L476)

Callback that is called when an update event is received.

#### Parameters

##### data

#### Returns

`void`

***

### reconnectOnMount?

> `optional` **reconnectOnMount**: `boolean` \| () => `RunMetadataStorage`

Defined in: [react/stream.tsx:520](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L520)

Will reconnect the stream on mount

***

### threadId?

> `optional` **threadId**: `null` \| `string`

Defined in: [react/stream.tsx:512](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L512)

The ID of the thread to fetch history and current values from.
