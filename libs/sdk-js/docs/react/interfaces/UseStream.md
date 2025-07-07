[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / UseStream

# Interface: UseStream\<StateType, Bag\>

Defined in: [react/stream.tsx:529](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L529)

## Type Parameters

• **StateType** *extends* `Record`\<`string`, `unknown`\> = `Record`\<`string`, `unknown`\>

• **Bag** *extends* `BagTemplate` = `BagTemplate`

## Properties

### assistantId

> **assistantId**: `string`

Defined in: [react/stream.tsx:614](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L614)

The ID of the assistant to use.

***

### branch

> **branch**: `string`

Defined in: [react/stream.tsx:564](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L564)

The current branch of the thread.

***

### client

> **client**: `Client`

Defined in: [react/stream.tsx:609](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L609)

LangGraph SDK client used to send request and receive responses.

***

### error

> **error**: `unknown`

Defined in: [react/stream.tsx:541](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L541)

Last seen error from the thread or during streaming.

***

### experimental\_branchTree

> **experimental\_branchTree**: `Sequence`\<`StateType`\>

Defined in: [react/stream.tsx:580](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L580)

**`Experimental`**

Tree of all branches for the thread.

***

### getMessagesMetadata()

> **getMessagesMetadata**: (`message`, `index`?) => `undefined` \| [`MessageMetadata`](../type-aliases/MessageMetadata.md)\<`StateType`\>

Defined in: [react/stream.tsx:601](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L601)

Get the metadata for a message, such as first thread state the message
was seen in and branch information.

#### Parameters

##### message

`Message`

The message to get the metadata for.

##### index?

`number`

The index of the message in the thread.

#### Returns

`undefined` \| [`MessageMetadata`](../type-aliases/MessageMetadata.md)\<`StateType`\>

The metadata for the message.

***

### history

> **history**: `ThreadState`\<`StateType`\>[]

Defined in: [react/stream.tsx:574](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L574)

Flattened history of thread states of a thread.

***

### interrupt

> **interrupt**: `undefined` \| `Interrupt`\<`GetInterruptType`\<`Bag`\>\>

Defined in: [react/stream.tsx:585](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L585)

Get the interrupt value for the stream if interrupted.

***

### isLoading

> **isLoading**: `boolean`

Defined in: [react/stream.tsx:546](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L546)

Whether the stream is currently running.

***

### joinStream()

> **joinStream**: (`runId`) => `Promise`\<`void`\>

Defined in: [react/stream.tsx:619](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L619)

Join an active stream.

#### Parameters

##### runId

`string`

#### Returns

`Promise`\<`void`\>

***

### messages

> **messages**: `Message`[]

Defined in: [react/stream.tsx:591](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L591)

Messages inferred from the thread.
Will automatically update with incoming message chunks.

***

### setBranch()

> **setBranch**: (`branch`) => `void`

Defined in: [react/stream.tsx:569](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L569)

Set the branch of the thread.

#### Parameters

##### branch

`string`

#### Returns

`void`

***

### stop()

> **stop**: () => `void`

Defined in: [react/stream.tsx:551](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L551)

Stops the stream.

#### Returns

`void`

***

### submit()

> **submit**: (`values`, `options`?) => `void`

Defined in: [react/stream.tsx:556](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L556)

Create and stream a run to the thread.

#### Parameters

##### values

`undefined` | `null` | `GetUpdateType`\<`Bag`, `StateType`\>

##### options?

`SubmitOptions`\<`StateType`, `GetConfigurableType`\<`Bag`\>\>

#### Returns

`void`

***

### values

> **values**: `StateType`

Defined in: [react/stream.tsx:536](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/react/stream.tsx#L536)

The current values of the thread.
