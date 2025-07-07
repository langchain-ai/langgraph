[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / ThreadsClient

# Class: ThreadsClient\<TStateType, TUpdateType\>

Defined in: [client.ts:626](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L626)

## Extends

- `BaseClient`

## Type Parameters

• **TStateType** = `DefaultValues`

• **TUpdateType** = `TStateType`

## Constructors

### new ThreadsClient()

> **new ThreadsClient**\<`TStateType`, `TUpdateType`\>(`config`?): [`ThreadsClient`](ThreadsClient.md)\<`TStateType`, `TUpdateType`\>

Defined in: [client.ts:183](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L183)

#### Parameters

##### config?

[`ClientConfig`](../interfaces/ClientConfig.md)

#### Returns

[`ThreadsClient`](ThreadsClient.md)\<`TStateType`, `TUpdateType`\>

#### Inherited from

`BaseClient.constructor`

## Methods

### copy()

> **copy**(`threadId`): `Promise`\<`Thread`\<`TStateType`\>\>

Defined in: [client.ts:703](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L703)

Copy an existing thread

#### Parameters

##### threadId

`string`

ID of the thread to be copied

#### Returns

`Promise`\<`Thread`\<`TStateType`\>\>

Newly copied thread

***

### create()

> **create**(`payload`?): `Promise`\<`Thread`\<`TStateType`\>\>

Defined in: [client.ts:648](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L648)

Create a new thread.

#### Parameters

##### payload?

Payload for creating a thread.

###### graphId?

`string`

Graph ID to associate with the thread.

###### ifExists?

`OnConflictBehavior`

How to handle duplicate creation.

**Default**

```ts
"raise"
```

###### metadata?

`Metadata`

Metadata for the thread.

###### supersteps?

`object`[]

Apply a list of supersteps when creating a thread, each containing a sequence of updates.

Used for copying a thread between deployments.

###### threadId?

`string`

ID of the thread to create.

If not provided, a random UUID will be generated.

#### Returns

`Promise`\<`Thread`\<`TStateType`\>\>

The created thread.

***

### delete()

> **delete**(`threadId`): `Promise`\<`void`\>

Defined in: [client.ts:736](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L736)

Delete a thread.

#### Parameters

##### threadId

`string`

ID of the thread.

#### Returns

`Promise`\<`void`\>

***

### get()

> **get**\<`ValuesType`\>(`threadId`): `Promise`\<`Thread`\<`ValuesType`\>\>

Defined in: [client.ts:636](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L636)

Get a thread by ID.

#### Type Parameters

• **ValuesType** = `TStateType`

#### Parameters

##### threadId

`string`

ID of the thread.

#### Returns

`Promise`\<`Thread`\<`ValuesType`\>\>

The thread.

***

### getHistory()

> **getHistory**\<`ValuesType`\>(`threadId`, `options`?): `Promise`\<`ThreadState`\<`ValuesType`\>[]\>

Defined in: [client.ts:888](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L888)

Get all past states for a thread.

#### Type Parameters

• **ValuesType** = `TStateType`

#### Parameters

##### threadId

`string`

ID of the thread.

##### options?

Additional options.

###### before?

`Config`

###### checkpoint?

`Partial`\<`Omit`\<`Checkpoint`, `"thread_id"`\>\>

###### limit?

`number`

###### metadata?

`Metadata`

#### Returns

`Promise`\<`ThreadState`\<`ValuesType`\>[]\>

List of thread states.

***

### getState()

> **getState**\<`ValuesType`\>(`threadId`, `checkpoint`?, `options`?): `Promise`\<`ThreadState`\<`ValuesType`\>\>

Defined in: [client.ts:795](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L795)

Get state for a thread.

#### Type Parameters

• **ValuesType** = `TStateType`

#### Parameters

##### threadId

`string`

ID of the thread.

##### checkpoint?

`string` | `Checkpoint`

##### options?

###### subgraphs?

`boolean`

#### Returns

`Promise`\<`ThreadState`\<`ValuesType`\>\>

Thread state.

***

### patchState()

> **patchState**(`threadIdOrConfig`, `metadata`): `Promise`\<`void`\>

Defined in: [client.ts:858](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L858)

Patch the metadata of a thread.

#### Parameters

##### threadIdOrConfig

Thread ID or config to patch the state of.

`string` | `Config`

##### metadata

`Metadata`

Metadata to patch the state with.

#### Returns

`Promise`\<`void`\>

***

### search()

> **search**\<`ValuesType`\>(`query`?): `Promise`\<`Thread`\<`ValuesType`\>[]\>

Defined in: [client.ts:748](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L748)

List threads

#### Type Parameters

• **ValuesType** = `TStateType`

#### Parameters

##### query?

Query options

###### limit?

`number`

Maximum number of threads to return.
Defaults to 10

###### metadata?

`Metadata`

Metadata to filter threads by.

###### offset?

`number`

Offset to start from.

###### sortBy?

`ThreadSortBy`

Sort by.

###### sortOrder?

`SortOrder`

Sort order.
Must be one of 'asc' or 'desc'.

###### status?

`ThreadStatus`

Thread status to filter on.

#### Returns

`Promise`\<`Thread`\<`ValuesType`\>[]\>

List of threads

***

### update()

> **update**(`threadId`, `payload`?): `Promise`\<`Thread`\<`DefaultValues`\>\>

Defined in: [client.ts:716](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L716)

Update a thread.

#### Parameters

##### threadId

`string`

ID of the thread.

##### payload?

Payload for updating the thread.

###### metadata?

`Metadata`

Metadata for the thread.

#### Returns

`Promise`\<`Thread`\<`DefaultValues`\>\>

The updated thread.

***

### updateState()

> **updateState**\<`ValuesType`\>(`threadId`, `options`): `Promise`\<`Pick`\<`Config`, `"configurable"`\>\>

Defined in: [client.ts:829](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L829)

Add state to a thread.

#### Type Parameters

• **ValuesType** = `TUpdateType`

#### Parameters

##### threadId

`string`

The ID of the thread.

##### options

###### asNode?

`string`

###### checkpoint?

`Checkpoint`

###### checkpointId?

`string`

###### values

`ValuesType`

#### Returns

`Promise`\<`Pick`\<`Config`, `"configurable"`\>\>
