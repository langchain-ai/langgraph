[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / RunsClient

# Class: RunsClient\<TStateType, TUpdateType, TCustomEventType\>

Defined in: [client.ts:912](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L912)

## Extends

- `BaseClient`

## Type Parameters

• **TStateType** = `DefaultValues`

• **TUpdateType** = `TStateType`

• **TCustomEventType** = `unknown`

## Constructors

### new RunsClient()

> **new RunsClient**\<`TStateType`, `TUpdateType`, `TCustomEventType`\>(`config`?): [`RunsClient`](RunsClient.md)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:183](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L183)

#### Parameters

##### config?

[`ClientConfig`](../interfaces/ClientConfig.md)

#### Returns

[`RunsClient`](RunsClient.md)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

#### Inherited from

`BaseClient.constructor`

## Methods

### cancel()

> **cancel**(`threadId`, `runId`, `wait`, `action`): `Promise`\<`void`\>

Defined in: [client.ts:1227](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1227)

Cancel a run.

#### Parameters

##### threadId

`string`

The ID of the thread.

##### runId

`string`

The ID of the run.

##### wait

`boolean` = `false`

Whether to block when canceling

##### action

`CancelAction` = `"interrupt"`

Action to take when cancelling the run. Possible values are `interrupt` or `rollback`. Default is `interrupt`.

#### Returns

`Promise`\<`void`\>

***

### create()

> **create**(`threadId`, `assistantId`, `payload`?): `Promise`\<`Run`\>

Defined in: [client.ts:1026](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1026)

Create a run.

#### Parameters

##### threadId

`string`

The ID of the thread.

##### assistantId

`string`

Assistant ID to use for this run.

##### payload?

`RunsCreatePayload`

Payload for creating a run.

#### Returns

`Promise`\<`Run`\>

The created run.

***

### createBatch()

> **createBatch**(`payloads`): `Promise`\<`Run`[]\>

Defined in: [client.ts:1076](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1076)

Create a batch of stateless background runs.

#### Parameters

##### payloads

`RunsCreatePayload` & `object`[]

An array of payloads for creating runs.

#### Returns

`Promise`\<`Run`[]\>

An array of created runs.

***

### delete()

> **delete**(`threadId`, `runId`): `Promise`\<`void`\>

Defined in: [client.ts:1329](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1329)

Delete a run.

#### Parameters

##### threadId

`string`

The ID of the thread.

##### runId

`string`

The ID of the run.

#### Returns

`Promise`\<`void`\>

***

### get()

> **get**(`threadId`, `runId`): `Promise`\<`Run`\>

Defined in: [client.ts:1214](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1214)

Get a run by ID.

#### Parameters

##### threadId

`string`

The ID of the thread.

##### runId

`string`

The ID of the run.

#### Returns

`Promise`\<`Run`\>

The run.

***

### join()

> **join**(`threadId`, `runId`, `options`?): `Promise`\<`void`\>

Defined in: [client.ts:1249](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1249)

Block until a run is done.

#### Parameters

##### threadId

`string`

The ID of the thread.

##### runId

`string`

The ID of the run.

##### options?

###### signal?

`AbortSignal`

#### Returns

`Promise`\<`void`\>

***

### joinStream()

> **joinStream**(`threadId`, `runId`, `options`?): `AsyncGenerator`\<\{ `data`: `any`; `event`: `StreamEvent`; `id`: `string`; \}\>

Defined in: [client.ts:1274](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1274)

Stream output from a run in real-time, until the run is done.

#### Parameters

##### threadId

The ID of the thread. Can be set to `null` | `undefined` for stateless runs.

`undefined` | `null` | `string`

##### runId

`string`

The ID of the run.

##### options?

Additional options for controlling the stream behavior:
  - signal: An AbortSignal that can be used to cancel the stream request
  - lastEventId: The ID of the last event received. Can be used to reconnect to a stream without losing events.
  - cancelOnDisconnect: When true, automatically cancels the run if the client disconnects from the stream
  - streamMode: Controls what types of events to receive from the stream (can be a single mode or array of modes)
       Must be a subset of the stream modes passed when creating the run. Background runs default to having the union of all
       stream modes enabled.

`AbortSignal` | \{ `cancelOnDisconnect`: `boolean`; `lastEventId`: `string`; `signal`: `AbortSignal`; `streamMode`: `StreamMode` \| `StreamMode`[]; \}

#### Returns

`AsyncGenerator`\<\{ `data`: `any`; `event`: `StreamEvent`; `id`: `string`; \}\>

An async generator yielding stream parts.

***

### list()

> **list**(`threadId`, `options`?): `Promise`\<`Run`[]\>

Defined in: [client.ts:1177](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1177)

List all runs for a thread.

#### Parameters

##### threadId

`string`

The ID of the thread.

##### options?

Filtering and pagination options.

###### limit?

`number`

Maximum number of runs to return.
Defaults to 10

###### offset?

`number`

Offset to start from.
Defaults to 0.

###### status?

`RunStatus`

Status of the run to filter by.

#### Returns

`Promise`\<`Run`[]\>

List of runs.

***

### stream()

Create a run and stream the results.

#### Param

The ID of the thread.

#### Param

Assistant ID to use for this run.

#### Param

Payload for creating a run.

#### Call Signature

> **stream**\<`TStreamMode`, `TSubgraphs`\>(`threadId`, `assistantId`, `payload`?): `TypedAsyncGenerator`\<`TStreamMode`, `TSubgraphs`, `TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:917](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L917)

##### Type Parameters

• **TStreamMode** *extends* `StreamMode` \| `StreamMode`[] = `StreamMode`

• **TSubgraphs** *extends* `boolean` = `false`

##### Parameters

###### threadId

`null`

###### assistantId

`string`

###### payload?

`Omit`\<`RunsStreamPayload`\<`TStreamMode`, `TSubgraphs`\>, `"multitaskStrategy"` \| `"onCompletion"`\>

##### Returns

`TypedAsyncGenerator`\<`TStreamMode`, `TSubgraphs`, `TStateType`, `TUpdateType`, `TCustomEventType`\>

#### Call Signature

> **stream**\<`TStreamMode`, `TSubgraphs`\>(`threadId`, `assistantId`, `payload`?): `TypedAsyncGenerator`\<`TStreamMode`, `TSubgraphs`, `TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:935](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L935)

##### Type Parameters

• **TStreamMode** *extends* `StreamMode` \| `StreamMode`[] = `StreamMode`

• **TSubgraphs** *extends* `boolean` = `false`

##### Parameters

###### threadId

`string`

###### assistantId

`string`

###### payload?

`RunsStreamPayload`\<`TStreamMode`, `TSubgraphs`\>

##### Returns

`TypedAsyncGenerator`\<`TStreamMode`, `TSubgraphs`, `TStateType`, `TUpdateType`, `TCustomEventType`\>

***

### wait()

Create a run and wait for it to complete.

#### Param

The ID of the thread.

#### Param

Assistant ID to use for this run.

#### Param

Payload for creating a run.

#### Call Signature

> **wait**(`threadId`, `assistantId`, `payload`?): `Promise`\<`DefaultValues`\>

Defined in: [client.ts:1093](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1093)

##### Parameters

###### threadId

`null`

###### assistantId

`string`

###### payload?

`Omit`\<`RunsWaitPayload`, `"multitaskStrategy"` \| `"onCompletion"`\>

##### Returns

`Promise`\<`DefaultValues`\>

#### Call Signature

> **wait**(`threadId`, `assistantId`, `payload`?): `Promise`\<`DefaultValues`\>

Defined in: [client.ts:1099](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1099)

##### Parameters

###### threadId

`string`

###### assistantId

`string`

###### payload?

`RunsWaitPayload`

##### Returns

`Promise`\<`DefaultValues`\>
