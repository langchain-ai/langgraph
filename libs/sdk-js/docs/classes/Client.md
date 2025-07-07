[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / Client

# Class: Client\<TStateType, TUpdateType, TCustomEventType\>

Defined in: [client.ts:1620](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1620)

## Type Parameters

• **TStateType** = `DefaultValues`

• **TUpdateType** = `TStateType`

• **TCustomEventType** = `unknown`

## Constructors

### new Client()

> **new Client**\<`TStateType`, `TUpdateType`, `TCustomEventType`\>(`config`?): [`Client`](Client.md)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:1656](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1656)

#### Parameters

##### config?

[`ClientConfig`](../interfaces/ClientConfig.md)

#### Returns

[`Client`](Client.md)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

## Properties

### ~ui

> **~ui**: `UiClient`

Defined in: [client.ts:1654](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1654)

**`Internal`**

The client for interacting with the UI.
 Used by LoadExternalComponent and the API might change in the future.

***

### assistants

> **assistants**: [`AssistantsClient`](AssistantsClient.md)

Defined in: [client.ts:1628](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1628)

The client for interacting with assistants.

***

### crons

> **crons**: [`CronsClient`](CronsClient.md)

Defined in: [client.ts:1643](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1643)

The client for interacting with cron runs.

***

### runs

> **runs**: [`RunsClient`](RunsClient.md)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:1638](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1638)

The client for interacting with runs.

***

### store

> **store**: [`StoreClient`](StoreClient.md)

Defined in: [client.ts:1648](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1648)

The client for interacting with the KV store.

***

### threads

> **threads**: [`ThreadsClient`](ThreadsClient.md)\<`TStateType`, `TUpdateType`\>

Defined in: [client.ts:1633](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1633)

The client for interacting with threads.
