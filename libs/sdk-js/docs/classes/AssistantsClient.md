[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / AssistantsClient

# Class: AssistantsClient

Defined in: [client.ts:431](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L431)

## Extends

- `BaseClient`

## Constructors

### new AssistantsClient()

> **new AssistantsClient**(`config`?): [`AssistantsClient`](AssistantsClient.md)

Defined in: [client.ts:183](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L183)

#### Parameters

##### config?

[`ClientConfig`](../interfaces/ClientConfig.md)

#### Returns

[`AssistantsClient`](AssistantsClient.md)

#### Inherited from

`BaseClient.constructor`

## Methods

### create()

> **create**(`payload`): `Promise`\<`Assistant`\>

Defined in: [client.ts:496](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L496)

Create a new assistant.

#### Parameters

##### payload

Payload for creating an assistant.

###### assistantId?

`string`

###### config?

`Config`

###### description?

`string`

###### graphId

`string`

###### ifExists?

`OnConflictBehavior`

###### metadata?

`Metadata`

###### name?

`string`

#### Returns

`Promise`\<`Assistant`\>

The created assistant.

***

### delete()

> **delete**(`assistantId`): `Promise`\<`void`\>

Defined in: [client.ts:552](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L552)

Delete an assistant.

#### Parameters

##### assistantId

`string`

ID of the assistant.

#### Returns

`Promise`\<`void`\>

***

### get()

> **get**(`assistantId`): `Promise`\<`Assistant`\>

Defined in: [client.ts:438](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L438)

Get an assistant by ID.

#### Parameters

##### assistantId

`string`

The ID of the assistant.

#### Returns

`Promise`\<`Assistant`\>

Assistant

***

### getGraph()

> **getGraph**(`assistantId`, `options`?): `Promise`\<`AssistantGraph`\>

Defined in: [client.ts:448](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L448)

Get the JSON representation of the graph assigned to a runnable

#### Parameters

##### assistantId

`string`

The ID of the assistant.

##### options?

###### xray?

`number` \| `boolean`

Whether to include subgraphs in the serialized graph representation. If an integer value is provided, only subgraphs with a depth less than or equal to the value will be included.

#### Returns

`Promise`\<`AssistantGraph`\>

Serialized graph

***

### getSchemas()

> **getSchemas**(`assistantId`): `Promise`\<`GraphSchema`\>

Defined in: [client.ts:462](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L462)

Get the state and config schema of the graph assigned to a runnable

#### Parameters

##### assistantId

`string`

The ID of the assistant.

#### Returns

`Promise`\<`GraphSchema`\>

Graph schema

***

### getSubgraphs()

> **getSubgraphs**(`assistantId`, `options`?): `Promise`\<`Subgraphs`\>

Defined in: [client.ts:473](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L473)

Get the schemas of an assistant by ID.

#### Parameters

##### assistantId

`string`

The ID of the assistant to get the schema of.

##### options?

Additional options for getting subgraphs, such as namespace or recursion extraction.

###### namespace?

`string`

###### recurse?

`boolean`

#### Returns

`Promise`\<`Subgraphs`\>

The subgraphs of the assistant.

***

### getVersions()

> **getVersions**(`assistantId`, `payload`?): `Promise`\<`AssistantVersion`[]\>

Defined in: [client.ts:590](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L590)

List all versions of an assistant.

#### Parameters

##### assistantId

`string`

ID of the assistant.

##### payload?

###### limit?

`number`

###### metadata?

`Metadata`

###### offset?

`number`

#### Returns

`Promise`\<`AssistantVersion`[]\>

List of assistant versions.

***

### search()

> **search**(`query`?): `Promise`\<`Assistant`[]\>

Defined in: [client.ts:563](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L563)

List assistants.

#### Parameters

##### query?

Query options.

###### graphId?

`string`

###### limit?

`number`

###### metadata?

`Metadata`

###### offset?

`number`

###### sortBy?

`AssistantSortBy`

###### sortOrder?

`SortOrder`

#### Returns

`Promise`\<`Assistant`[]\>

List of assistants.

***

### setLatest()

> **setLatest**(`assistantId`, `version`): `Promise`\<`Assistant`\>

Defined in: [client.ts:618](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L618)

Change the version of an assistant.

#### Parameters

##### assistantId

`string`

ID of the assistant.

##### version

`number`

The version to change to.

#### Returns

`Promise`\<`Assistant`\>

The updated assistant.

***

### update()

> **update**(`assistantId`, `payload`): `Promise`\<`Assistant`\>

Defined in: [client.ts:525](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L525)

Update an assistant.

#### Parameters

##### assistantId

`string`

ID of the assistant.

##### payload

Payload for updating the assistant.

###### config?

`Config`

###### description?

`string`

###### graphId?

`string`

###### metadata?

`Metadata`

###### name?

`string`

#### Returns

`Promise`\<`Assistant`\>

The updated assistant.
