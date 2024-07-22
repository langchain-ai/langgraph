
<a name="readmemd"></a>

**[@langchain/langgraph-sdk](https://github.com/langchain-ai/langgraph/tree/main/libs/sdk-js)** • **Docs**

***

## [@langchain/langgraph-sdk](https://github.com/langchain-ai/langgraph/tree/main/libs/sdk-js)

### Modules

- [client](#clientreadmemd)
- [schema](#schemareadmemd)
- [types](#typesreadmemd)


<a name="clientreadmemd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / client

## client

### Index

#### Classes

- [AssistantsClient](#clientclassesassistantsclientmd)
- [BaseClient](#clientclassesbaseclientmd)
- [Client](#clientclassesclientmd)
- [RunsClient](#clientclassesrunsclientmd)
- [ThreadsClient](#clientclassesthreadsclientmd)

#### Interfaces

- [ClientConfig](#clientinterfacesclientconfigmd)


<a name="clientclassesassistantsclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [client](#clientreadmemd) / AssistantsClient

## Class: AssistantsClient

### Extends

- [`BaseClient`](#clientclassesbaseclientmd)

### Constructors

#### new AssistantsClient()

> **new AssistantsClient**(`config`?): [`AssistantsClient`](#clientclassesassistantsclientmd)

##### Parameters

• **config?**: [`ClientConfig`](#clientinterfacesclientconfigmd)

##### Returns

[`AssistantsClient`](#clientclassesassistantsclientmd)

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`constructor`](#constructors)

##### Defined in

[client.mts:38](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L38)

### Properties

#### apiUrl

> `protected` **apiUrl**: `string`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`apiUrl`](#apiurl)

##### Defined in

[client.mts:34](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L34)

***

#### asyncCaller

> `protected` **asyncCaller**: `AsyncCaller`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`asyncCaller`](#asynccaller)

##### Defined in

[client.mts:30](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L30)

***

#### defaultHeaders

> `protected` **defaultHeaders**: `Record`\<`string`, `undefined` \| `null` \| `string`\>

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`defaultHeaders`](#defaultheaders)

##### Defined in

[client.mts:36](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L36)

***

#### timeoutMs

> `protected` **timeoutMs**: `number`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`timeoutMs`](#timeoutms)

##### Defined in

[client.mts:32](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L32)

### Methods

#### create()

> **create**(`payload`): `Promise`\<[`Assistant`](#schemainterfacesassistantmd)\>

Create a new assistant.

##### Parameters

• **payload**

Payload for creating an assistant.

• **payload.config?**: [`Config`](#schemainterfacesconfigmd)

• **payload.graphId**: `string`

• **payload.metadata?**: [`Metadata`](#schematype-aliasesmetadatamd)

##### Returns

`Promise`\<[`Assistant`](#schemainterfacesassistantmd)\>

The created assistant.

##### Defined in

[client.mts:141](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L141)

***

#### delete()

> **delete**(`assistantId`): `Promise`\<`void`\>

Delete an assistant.

##### Parameters

• **assistantId**: `string`

ID of the assistant.

##### Returns

`Promise`\<`void`\>

##### Defined in

[client.mts:185](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L185)

***

#### fetch()

> `protected` **fetch**\<`T`\>(`path`, `options`?): `Promise`\<`T`\>

##### Type Parameters

• **T**

##### Parameters

• **path**: `string`

• **options?**: `RequestInit` & `object`

##### Returns

`Promise`\<`T`\>

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`fetch`](#fetch)

##### Defined in

[client.mts:90](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L90)

***

#### get()

> **get**(`assistantId`): `Promise`\<[`Assistant`](#schemainterfacesassistantmd)\>

Get an assistant by ID.

##### Parameters

• **assistantId**: `string`

The ID of the assistant.

##### Returns

`Promise`\<[`Assistant`](#schemainterfacesassistantmd)\>

Assistant

##### Defined in

[client.mts:114](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L114)

***

#### getGraph()

> **getGraph**(`assistantId`): `Promise`\<[`AssistantGraph`](#schematype-aliasesassistantgraphmd)\>

Get the JSON representation of the graph assigned to a runnable

##### Parameters

• **assistantId**: `string`

The ID of the assistant.

##### Returns

`Promise`\<[`AssistantGraph`](#schematype-aliasesassistantgraphmd)\>

Serialized graph

##### Defined in

[client.mts:123](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L123)

***

#### getSchemas()

> **getSchemas**(`assistantId`): `Promise`\<[`GraphSchema`](#schemainterfacesgraphschemamd)\>

Get the state and config schema of the graph assigned to a runnable

##### Parameters

• **assistantId**: `string`

The ID of the assistant.

##### Returns

`Promise`\<[`GraphSchema`](#schemainterfacesgraphschemamd)\>

Graph schema

##### Defined in

[client.mts:132](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L132)

***

#### prepareFetchOptions()

> `protected` **prepareFetchOptions**(`path`, `options`?): [`URL`, `RequestInit`]

##### Parameters

• **path**: `string`

• **options?**: `RequestInit` & `object`

##### Returns

[`URL`, `RequestInit`]

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`prepareFetchOptions`](#preparefetchoptions)

##### Defined in

[client.mts:50](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L50)

***

#### search()

> **search**(`query`?): `Promise`\<[`Assistant`](#schemainterfacesassistantmd)[]\>

List assistants.

##### Parameters

• **query?**

Query options.

• **query.limit?**: `number`

• **query.metadata?**: [`Metadata`](#schematype-aliasesmetadatamd)

• **query.offset?**: `number`

##### Returns

`Promise`\<[`Assistant`](#schemainterfacesassistantmd)[]\>

List of assistants.

##### Defined in

[client.mts:196](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L196)

***

#### update()

> **update**(`assistantId`, `payload`): `Promise`\<[`Assistant`](#schemainterfacesassistantmd)\>

Update an assistant.

##### Parameters

• **assistantId**: `string`

ID of the assistant.

• **payload**

Payload for updating the assistant.

• **payload.config?**: [`Config`](#schemainterfacesconfigmd)

• **payload.graphId**: `string`

• **payload.metadata?**: [`Metadata`](#schematype-aliasesmetadatamd)

##### Returns

`Promise`\<[`Assistant`](#schemainterfacesassistantmd)\>

The updated assistant.

##### Defined in

[client.mts:162](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L162)


<a name="clientclassesbaseclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [client](#clientreadmemd) / BaseClient

## Class: BaseClient

### Extended by

- [`AssistantsClient`](#clientclassesassistantsclientmd)
- [`ThreadsClient`](#clientclassesthreadsclientmd)
- [`RunsClient`](#clientclassesrunsclientmd)

### Constructors

#### new BaseClient()

> **new BaseClient**(`config`?): [`BaseClient`](#clientclassesbaseclientmd)

##### Parameters

• **config?**: [`ClientConfig`](#clientinterfacesclientconfigmd)

##### Returns

[`BaseClient`](#clientclassesbaseclientmd)

##### Defined in

[client.mts:38](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L38)

### Properties

#### apiUrl

> `protected` **apiUrl**: `string`

##### Defined in

[client.mts:34](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L34)

***

#### asyncCaller

> `protected` **asyncCaller**: `AsyncCaller`

##### Defined in

[client.mts:30](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L30)

***

#### defaultHeaders

> `protected` **defaultHeaders**: `Record`\<`string`, `undefined` \| `null` \| `string`\>

##### Defined in

[client.mts:36](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L36)

***

#### timeoutMs

> `protected` **timeoutMs**: `number`

##### Defined in

[client.mts:32](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L32)

### Methods

#### fetch()

> `protected` **fetch**\<`T`\>(`path`, `options`?): `Promise`\<`T`\>

##### Type Parameters

• **T**

##### Parameters

• **path**: `string`

• **options?**: `RequestInit` & `object`

##### Returns

`Promise`\<`T`\>

##### Defined in

[client.mts:90](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L90)

***

#### prepareFetchOptions()

> `protected` **prepareFetchOptions**(`path`, `options`?): [`URL`, `RequestInit`]

##### Parameters

• **path**: `string`

• **options?**: `RequestInit` & `object`

##### Returns

[`URL`, `RequestInit`]

##### Defined in

[client.mts:50](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L50)


<a name="clientclassesclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [client](#clientreadmemd) / Client

## Class: Client

### Constructors

#### new Client()

> **new Client**(`config`?): [`Client`](#clientclassesclientmd)

##### Parameters

• **config?**: [`ClientConfig`](#clientinterfacesclientconfigmd)

##### Returns

[`Client`](#clientclassesclientmd)

##### Defined in

[client.mts:675](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L675)

### Properties

#### assistants

> **assistants**: [`AssistantsClient`](#clientclassesassistantsclientmd)

The client for interacting with assistants.

##### Defined in

[client.mts:663](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L663)

***

#### runs

> **runs**: [`RunsClient`](#clientclassesrunsclientmd)

The client for interacting with runs.

##### Defined in

[client.mts:673](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L673)

***

#### threads

> **threads**: [`ThreadsClient`](#clientclassesthreadsclientmd)

The client for interacting with threads.

##### Defined in

[client.mts:668](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L668)


<a name="clientclassesrunsclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [client](#clientreadmemd) / RunsClient

## Class: RunsClient

### Extends

- [`BaseClient`](#clientclassesbaseclientmd)

### Constructors

#### new RunsClient()

> **new RunsClient**(`config`?): [`RunsClient`](#clientclassesrunsclientmd)

##### Parameters

• **config?**: [`ClientConfig`](#clientinterfacesclientconfigmd)

##### Returns

[`RunsClient`](#clientclassesrunsclientmd)

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`constructor`](#constructors)

##### Defined in

[client.mts:38](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L38)

### Properties

#### apiUrl

> `protected` **apiUrl**: `string`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`apiUrl`](#apiurl)

##### Defined in

[client.mts:34](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L34)

***

#### asyncCaller

> `protected` **asyncCaller**: `AsyncCaller`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`asyncCaller`](#asynccaller)

##### Defined in

[client.mts:30](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L30)

***

#### defaultHeaders

> `protected` **defaultHeaders**: `Record`\<`string`, `undefined` \| `null` \| `string`\>

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`defaultHeaders`](#defaultheaders)

##### Defined in

[client.mts:36](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L36)

***

#### timeoutMs

> `protected` **timeoutMs**: `number`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`timeoutMs`](#timeoutms)

##### Defined in

[client.mts:32](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L32)

### Methods

#### cancel()

> **cancel**(`threadId`, `runId`, `wait`): `Promise`\<`void`\>

Cancel a run.

##### Parameters

• **threadId**: `string`

The ID of the thread.

• **runId**: `string`

The ID of the run.

• **wait**: `boolean` = `false`

Whether to block when canceling

##### Returns

`Promise`\<`void`\>

##### Defined in

[client.mts:621](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L621)

***

#### create()

> **create**(`threadId`, `assistantId`, `payload`?): `Promise`\<[`Run`](#schemainterfacesrunmd)\>

Create a run.

##### Parameters

• **threadId**: `string`

The ID of the thread.

• **assistantId**: `string`

Assistant ID to use for this run.

• **payload?**: [`RunsCreatePayload`](#typesinterfacesrunscreatepayloadmd)

Payload for creating a run.

##### Returns

`Promise`\<[`Run`](#schemainterfacesrunmd)\>

The created run.

##### Defined in

[client.mts:502](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L502)

***

#### delete()

> **delete**(`threadId`, `runId`): `Promise`\<`void`\>

Delete a run.

##### Parameters

• **threadId**: `string`

The ID of the thread.

• **runId**: `string`

The ID of the run.

##### Returns

`Promise`\<`void`\>

##### Defined in

[client.mts:652](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L652)

***

#### fetch()

> `protected` **fetch**\<`T`\>(`path`, `options`?): `Promise`\<`T`\>

##### Type Parameters

• **T**

##### Parameters

• **path**: `string`

• **options?**: `RequestInit` & `object`

##### Returns

`Promise`\<`T`\>

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`fetch`](#fetch)

##### Defined in

[client.mts:90](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L90)

***

#### get()

> **get**(`threadId`, `runId`): `Promise`\<[`Run`](#schemainterfacesrunmd)\>

Get a run by ID.

##### Parameters

• **threadId**: `string`

The ID of the thread.

• **runId**: `string`

The ID of the run.

##### Returns

`Promise`\<[`Run`](#schemainterfacesrunmd)\>

The run.

##### Defined in

[client.mts:609](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L609)

***

#### join()

> **join**(`threadId`, `runId`): `Promise`\<`void`\>

Block until a run is done.

##### Parameters

• **threadId**: `string`

The ID of the thread.

• **runId**: `string`

The ID of the run.

##### Returns

`Promise`\<`void`\>

##### Defined in

[client.mts:641](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L641)

***

#### list()

> **list**(`threadId`, `options`?): `Promise`\<[`Run`](#schemainterfacesrunmd)[]\>

List all runs for a thread.

##### Parameters

• **threadId**: `string`

The ID of the thread.

• **options?**

Filtering and pagination options.

• **options.limit?**: `number`

Maximum number of runs to return.
Defaults to 10

• **options.offset?**: `number`

Offset to start from.
Defaults to 0.

##### Returns

`Promise`\<[`Run`](#schemainterfacesrunmd)[]\>

List of runs.

##### Defined in

[client.mts:578](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L578)

***

#### prepareFetchOptions()

> `protected` **prepareFetchOptions**(`path`, `options`?): [`URL`, `RequestInit`]

##### Parameters

• **path**: `string`

• **options?**: `RequestInit` & `object`

##### Returns

[`URL`, `RequestInit`]

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`prepareFetchOptions`](#preparefetchoptions)

##### Defined in

[client.mts:50](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L50)

***

#### stream()

Create a run and stream the results.

##### Param

The ID of the thread.

##### Param

Assistant ID to use for this run.

##### Param

Payload for creating a run.

##### stream(threadId, assistantId, payload)

> **stream**(`threadId`, `assistantId`, `payload`?): `AsyncGenerator`\<`object`, `any`, `unknown`\>

Create a run and stream the results.

###### Parameters

• **threadId**: `null`

• **assistantId**: `string`

• **payload?**: `Omit`\<[`RunsStreamPayload`](#typesinterfacesrunsstreampayloadmd), `"multitaskStrategy"`\>

###### Returns

`AsyncGenerator`\<`object`, `any`, `unknown`\>

####### data

> **data**: `any`

####### event

> **event**: [`StreamEvent`](#typestype-aliasesstreameventmd)

###### Defined in

[client.mts:401](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L401)

##### stream(threadId, assistantId, payload)

> **stream**(`threadId`, `assistantId`, `payload`?): `AsyncGenerator`\<`object`, `any`, `unknown`\>

Create a run and stream the results.

###### Parameters

• **threadId**: `string`

• **assistantId**: `string`

• **payload?**: [`RunsStreamPayload`](#typesinterfacesrunsstreampayloadmd)

###### Returns

`AsyncGenerator`\<`object`, `any`, `unknown`\>

####### data

> **data**: `any`

####### event

> **event**: [`StreamEvent`](#typestype-aliasesstreameventmd)

###### Defined in

[client.mts:410](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L410)

***

#### wait()

Create a run and wait for it to complete.

##### Param

The ID of the thread.

##### Param

Assistant ID to use for this run.

##### Param

Payload for creating a run.

##### wait(threadId, assistantId, payload)

> **wait**(`threadId`, `assistantId`, `payload`?): `Promise`\<[`DefaultValues`](#schematype-aliasesdefaultvaluesmd)\>

Create a run and wait for it to complete.

###### Parameters

• **threadId**: `null`

• **assistantId**: `string`

• **payload?**: `Omit`\<[`RunsStreamPayload`](#typesinterfacesrunsstreampayloadmd), `"multitaskStrategy"`\>

###### Returns

`Promise`\<[`DefaultValues`](#schematype-aliasesdefaultvaluesmd)\>

###### Defined in

[client.mts:526](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L526)

##### wait(threadId, assistantId, payload)

> **wait**(`threadId`, `assistantId`, `payload`?): `Promise`\<[`DefaultValues`](#schematype-aliasesdefaultvaluesmd)\>

Create a run and wait for it to complete.

###### Parameters

• **threadId**: `string`

• **assistantId**: `string`

• **payload?**: [`RunsStreamPayload`](#typesinterfacesrunsstreampayloadmd)

###### Returns

`Promise`\<[`DefaultValues`](#schematype-aliasesdefaultvaluesmd)\>

###### Defined in

[client.mts:532](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L532)


<a name="clientclassesthreadsclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [client](#clientreadmemd) / ThreadsClient

## Class: ThreadsClient

### Extends

- [`BaseClient`](#clientclassesbaseclientmd)

### Constructors

#### new ThreadsClient()

> **new ThreadsClient**(`config`?): [`ThreadsClient`](#clientclassesthreadsclientmd)

##### Parameters

• **config?**: [`ClientConfig`](#clientinterfacesclientconfigmd)

##### Returns

[`ThreadsClient`](#clientclassesthreadsclientmd)

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`constructor`](#constructors)

##### Defined in

[client.mts:38](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L38)

### Properties

#### apiUrl

> `protected` **apiUrl**: `string`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`apiUrl`](#apiurl)

##### Defined in

[client.mts:34](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L34)

***

#### asyncCaller

> `protected` **asyncCaller**: `AsyncCaller`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`asyncCaller`](#asynccaller)

##### Defined in

[client.mts:30](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L30)

***

#### defaultHeaders

> `protected` **defaultHeaders**: `Record`\<`string`, `undefined` \| `null` \| `string`\>

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`defaultHeaders`](#defaultheaders)

##### Defined in

[client.mts:36](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L36)

***

#### timeoutMs

> `protected` **timeoutMs**: `number`

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`timeoutMs`](#timeoutms)

##### Defined in

[client.mts:32](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L32)

### Methods

#### create()

> **create**(`payload`?): `Promise`\<[`Thread`](#schemainterfacesthreadmd)\>

Create a new thread.

##### Parameters

• **payload?**

Payload for creating a thread.

• **payload.metadata?**: [`Metadata`](#schematype-aliasesmetadatamd)

Metadata for the thread.

##### Returns

`Promise`\<[`Thread`](#schemainterfacesthreadmd)\>

The created thread.

##### Defined in

[client.mts:229](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L229)

***

#### delete()

> **delete**(`threadId`): `Promise`\<`void`\>

Delete a thread.

##### Parameters

• **threadId**: `string`

ID of the thread.

##### Returns

`Promise`\<`void`\>

##### Defined in

[client.mts:268](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L268)

***

#### fetch()

> `protected` **fetch**\<`T`\>(`path`, `options`?): `Promise`\<`T`\>

##### Type Parameters

• **T**

##### Parameters

• **path**: `string`

• **options?**: `RequestInit` & `object`

##### Returns

`Promise`\<`T`\>

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`fetch`](#fetch)

##### Defined in

[client.mts:90](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L90)

***

#### get()

> **get**(`threadId`): `Promise`\<[`Thread`](#schemainterfacesthreadmd)\>

Get a thread by ID.

##### Parameters

• **threadId**: `string`

ID of the thread.

##### Returns

`Promise`\<[`Thread`](#schemainterfacesthreadmd)\>

The thread.

##### Defined in

[client.mts:219](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L219)

***

#### getHistory()

> **getHistory**\<`ValuesType`\>(`threadId`, `options`?): `Promise`\<[`ThreadState`](#schemainterfacesthreadstatemd)\<`ValuesType`\>[]\>

Get all past states for a thread.

##### Type Parameters

• **ValuesType** = [`DefaultValues`](#schematype-aliasesdefaultvaluesmd)

##### Parameters

• **threadId**: `string`

ID of the thread.

• **options?**

Additional options.

• **options.before?**: [`Config`](#schemainterfacesconfigmd)

• **options.limit?**: `number`

• **options.metadata?**: [`Metadata`](#schematype-aliasesmetadatamd)

##### Returns

`Promise`\<[`ThreadState`](#schemainterfacesthreadstatemd)\<`ValuesType`\>[]\>

List of thread states.

##### Defined in

[client.mts:378](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L378)

***

#### getState()

> **getState**\<`ValuesType`\>(`threadId`, `checkpointId`?): `Promise`\<[`ThreadState`](#schemainterfacesthreadstatemd)\<`ValuesType`\>\>

Get state for a thread.

##### Type Parameters

• **ValuesType** = [`DefaultValues`](#schematype-aliasesdefaultvaluesmd)

##### Parameters

• **threadId**: `string`

ID of the thread.

• **checkpointId?**: `string`

##### Returns

`Promise`\<[`ThreadState`](#schemainterfacesthreadstatemd)\<`ValuesType`\>\>

Thread state.

##### Defined in

[client.mts:311](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L311)

***

#### patchState()

> **patchState**(`threadIdOrConfig`, `metadata`): `Promise`\<`void`\>

Patch the metadata of a thread.

##### Parameters

• **threadIdOrConfig**: `string` \| [`Config`](#schemainterfacesconfigmd)

Thread ID or config to patch the state of.

• **metadata**: [`Metadata`](#schematype-aliasesmetadatamd)

Metadata to patch the state with.

##### Returns

`Promise`\<`void`\>

##### Defined in

[client.mts:348](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L348)

***

#### prepareFetchOptions()

> `protected` **prepareFetchOptions**(`path`, `options`?): [`URL`, `RequestInit`]

##### Parameters

• **path**: `string`

• **options?**: `RequestInit` & `object`

##### Returns

[`URL`, `RequestInit`]

##### Inherited from

[`BaseClient`](#clientclassesbaseclientmd).[`prepareFetchOptions`](#preparefetchoptions)

##### Defined in

[client.mts:50](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L50)

***

#### search()

> **search**(`query`?): `Promise`\<[`Thread`](#schemainterfacesthreadmd)[]\>

List threads

##### Parameters

• **query?**

Query options

• **query.limit?**: `number`

Maximum number of threads to return.
Defaults to 10

• **query.metadata?**: [`Metadata`](#schematype-aliasesmetadatamd)

Metadata to filter threads by.

• **query.offset?**: `number`

Offset to start from.

##### Returns

`Promise`\<[`Thread`](#schemainterfacesthreadmd)[]\>

List of threads

##### Defined in

[client.mts:280](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L280)

***

#### update()

> **update**(`threadId`, `payload`?): `Promise`\<[`Thread`](#schemainterfacesthreadmd)\>

Update a thread.

##### Parameters

• **threadId**: `string`

ID of the thread.

• **payload?**

Payload for updating the thread.

• **payload.metadata?**: [`Metadata`](#schematype-aliasesmetadatamd)

Metadata for the thread.

##### Returns

`Promise`\<[`Thread`](#schemainterfacesthreadmd)\>

The updated thread.

##### Defined in

[client.mts:248](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L248)

***

#### updateState()

> **updateState**\<`ValuesType`\>(`threadId`, `options`): `Promise`\<`void`\>

Add state to a thread.

##### Type Parameters

• **ValuesType** = [`DefaultValues`](#schematype-aliasesdefaultvaluesmd)

##### Parameters

• **threadId**: `string`

The ID of the thread.

• **options**

• **options.asNode?**: `string`

• **options.checkpointId?**: `string`

• **options.values**: `ValuesType`

##### Returns

`Promise`\<`void`\>

##### Defined in

[client.mts:328](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L328)


<a name="clientinterfacesclientconfigmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [client](#clientreadmemd) / ClientConfig

## Interface: ClientConfig

### Properties

#### apiUrl?

> `optional` **apiUrl**: `string`

##### Defined in

[client.mts:23](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L23)

***

#### callerOptions?

> `optional` **callerOptions**: `AsyncCallerParams`

##### Defined in

[client.mts:24](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L24)

***

#### defaultHeaders?

> `optional` **defaultHeaders**: `Record`\<`string`, `undefined` \| `null` \| `string`\>

##### Defined in

[client.mts:26](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L26)

***

#### timeoutMs?

> `optional` **timeoutMs**: `number`

##### Defined in

[client.mts:25](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/client.mts#L25)


<a name="schemareadmemd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / schema

## schema

### Index

#### Interfaces

- [Assistant](#schemainterfacesassistantmd)
- [Config](#schemainterfacesconfigmd)
- [GraphSchema](#schemainterfacesgraphschemamd)
- [Run](#schemainterfacesrunmd)
- [Thread](#schemainterfacesthreadmd)
- [ThreadState](#schemainterfacesthreadstatemd)

#### Type Aliases

- [AssistantGraph](#schematype-aliasesassistantgraphmd)
- [DefaultValues](#schematype-aliasesdefaultvaluesmd)
- [Metadata](#schematype-aliasesmetadatamd)


<a name="schemainterfacesassistantmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / Assistant

## Interface: Assistant

### Properties

#### assistant\_id

> **assistant\_id**: `string`

##### Defined in

[schema.ts:55](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L55)

***

#### config

> **config**: [`Config`](#schemainterfacesconfigmd)

##### Defined in

[schema.ts:57](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L57)

***

#### created\_at

> **created\_at**: `string`

##### Defined in

[schema.ts:58](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L58)

***

#### graph\_id

> **graph\_id**: `string`

##### Defined in

[schema.ts:56](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L56)

***

#### metadata

> **metadata**: [`Metadata`](#schematype-aliasesmetadatamd)

##### Defined in

[schema.ts:60](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L60)

***

#### updated\_at

> **updated\_at**: `string`

##### Defined in

[schema.ts:59](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L59)


<a name="schemainterfacesconfigmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / Config

## Interface: Config

### Properties

#### configurable

> **configurable**: `object`

Runtime values for attributes previously made configurable on this Runnable.

##### Index Signature

 \[`key`: `string`\]: `unknown`

##### thread\_id?

> `optional` **thread\_id**: `string`

ID of the thread

##### thread\_ts?

> `optional` **thread\_ts**: `string`

Timestamp of the state checkpoint

##### Defined in

[schema.ts:21](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L21)

***

#### recursion\_limit?

> `optional` **recursion\_limit**: `number`

Maximum number of times a call can recurse.
If not provided, defaults to 25.

##### Defined in

[schema.ts:16](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L16)

***

#### tags?

> `optional` **tags**: `string`[]

Tags for this call and any sub-calls (eg. a Chain calling an LLM).
You can use these to filter calls.

##### Defined in

[schema.ts:10](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L10)


<a name="schemainterfacesgraphschemamd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / GraphSchema

## Interface: GraphSchema

### Properties

#### config\_schema

> **config\_schema**: `JSONSchema7`

The schema for the graph config

##### Defined in

[schema.ts:49](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L49)

***

#### graph\_id

> **graph\_id**: `string`

The ID of the graph.

##### Defined in

[schema.ts:39](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L39)

***

#### state\_schema

> **state\_schema**: `JSONSchema7`

The schema for the graph state

##### Defined in

[schema.ts:44](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L44)


<a name="schemainterfacesrunmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / Run

## Interface: Run

### Properties

#### assistant\_id

> **assistant\_id**: `string`

##### Defined in

[schema.ts:85](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L85)

***

#### created\_at

> **created\_at**: `string`

##### Defined in

[schema.ts:86](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L86)

***

#### metadata

> **metadata**: [`Metadata`](#schematype-aliasesmetadatamd)

##### Defined in

[schema.ts:95](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L95)

***

#### run\_id

> **run\_id**: `string`

##### Defined in

[schema.ts:83](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L83)

***

#### status

> **status**: `"pending"` \| `"running"` \| `"error"` \| `"success"` \| `"timeout"` \| `"interrupted"`

##### Defined in

[schema.ts:88](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L88)

***

#### thread\_id

> **thread\_id**: `string`

##### Defined in

[schema.ts:84](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L84)

***

#### updated\_at

> **updated\_at**: `string`

##### Defined in

[schema.ts:87](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L87)


<a name="schemainterfacesthreadmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / Thread

## Interface: Thread

### Properties

#### created\_at

> **created\_at**: `string`

##### Defined in

[schema.ts:66](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L66)

***

#### metadata

> **metadata**: [`Metadata`](#schematype-aliasesmetadatamd)

##### Defined in

[schema.ts:68](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L68)

***

#### thread\_id

> **thread\_id**: `string`

##### Defined in

[schema.ts:65](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L65)

***

#### updated\_at

> **updated\_at**: `string`

##### Defined in

[schema.ts:67](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L67)


<a name="schemainterfacesthreadstatemd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / ThreadState

## Interface: ThreadState\<ValuesType\>

### Type Parameters

• **ValuesType** = [`DefaultValues`](#schematype-aliasesdefaultvaluesmd)

### Properties

#### checkpoint\_id

> **checkpoint\_id**: `string`

##### Defined in

[schema.ts:76](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L76)

***

#### created\_at

> **created\_at**: `Optional`\<`string`\>

##### Defined in

[schema.ts:78](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L78)

***

#### metadata

> **metadata**: [`Metadata`](#schematype-aliasesmetadatamd)

##### Defined in

[schema.ts:77](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L77)

***

#### next

> **next**: `string`[]

##### Defined in

[schema.ts:75](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L75)

***

#### parent\_checkpoint\_id

> **parent\_checkpoint\_id**: `Optional`\<`string`\>

##### Defined in

[schema.ts:79](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L79)

***

#### values

> **values**: `ValuesType`

##### Defined in

[schema.ts:74](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L74)


<a name="schematype-aliasesassistantgraphmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / AssistantGraph

## Type Alias: AssistantGraph

> **AssistantGraph**: `Record`\<`string`, `Record`\<`string`, `unknown`\>[]\>

### Defined in

[schema.ts:62](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L62)


<a name="schematype-aliasesdefaultvaluesmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / DefaultValues

## Type Alias: DefaultValues

> **DefaultValues**: `Record`\<`string`, `unknown`\>[] \| `Record`\<`string`, `unknown`\>

### Defined in

[schema.ts:71](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L71)


<a name="schematype-aliasesmetadatamd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [schema](#schemareadmemd) / Metadata

## Type Alias: Metadata

> **Metadata**: `Optional`\<`Record`\<`string`, `unknown`\>\>

### Defined in

[schema.ts:52](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/schema.ts#L52)


<a name="typesreadmemd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / types

## types

### Index

#### Interfaces

- [RunsCreatePayload](#typesinterfacesrunscreatepayloadmd)
- [RunsStreamPayload](#typesinterfacesrunsstreampayloadmd)

#### Type Aliases

- [MultitaskStrategy](#typestype-aliasesmultitaskstrategymd)
- [RunsWaitPayload](#typestype-aliasesrunswaitpayloadmd)
- [StreamEvent](#typestype-aliasesstreameventmd)
- [StreamMode](#typestype-aliasesstreammodemd)


<a name="typesinterfacesrunscreatepayloadmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [types](#typesreadmemd) / RunsCreatePayload

## Interface: RunsCreatePayload

### Extends

- `RunsInvokePayload`

### Properties

#### config?

> `optional` **config**: [`Config`](#schemainterfacesconfigmd)

Additional configuration for the run.

##### Inherited from

`RunsInvokePayload.config`

##### Defined in

[types.mts:30](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L30)

***

#### input?

> `optional` **input**: `null` \| `Record`\<`string`, `unknown`\>

Input to the run. Pass `null` to resume from the current state of the thread.

##### Inherited from

`RunsInvokePayload.input`

##### Defined in

[types.mts:20](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L20)

***

#### interruptAfter?

> `optional` **interruptAfter**: `string`[]

Interrupt execution after leaving these nodes.

##### Inherited from

`RunsInvokePayload.interruptAfter`

##### Defined in

[types.mts:40](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L40)

***

#### interruptBefore?

> `optional` **interruptBefore**: `string`[]

Interrupt execution before entering these nodes.

##### Inherited from

`RunsInvokePayload.interruptBefore`

##### Defined in

[types.mts:35](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L35)

***

#### metadata?

> `optional` **metadata**: [`Metadata`](#schematype-aliasesmetadatamd)

Metadata for the run.

##### Inherited from

`RunsInvokePayload.metadata`

##### Defined in

[types.mts:25](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L25)

***

#### multitaskStrategy?

> `optional` **multitaskStrategy**: [`MultitaskStrategy`](#typestype-aliasesmultitaskstrategymd)

Strategy to handle concurrent runs on the same thread. Only relevant if
there is a pending/inflight run on the same thread. One of:
- "reject": Reject the new run.
- "interrupt": Interrupt the current run, keeping steps completed until now,
    and start a new one.
- "rollback": Cancel and delete the existing run, rolling back the thread to
   the state before it had started, then start the new run.
- "enqueue": Queue up the new run to start after the current run finishes.

##### Inherited from

`RunsInvokePayload.multitaskStrategy`

##### Defined in

[types.mts:52](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L52)

***

#### signal?

> `optional` **signal**: `AbortSignal`

Abort controller signal to cancel the run.

##### Inherited from

`RunsInvokePayload.signal`

##### Defined in

[types.mts:57](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L57)

***

#### webhook?

> `optional` **webhook**: `string`

Webhook to call when the run is complete.

##### Defined in

[types.mts:83](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L83)


<a name="typesinterfacesrunsstreampayloadmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [types](#typesreadmemd) / RunsStreamPayload

## Interface: RunsStreamPayload

### Extends

- `RunsInvokePayload`

### Properties

#### config?

> `optional` **config**: [`Config`](#schemainterfacesconfigmd)

Additional configuration for the run.

##### Inherited from

`RunsInvokePayload.config`

##### Defined in

[types.mts:30](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L30)

***

#### feedbackKeys?

> `optional` **feedbackKeys**: `string`[]

Pass one or more feedbackKeys if you want to request short-lived signed URLs
for submitting feedback to LangSmith with this key for this run.

##### Defined in

[types.mts:76](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L76)

***

#### input?

> `optional` **input**: `null` \| `Record`\<`string`, `unknown`\>

Input to the run. Pass `null` to resume from the current state of the thread.

##### Inherited from

`RunsInvokePayload.input`

##### Defined in

[types.mts:20](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L20)

***

#### interruptAfter?

> `optional` **interruptAfter**: `string`[]

Interrupt execution after leaving these nodes.

##### Inherited from

`RunsInvokePayload.interruptAfter`

##### Defined in

[types.mts:40](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L40)

***

#### interruptBefore?

> `optional` **interruptBefore**: `string`[]

Interrupt execution before entering these nodes.

##### Inherited from

`RunsInvokePayload.interruptBefore`

##### Defined in

[types.mts:35](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L35)

***

#### metadata?

> `optional` **metadata**: [`Metadata`](#schematype-aliasesmetadatamd)

Metadata for the run.

##### Inherited from

`RunsInvokePayload.metadata`

##### Defined in

[types.mts:25](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L25)

***

#### multitaskStrategy?

> `optional` **multitaskStrategy**: [`MultitaskStrategy`](#typestype-aliasesmultitaskstrategymd)

Strategy to handle concurrent runs on the same thread. Only relevant if
there is a pending/inflight run on the same thread. One of:
- "reject": Reject the new run.
- "interrupt": Interrupt the current run, keeping steps completed until now,
    and start a new one.
- "rollback": Cancel and delete the existing run, rolling back the thread to
   the state before it had started, then start the new run.
- "enqueue": Queue up the new run to start after the current run finishes.

##### Inherited from

`RunsInvokePayload.multitaskStrategy`

##### Defined in

[types.mts:52](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L52)

***

#### signal?

> `optional` **signal**: `AbortSignal`

Abort controller signal to cancel the run.

##### Inherited from

`RunsInvokePayload.signal`

##### Defined in

[types.mts:57](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L57)

***

#### streamMode?

> `optional` **streamMode**: [`StreamMode`](#typestype-aliasesstreammodemd) \| [`StreamMode`](#typestype-aliasesstreammodemd)[]

One of `"values"`, `"messages"`, `"updates"` or `"events"`.
- `"values"`: Stream the thread state any time it changes.
- `"messages"`: Stream chat messages from thread state and calls to chat models,
                token-by-token where possible.
- `"updates"`: Stream the state updates returned by each node.
- `"events"`: Stream all events produced by the run. You can also access these
              afterwards using the `client.runs.listEvents()` method.

##### Defined in

[types.mts:70](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L70)


<a name="typestype-aliasesmultitaskstrategymd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [types](#typesreadmemd) / MultitaskStrategy

## Type Alias: MultitaskStrategy

> **MultitaskStrategy**: `"reject"` \| `"interrupt"` \| `"rollback"` \| `"enqueue"`

### Defined in

[types.mts:4](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L4)


<a name="typestype-aliasesrunswaitpayloadmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [types](#typesreadmemd) / RunsWaitPayload

## Type Alias: RunsWaitPayload

> **RunsWaitPayload**: [`RunsStreamPayload`](#typesinterfacesrunsstreampayloadmd)

### Defined in

[types.mts:86](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L86)


<a name="typestype-aliasesstreameventmd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [types](#typesreadmemd) / StreamEvent

## Type Alias: StreamEvent

> **StreamEvent**: `"events"` \| `"metadata"` \| `"debug"` \| `"updates"` \| `"values"` \| `"messages/partial"` \| `"messages/metadata"` \| `"messages/complete"` \| `string` & `object`

### Defined in

[types.mts:5](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L5)


<a name="typestype-aliasesstreammodemd"></a>

[**@langchain/langgraph-sdk**](#readmemd) • **Docs**

***

[@langchain/langgraph-sdk](#readmemd) / [types](#typesreadmemd) / StreamMode

## Type Alias: StreamMode

> **StreamMode**: `"values"` \| `"messages"` \| `"updates"` \| `"events"` \| `"debug"`

### Defined in

[types.mts:3](https://github.com/langchain-ai/langgraph/blob/d3ec367566bf3d5c58b8150f1d8208e8b21e6144/libs/sdk-js/src/types.mts#L3)
