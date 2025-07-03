
<a name="readmemd"></a>

**[@langchain/langgraph-sdk](https://github.com/langchain-ai/langgraph/tree/main/libs/sdk-js)**

***

## [@langchain/langgraph-sdk](https://github.com/langchain-ai/langgraph/tree/main/libs/sdk-js)

### Classes

- [AssistantsClient](#classesassistantsclientmd)
- [Client](#classesclientmd)
- [CronsClient](#classescronsclientmd)
- [RunsClient](#classesrunsclientmd)
- [StoreClient](#classesstoreclientmd)
- [ThreadsClient](#classesthreadsclientmd)

### Interfaces

- [ClientConfig](#interfacesclientconfigmd)

### Functions

- [getApiKey](#functionsgetapikeymd)


<a name="authreadmemd"></a>

**@langchain/langgraph-sdk**

***

## @langchain/langgraph-sdk/auth

### Classes

- [Auth](#authclassesauthmd)
- [HTTPException](#authclasseshttpexceptionmd)

### Interfaces

- [AuthEventValueMap](#authinterfacesautheventvaluemapmd)

### Type Aliases

- [AuthFilters](#authtype-aliasesauthfiltersmd)


<a name="authclassesauthmd"></a>

[**@langchain/langgraph-sdk**](#authreadmemd)

***

[@langchain/langgraph-sdk](#authreadmemd) / Auth

## Class: Auth\<TExtra, TAuthReturn, TUser\>

Defined in: [src/auth/index.ts:11](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/index.ts#L11)

### Type Parameters

• **TExtra** = \{\}

• **TAuthReturn** *extends* `BaseAuthReturn` = `BaseAuthReturn`

• **TUser** *extends* `BaseUser` = `ToUserLike`\<`TAuthReturn`\>

### Constructors

#### new Auth()

> **new Auth**\<`TExtra`, `TAuthReturn`, `TUser`\>(): [`Auth`](#authclassesauthmd)\<`TExtra`, `TAuthReturn`, `TUser`\>

##### Returns

[`Auth`](#authclassesauthmd)\<`TExtra`, `TAuthReturn`, `TUser`\>

### Methods

#### authenticate()

> **authenticate**\<`T`\>(`cb`): [`Auth`](#authclassesauthmd)\<`TExtra`, `T`\>

Defined in: [src/auth/index.ts:25](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/index.ts#L25)

##### Type Parameters

• **T** *extends* `BaseAuthReturn`

##### Parameters

###### cb

`AuthenticateCallback`\<`T`\>

##### Returns

[`Auth`](#authclassesauthmd)\<`TExtra`, `T`\>

***

#### on()

> **on**\<`T`\>(`event`, `callback`): `this`

Defined in: [src/auth/index.ts:32](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/index.ts#L32)

##### Type Parameters

• **T** *extends* `CallbackEvent`

##### Parameters

###### event

`T`

###### callback

`OnCallback`\<`T`, `TUser`\>

##### Returns

`this`


<a name="authclasseshttpexceptionmd"></a>

[**@langchain/langgraph-sdk**](#authreadmemd)

***

[@langchain/langgraph-sdk](#authreadmemd) / HTTPException

## Class: HTTPException

Defined in: [src/auth/error.ts:66](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/error.ts#L66)

### Extends

- `Error`

### Constructors

#### new HTTPException()

> **new HTTPException**(`status`, `options`?): [`HTTPException`](#authclasseshttpexceptionmd)

Defined in: [src/auth/error.ts:70](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/error.ts#L70)

##### Parameters

###### status

`number`

###### options?

####### cause?

`unknown`

####### headers?

`HeadersInit`

####### message?

`string`

##### Returns

[`HTTPException`](#authclasseshttpexceptionmd)

##### Overrides

`Error.constructor`

### Properties

#### cause?

> `optional` **cause**: `unknown`

Defined in: node\_modules/typescript/lib/lib.es2022.error.d.ts:24

##### Inherited from

`Error.cause`

***

#### headers

> **headers**: `HeadersInit`

Defined in: [src/auth/error.ts:68](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/error.ts#L68)

***

#### message

> **message**: `string`

Defined in: node\_modules/typescript/lib/lib.es5.d.ts:1077

##### Inherited from

`Error.message`

***

#### name

> **name**: `string`

Defined in: node\_modules/typescript/lib/lib.es5.d.ts:1076

##### Inherited from

`Error.name`

***

#### stack?

> `optional` **stack**: `string`

Defined in: node\_modules/typescript/lib/lib.es5.d.ts:1078

##### Inherited from

`Error.stack`

***

#### status

> **status**: `number`

Defined in: [src/auth/error.ts:67](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/error.ts#L67)

***

#### prepareStackTrace()?

> `static` `optional` **prepareStackTrace**: (`err`, `stackTraces`) => `any`

Defined in: node\_modules/@types/node/globals.d.ts:28

Optional override for formatting stack traces

##### Parameters

###### err

`Error`

###### stackTraces

`CallSite`[]

##### Returns

`any`

##### See

https://v8.dev/docs/stack-trace-api#customizing-stack-traces

##### Inherited from

`Error.prepareStackTrace`

***

#### stackTraceLimit

> `static` **stackTraceLimit**: `number`

Defined in: node\_modules/@types/node/globals.d.ts:30

##### Inherited from

`Error.stackTraceLimit`

### Methods

#### captureStackTrace()

> `static` **captureStackTrace**(`targetObject`, `constructorOpt`?): `void`

Defined in: node\_modules/@types/node/globals.d.ts:21

Create .stack property on a target object

##### Parameters

###### targetObject

`object`

###### constructorOpt?

`Function`

##### Returns

`void`

##### Inherited from

`Error.captureStackTrace`


<a name="authinterfacesautheventvaluemapmd"></a>

[**@langchain/langgraph-sdk**](#authreadmemd)

***

[@langchain/langgraph-sdk](#authreadmemd) / AuthEventValueMap

## Interface: AuthEventValueMap

Defined in: [src/auth/types.ts:218](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L218)

### Properties

#### assistants:create

> **assistants:create**: `object`

Defined in: [src/auth/types.ts:226](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L226)

##### assistant\_id?

> `optional` **assistant\_id**: `Maybe`\<`string`\>

##### config?

> `optional` **config**: `Maybe`\<`AssistantConfig`\>

##### graph\_id

> **graph\_id**: `string`

##### if\_exists?

> `optional` **if\_exists**: `Maybe`\<`"raise"` \| `"do_nothing"`\>

##### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### name?

> `optional` **name**: `Maybe`\<`string`\>

***

#### assistants:delete

> **assistants:delete**: `object`

Defined in: [src/auth/types.ts:229](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L229)

##### assistant\_id

> **assistant\_id**: `string`

***

#### assistants:read

> **assistants:read**: `object`

Defined in: [src/auth/types.ts:227](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L227)

##### assistant\_id

> **assistant\_id**: `string`

##### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

***

#### assistants:search

> **assistants:search**: `object`

Defined in: [src/auth/types.ts:230](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L230)

##### graph\_id?

> `optional` **graph\_id**: `Maybe`\<`string`\>

##### limit?

> `optional` **limit**: `Maybe`\<`number`\>

##### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### offset?

> `optional` **offset**: `Maybe`\<`number`\>

***

#### assistants:update

> **assistants:update**: `object`

Defined in: [src/auth/types.ts:228](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L228)

##### assistant\_id

> **assistant\_id**: `string`

##### config?

> `optional` **config**: `Maybe`\<`AssistantConfig`\>

##### graph\_id?

> `optional` **graph\_id**: `Maybe`\<`string`\>

##### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### name?

> `optional` **name**: `Maybe`\<`string`\>

##### version?

> `optional` **version**: `Maybe`\<`number`\>

***

#### crons:create

> **crons:create**: `object`

Defined in: [src/auth/types.ts:232](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L232)

##### cron\_id?

> `optional` **cron\_id**: `Maybe`\<`string`\>

##### end\_time?

> `optional` **end\_time**: `Maybe`\<`string`\>

##### payload?

> `optional` **payload**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### schedule

> **schedule**: `string`

##### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

##### user\_id?

> `optional` **user\_id**: `Maybe`\<`string`\>

***

#### crons:delete

> **crons:delete**: `object`

Defined in: [src/auth/types.ts:235](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L235)

##### cron\_id

> **cron\_id**: `string`

***

#### crons:read

> **crons:read**: `object`

Defined in: [src/auth/types.ts:233](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L233)

##### cron\_id

> **cron\_id**: `string`

***

#### crons:search

> **crons:search**: `object`

Defined in: [src/auth/types.ts:236](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L236)

##### assistant\_id?

> `optional` **assistant\_id**: `Maybe`\<`string`\>

##### limit?

> `optional` **limit**: `Maybe`\<`number`\>

##### offset?

> `optional` **offset**: `Maybe`\<`number`\>

##### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

#### crons:update

> **crons:update**: `object`

Defined in: [src/auth/types.ts:234](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L234)

##### cron\_id

> **cron\_id**: `string`

##### payload?

> `optional` **payload**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### schedule?

> `optional` **schedule**: `Maybe`\<`string`\>

***

#### store:delete

> **store:delete**: `object`

Defined in: [src/auth/types.ts:242](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L242)

##### key

> **key**: `string`

##### namespace?

> `optional` **namespace**: `Maybe`\<`string`[]\>

***

#### store:get

> **store:get**: `object`

Defined in: [src/auth/types.ts:239](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L239)

##### key

> **key**: `string`

##### namespace

> **namespace**: `Maybe`\<`string`[]\>

***

#### store:list\_namespaces

> **store:list\_namespaces**: `object`

Defined in: [src/auth/types.ts:241](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L241)

##### limit?

> `optional` **limit**: `Maybe`\<`number`\>

##### max\_depth?

> `optional` **max\_depth**: `Maybe`\<`number`\>

##### namespace?

> `optional` **namespace**: `Maybe`\<`string`[]\>

##### offset?

> `optional` **offset**: `Maybe`\<`number`\>

##### suffix?

> `optional` **suffix**: `Maybe`\<`string`[]\>

***

#### store:put

> **store:put**: `object`

Defined in: [src/auth/types.ts:238](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L238)

##### key

> **key**: `string`

##### namespace

> **namespace**: `string`[]

##### value

> **value**: `Record`\<`string`, `unknown`\>

***

#### store:search

> **store:search**: `object`

Defined in: [src/auth/types.ts:240](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L240)

##### filter?

> `optional` **filter**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### limit?

> `optional` **limit**: `Maybe`\<`number`\>

##### namespace?

> `optional` **namespace**: `Maybe`\<`string`[]\>

##### offset?

> `optional` **offset**: `Maybe`\<`number`\>

##### query?

> `optional` **query**: `Maybe`\<`string`\>

***

#### threads:create

> **threads:create**: `object`

Defined in: [src/auth/types.ts:219](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L219)

##### if\_exists?

> `optional` **if\_exists**: `Maybe`\<`"raise"` \| `"do_nothing"`\>

##### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

#### threads:create\_run

> **threads:create\_run**: `object`

Defined in: [src/auth/types.ts:224](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L224)

##### after\_seconds?

> `optional` **after\_seconds**: `Maybe`\<`number`\>

##### assistant\_id

> **assistant\_id**: `string`

##### if\_not\_exists?

> `optional` **if\_not\_exists**: `Maybe`\<`"reject"` \| `"create"`\>

##### kwargs

> **kwargs**: `Record`\<`string`, `unknown`\>

##### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### multitask\_strategy?

> `optional` **multitask\_strategy**: `Maybe`\<`"reject"` \| `"interrupt"` \| `"rollback"` \| `"enqueue"`\>

##### prevent\_insert\_if\_inflight?

> `optional` **prevent\_insert\_if\_inflight**: `Maybe`\<`boolean`\>

##### run\_id

> **run\_id**: `string`

##### status

> **status**: `Maybe`\<`"pending"` \| `"running"` \| `"error"` \| `"success"` \| `"timeout"` \| `"interrupted"`\>

##### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

#### threads:delete

> **threads:delete**: `object`

Defined in: [src/auth/types.ts:222](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L222)

##### run\_id?

> `optional` **run\_id**: `Maybe`\<`string`\>

##### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

#### threads:read

> **threads:read**: `object`

Defined in: [src/auth/types.ts:220](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L220)

##### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

#### threads:search

> **threads:search**: `object`

Defined in: [src/auth/types.ts:223](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L223)

##### limit?

> `optional` **limit**: `Maybe`\<`number`\>

##### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### offset?

> `optional` **offset**: `Maybe`\<`number`\>

##### status?

> `optional` **status**: `Maybe`\<`"error"` \| `"interrupted"` \| `"idle"` \| `"busy"` \| `string` & `object`\>

##### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

##### values?

> `optional` **values**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

***

#### threads:update

> **threads:update**: `object`

Defined in: [src/auth/types.ts:221](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L221)

##### action?

> `optional` **action**: `Maybe`\<`"interrupt"` \| `"rollback"`\>

##### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

##### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>


<a name="authtype-aliasesauthfiltersmd"></a>

[**@langchain/langgraph-sdk**](#authreadmemd)

***

[@langchain/langgraph-sdk](#authreadmemd) / AuthFilters

## Type Alias: AuthFilters\<TKey\>

> **AuthFilters**\<`TKey`\>: \{ \[key in TKey\]: string \| \{ \[op in "$contains" \| "$eq"\]?: string \} \}

Defined in: [src/auth/types.ts:367](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/auth/types.ts#L367)

### Type Parameters

• **TKey** *extends* `string` \| `number` \| `symbol`


<a name="classesassistantsclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd)

***

[@langchain/langgraph-sdk](#readmemd) / AssistantsClient

## Class: AssistantsClient

Defined in: [client.ts:294](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L294)

### Extends

- `BaseClient`

### Constructors

#### new AssistantsClient()

> **new AssistantsClient**(`config`?): [`AssistantsClient`](#classesassistantsclientmd)

Defined in: [client.ts:88](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L88)

##### Parameters

###### config?

[`ClientConfig`](#interfacesclientconfigmd)

##### Returns

[`AssistantsClient`](#classesassistantsclientmd)

##### Inherited from

`BaseClient.constructor`

### Methods

#### create()

> **create**(`payload`): `Promise`\<`Assistant`\>

Defined in: [client.ts:359](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L359)

Create a new assistant.

##### Parameters

###### payload

Payload for creating an assistant.

####### assistantId?

`string`

####### config?

`Config`

####### description?

`string`

####### graphId

`string`

####### ifExists?

`OnConflictBehavior`

####### metadata?

`Metadata`

####### name?

`string`

##### Returns

`Promise`\<`Assistant`\>

The created assistant.

***

#### delete()

> **delete**(`assistantId`): `Promise`\<`void`\>

Defined in: [client.ts:415](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L415)

Delete an assistant.

##### Parameters

###### assistantId

`string`

ID of the assistant.

##### Returns

`Promise`\<`void`\>

***

#### get()

> **get**(`assistantId`): `Promise`\<`Assistant`\>

Defined in: [client.ts:301](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L301)

Get an assistant by ID.

##### Parameters

###### assistantId

`string`

The ID of the assistant.

##### Returns

`Promise`\<`Assistant`\>

Assistant

***

#### getGraph()

> **getGraph**(`assistantId`, `options`?): `Promise`\<`AssistantGraph`\>

Defined in: [client.ts:311](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L311)

Get the JSON representation of the graph assigned to a runnable

##### Parameters

###### assistantId

`string`

The ID of the assistant.

###### options?

####### xray?

`number` \| `boolean`

Whether to include subgraphs in the serialized graph representation. If an integer value is provided, only subgraphs with a depth less than or equal to the value will be included.

##### Returns

`Promise`\<`AssistantGraph`\>

Serialized graph

***

#### getSchemas()

> **getSchemas**(`assistantId`): `Promise`\<`GraphSchema`\>

Defined in: [client.ts:325](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L325)

Get the state and config schema of the graph assigned to a runnable

##### Parameters

###### assistantId

`string`

The ID of the assistant.

##### Returns

`Promise`\<`GraphSchema`\>

Graph schema

***

#### getSubgraphs()

> **getSubgraphs**(`assistantId`, `options`?): `Promise`\<`Subgraphs`\>

Defined in: [client.ts:336](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L336)

Get the schemas of an assistant by ID.

##### Parameters

###### assistantId

`string`

The ID of the assistant to get the schema of.

###### options?

Additional options for getting subgraphs, such as namespace or recursion extraction.

####### namespace?

`string`

####### recurse?

`boolean`

##### Returns

`Promise`\<`Subgraphs`\>

The subgraphs of the assistant.

***

#### getVersions()

> **getVersions**(`assistantId`, `payload`?): `Promise`\<`AssistantVersion`[]\>

Defined in: [client.ts:453](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L453)

List all versions of an assistant.

##### Parameters

###### assistantId

`string`

ID of the assistant.

###### payload?

####### limit?

`number`

####### metadata?

`Metadata`

####### offset?

`number`

##### Returns

`Promise`\<`AssistantVersion`[]\>

List of assistant versions.

***

#### search()

> **search**(`query`?): `Promise`\<`Assistant`[]\>

Defined in: [client.ts:426](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L426)

List assistants.

##### Parameters

###### query?

Query options.

####### graphId?

`string`

####### limit?

`number`

####### metadata?

`Metadata`

####### offset?

`number`

####### sortBy?

`AssistantSortBy`

####### sortOrder?

`SortOrder`

##### Returns

`Promise`\<`Assistant`[]\>

List of assistants.

***

#### setLatest()

> **setLatest**(`assistantId`, `version`): `Promise`\<`Assistant`\>

Defined in: [client.ts:481](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L481)

Change the version of an assistant.

##### Parameters

###### assistantId

`string`

ID of the assistant.

###### version

`number`

The version to change to.

##### Returns

`Promise`\<`Assistant`\>

The updated assistant.

***

#### update()

> **update**(`assistantId`, `payload`): `Promise`\<`Assistant`\>

Defined in: [client.ts:388](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L388)

Update an assistant.

##### Parameters

###### assistantId

`string`

ID of the assistant.

###### payload

Payload for updating the assistant.

####### config?

`Config`

####### description?

`string`

####### graphId?

`string`

####### metadata?

`Metadata`

####### name?

`string`

##### Returns

`Promise`\<`Assistant`\>

The updated assistant.


<a name="classesclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd)

***

[@langchain/langgraph-sdk](#readmemd) / Client

## Class: Client\<TStateType, TUpdateType, TCustomEventType\>

Defined in: [client.ts:1448](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1448)

### Type Parameters

• **TStateType** = `DefaultValues`

• **TUpdateType** = `TStateType`

• **TCustomEventType** = `unknown`

### Constructors

#### new Client()

> **new Client**\<`TStateType`, `TUpdateType`, `TCustomEventType`\>(`config`?): [`Client`](#classesclientmd)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:1484](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1484)

##### Parameters

###### config?

[`ClientConfig`](#interfacesclientconfigmd)

##### Returns

[`Client`](#classesclientmd)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

### Properties

#### ~ui

> **~ui**: `UiClient`

Defined in: [client.ts:1482](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1482)

**`Internal`**

The client for interacting with the UI.
 Used by LoadExternalComponent and the API might change in the future.

***

#### assistants

> **assistants**: [`AssistantsClient`](#classesassistantsclientmd)

Defined in: [client.ts:1456](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1456)

The client for interacting with assistants.

***

#### crons

> **crons**: [`CronsClient`](#classescronsclientmd)

Defined in: [client.ts:1471](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1471)

The client for interacting with cron runs.

***

#### runs

> **runs**: [`RunsClient`](#classesrunsclientmd)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:1466](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1466)

The client for interacting with runs.

***

#### store

> **store**: [`StoreClient`](#classesstoreclientmd)

Defined in: [client.ts:1476](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1476)

The client for interacting with the KV store.

***

#### threads

> **threads**: [`ThreadsClient`](#classesthreadsclientmd)\<`TStateType`, `TUpdateType`\>

Defined in: [client.ts:1461](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1461)

The client for interacting with threads.


<a name="classescronsclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd)

***

[@langchain/langgraph-sdk](#readmemd) / CronsClient

## Class: CronsClient

Defined in: [client.ts:197](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L197)

### Extends

- `BaseClient`

### Constructors

#### new CronsClient()

> **new CronsClient**(`config`?): [`CronsClient`](#classescronsclientmd)

Defined in: [client.ts:88](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L88)

##### Parameters

###### config?

[`ClientConfig`](#interfacesclientconfigmd)

##### Returns

[`CronsClient`](#classescronsclientmd)

##### Inherited from

`BaseClient.constructor`

### Methods

#### create()

> **create**(`assistantId`, `payload`?): `Promise`\<`CronCreateResponse`\>

Defined in: [client.ts:238](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L238)

##### Parameters

###### assistantId

`string`

Assistant ID to use for this cron job.

###### payload?

`CronsCreatePayload`

Payload for creating a cron job.

##### Returns

`Promise`\<`CronCreateResponse`\>

***

#### createForThread()

> **createForThread**(`threadId`, `assistantId`, `payload`?): `Promise`\<`CronCreateForThreadResponse`\>

Defined in: [client.ts:205](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L205)

##### Parameters

###### threadId

`string`

The ID of the thread.

###### assistantId

`string`

Assistant ID to use for this cron job.

###### payload?

`CronsCreatePayload`

Payload for creating a cron job.

##### Returns

`Promise`\<`CronCreateForThreadResponse`\>

The created background run.

***

#### delete()

> **delete**(`cronId`): `Promise`\<`void`\>

Defined in: [client.ts:265](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L265)

##### Parameters

###### cronId

`string`

Cron ID of Cron job to delete.

##### Returns

`Promise`\<`void`\>

***

#### search()

> **search**(`query`?): `Promise`\<`Cron`[]\>

Defined in: [client.ts:276](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L276)

##### Parameters

###### query?

Query options.

####### assistantId?

`string`

####### limit?

`number`

####### offset?

`number`

####### threadId?

`string`

##### Returns

`Promise`\<`Cron`[]\>

List of crons.


<a name="classesrunsclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd)

***

[@langchain/langgraph-sdk](#readmemd) / RunsClient

## Class: RunsClient\<TStateType, TUpdateType, TCustomEventType\>

Defined in: [client.ts:776](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L776)

### Extends

- `BaseClient`

### Type Parameters

• **TStateType** = `DefaultValues`

• **TUpdateType** = `TStateType`

• **TCustomEventType** = `unknown`

### Constructors

#### new RunsClient()

> **new RunsClient**\<`TStateType`, `TUpdateType`, `TCustomEventType`\>(`config`?): [`RunsClient`](#classesrunsclientmd)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:88](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L88)

##### Parameters

###### config?

[`ClientConfig`](#interfacesclientconfigmd)

##### Returns

[`RunsClient`](#classesrunsclientmd)\<`TStateType`, `TUpdateType`, `TCustomEventType`\>

##### Inherited from

`BaseClient.constructor`

### Methods

#### cancel()

> **cancel**(`threadId`, `runId`, `wait`, `action`): `Promise`\<`void`\>

Defined in: [client.ts:1063](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1063)

Cancel a run.

##### Parameters

###### threadId

`string`

The ID of the thread.

###### runId

`string`

The ID of the run.

###### wait

`boolean` = `false`

Whether to block when canceling

###### action

`CancelAction` = `"interrupt"`

Action to take when cancelling the run. Possible values are `interrupt` or `rollback`. Default is `interrupt`.

##### Returns

`Promise`\<`void`\>

***

#### create()

> **create**(`threadId`, `assistantId`, `payload`?): `Promise`\<`Run`\>

Defined in: [client.ts:885](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L885)

Create a run.

##### Parameters

###### threadId

`string`

The ID of the thread.

###### assistantId

`string`

Assistant ID to use for this run.

###### payload?

`RunsCreatePayload`

Payload for creating a run.

##### Returns

`Promise`\<`Run`\>

The created run.

***

#### createBatch()

> **createBatch**(`payloads`): `Promise`\<`Run`[]\>

Defined in: [client.ts:921](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L921)

Create a batch of stateless background runs.

##### Parameters

###### payloads

`RunsCreatePayload` & `object`[]

An array of payloads for creating runs.

##### Returns

`Promise`\<`Run`[]\>

An array of created runs.

***

#### delete()

> **delete**(`threadId`, `runId`): `Promise`\<`void`\>

Defined in: [client.ts:1157](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1157)

Delete a run.

##### Parameters

###### threadId

`string`

The ID of the thread.

###### runId

`string`

The ID of the run.

##### Returns

`Promise`\<`void`\>

***

#### get()

> **get**(`threadId`, `runId`): `Promise`\<`Run`\>

Defined in: [client.ts:1050](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1050)

Get a run by ID.

##### Parameters

###### threadId

`string`

The ID of the thread.

###### runId

`string`

The ID of the run.

##### Returns

`Promise`\<`Run`\>

The run.

***

#### join()

> **join**(`threadId`, `runId`, `options`?): `Promise`\<`void`\>

Defined in: [client.ts:1085](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1085)

Block until a run is done.

##### Parameters

###### threadId

`string`

The ID of the thread.

###### runId

`string`

The ID of the run.

###### options?

####### signal?

`AbortSignal`

##### Returns

`Promise`\<`void`\>

***

#### joinStream()

> **joinStream**(`threadId`, `runId`, `options`?): `AsyncGenerator`\<\{ `data`: `any`; `event`: `StreamEvent`; \}\>

Defined in: [client.ts:1111](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1111)

Stream output from a run in real-time, until the run is done.
Output is not buffered, so any output produced before this call will
not be received here.

##### Parameters

###### threadId

`string`

The ID of the thread.

###### runId

`string`

The ID of the run.

###### options?

Additional options for controlling the stream behavior:
  - signal: An AbortSignal that can be used to cancel the stream request
  - cancelOnDisconnect: When true, automatically cancels the run if the client disconnects from the stream
  - streamMode: Controls what types of events to receive from the stream (can be a single mode or array of modes)
       Must be a subset of the stream modes passed when creating the run. Background runs default to having the union of all
       stream modes enabled.

`AbortSignal` | \{ `cancelOnDisconnect`: `boolean`; `signal`: `AbortSignal`; `streamMode`: `StreamMode` \| `StreamMode`[]; \}

##### Returns

`AsyncGenerator`\<\{ `data`: `any`; `event`: `StreamEvent`; \}\>

An async generator yielding stream parts.

***

#### list()

> **list**(`threadId`, `options`?): `Promise`\<`Run`[]\>

Defined in: [client.ts:1013](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1013)

List all runs for a thread.

##### Parameters

###### threadId

`string`

The ID of the thread.

###### options?

Filtering and pagination options.

####### limit?

`number`

Maximum number of runs to return.
Defaults to 10

####### offset?

`number`

Offset to start from.
Defaults to 0.

####### status?

`RunStatus`

Status of the run to filter by.

##### Returns

`Promise`\<`Run`[]\>

List of runs.

***

#### stream()

Create a run and stream the results.

##### Param

The ID of the thread.

##### Param

Assistant ID to use for this run.

##### Param

Payload for creating a run.

##### Call Signature

> **stream**\<`TStreamMode`, `TSubgraphs`\>(`threadId`, `assistantId`, `payload`?): `TypedAsyncGenerator`\<`TStreamMode`, `TSubgraphs`, `TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:781](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L781)

###### Type Parameters

• **TStreamMode** *extends* `StreamMode` \| `StreamMode`[] = `StreamMode`

• **TSubgraphs** *extends* `boolean` = `false`

###### Parameters

####### threadId

`null`

####### assistantId

`string`

####### payload?

`Omit`\<`RunsStreamPayload`\<`TStreamMode`, `TSubgraphs`\>, `"multitaskStrategy"` \| `"onCompletion"`\>

###### Returns

`TypedAsyncGenerator`\<`TStreamMode`, `TSubgraphs`, `TStateType`, `TUpdateType`, `TCustomEventType`\>

##### Call Signature

> **stream**\<`TStreamMode`, `TSubgraphs`\>(`threadId`, `assistantId`, `payload`?): `TypedAsyncGenerator`\<`TStreamMode`, `TSubgraphs`, `TStateType`, `TUpdateType`, `TCustomEventType`\>

Defined in: [client.ts:799](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L799)

###### Type Parameters

• **TStreamMode** *extends* `StreamMode` \| `StreamMode`[] = `StreamMode`

• **TSubgraphs** *extends* `boolean` = `false`

###### Parameters

####### threadId

`string`

####### assistantId

`string`

####### payload?

`RunsStreamPayload`\<`TStreamMode`, `TSubgraphs`\>

###### Returns

`TypedAsyncGenerator`\<`TStreamMode`, `TSubgraphs`, `TStateType`, `TUpdateType`, `TCustomEventType`\>

***

#### wait()

Create a run and wait for it to complete.

##### Param

The ID of the thread.

##### Param

Assistant ID to use for this run.

##### Param

Payload for creating a run.

##### Call Signature

> **wait**(`threadId`, `assistantId`, `payload`?): `Promise`\<`DefaultValues`\>

Defined in: [client.ts:938](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L938)

###### Parameters

####### threadId

`null`

####### assistantId

`string`

####### payload?

`Omit`\<`RunsWaitPayload`, `"multitaskStrategy"` \| `"onCompletion"`\>

###### Returns

`Promise`\<`DefaultValues`\>

##### Call Signature

> **wait**(`threadId`, `assistantId`, `payload`?): `Promise`\<`DefaultValues`\>

Defined in: [client.ts:944](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L944)

###### Parameters

####### threadId

`string`

####### assistantId

`string`

####### payload?

`RunsWaitPayload`

###### Returns

`Promise`\<`DefaultValues`\>


<a name="classesstoreclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd)

***

[@langchain/langgraph-sdk](#readmemd) / StoreClient

## Class: StoreClient

Defined in: [client.ts:1175](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1175)

### Extends

- `BaseClient`

### Constructors

#### new StoreClient()

> **new StoreClient**(`config`?): [`StoreClient`](#classesstoreclientmd)

Defined in: [client.ts:88](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L88)

##### Parameters

###### config?

[`ClientConfig`](#interfacesclientconfigmd)

##### Returns

[`StoreClient`](#classesstoreclientmd)

##### Inherited from

`BaseClient.constructor`

### Methods

#### deleteItem()

> **deleteItem**(`namespace`, `key`): `Promise`\<`void`\>

Defined in: [client.ts:1296](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1296)

Delete an item.

##### Parameters

###### namespace

`string`[]

A list of strings representing the namespace path.

###### key

`string`

The unique identifier for the item.

##### Returns

`Promise`\<`void`\>

Promise<void>

***

#### getItem()

> **getItem**(`namespace`, `key`, `options`?): `Promise`\<`null` \| `Item`\>

Defined in: [client.ts:1252](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1252)

Retrieve a single item.

##### Parameters

###### namespace

`string`[]

A list of strings representing the namespace path.

###### key

`string`

The unique identifier for the item.

###### options?

####### refreshTtl?

`null` \| `boolean`

Whether to refresh the TTL on this read operation. If null, uses the store's default behavior.

##### Returns

`Promise`\<`null` \| `Item`\>

Promise<Item>

##### Example

```typescript
const item = await client.store.getItem(
  ["documents", "user123"],
  "item456",
  { refreshTtl: true }
);
console.log(item);
// {
//   namespace: ["documents", "user123"],
//   key: "item456",
//   value: { title: "My Document", content: "Hello World" },
//   createdAt: "2024-07-30T12:00:00Z",
//   updatedAt: "2024-07-30T12:00:00Z"
// }
```

***

#### listNamespaces()

> **listNamespaces**(`options`?): `Promise`\<`ListNamespaceResponse`\>

Defined in: [client.ts:1392](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1392)

List namespaces with optional match conditions.

##### Parameters

###### options?

####### limit?

`number`

Maximum number of namespaces to return (default is 100).

####### maxDepth?

`number`

Optional integer specifying the maximum depth of namespaces to return.

####### offset?

`number`

Number of namespaces to skip before returning results (default is 0).

####### prefix?

`string`[]

Optional list of strings representing the prefix to filter namespaces.

####### suffix?

`string`[]

Optional list of strings representing the suffix to filter namespaces.

##### Returns

`Promise`\<`ListNamespaceResponse`\>

Promise<ListNamespaceResponse>

***

#### putItem()

> **putItem**(`namespace`, `key`, `value`, `options`?): `Promise`\<`void`\>

Defined in: [client.ts:1196](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1196)

Store or update an item.

##### Parameters

###### namespace

`string`[]

A list of strings representing the namespace path.

###### key

`string`

The unique identifier for the item within the namespace.

###### value

`Record`\<`string`, `any`\>

A dictionary containing the item's data.

###### options?

####### index?

`null` \| `false` \| `string`[]

Controls search indexing - null (use defaults), false (disable), or list of field paths to index.

####### ttl?

`null` \| `number`

Optional time-to-live in minutes for the item, or null for no expiration.

##### Returns

`Promise`\<`void`\>

Promise<void>

##### Example

```typescript
await client.store.putItem(
  ["documents", "user123"],
  "item456",
  { title: "My Document", content: "Hello World" },
  { ttl: 60 } // expires in 60 minutes
);
```

***

#### searchItems()

> **searchItems**(`namespacePrefix`, `options`?): `Promise`\<`SearchItemsResponse`\>

Defined in: [client.ts:1347](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L1347)

Search for items within a namespace prefix.

##### Parameters

###### namespacePrefix

`string`[]

List of strings representing the namespace prefix.

###### options?

####### filter?

`Record`\<`string`, `any`\>

Optional dictionary of key-value pairs to filter results.

####### limit?

`number`

Maximum number of items to return (default is 10).

####### offset?

`number`

Number of items to skip before returning results (default is 0).

####### query?

`string`

Optional search query.

####### refreshTtl?

`null` \| `boolean`

Whether to refresh the TTL on items returned by this search. If null, uses the store's default behavior.

##### Returns

`Promise`\<`SearchItemsResponse`\>

Promise<SearchItemsResponse>

##### Example

```typescript
const results = await client.store.searchItems(
  ["documents"],
  {
    filter: { author: "John Doe" },
    limit: 5,
    refreshTtl: true
  }
);
console.log(results);
// {
//   items: [
//     {
//       namespace: ["documents", "user123"],
//       key: "item789",
//       value: { title: "Another Document", author: "John Doe" },
//       createdAt: "2024-07-30T12:00:00Z",
//       updatedAt: "2024-07-30T12:00:00Z"
//     },
//     // ... additional items ...
//   ]
// }
```


<a name="classesthreadsclientmd"></a>

[**@langchain/langgraph-sdk**](#readmemd)

***

[@langchain/langgraph-sdk](#readmemd) / ThreadsClient

## Class: ThreadsClient\<TStateType, TUpdateType\>

Defined in: [client.ts:489](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L489)

### Extends

- `BaseClient`

### Type Parameters

• **TStateType** = `DefaultValues`

• **TUpdateType** = `TStateType`

### Constructors

#### new ThreadsClient()

> **new ThreadsClient**\<`TStateType`, `TUpdateType`\>(`config`?): [`ThreadsClient`](#classesthreadsclientmd)\<`TStateType`, `TUpdateType`\>

Defined in: [client.ts:88](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L88)

##### Parameters

###### config?

[`ClientConfig`](#interfacesclientconfigmd)

##### Returns

[`ThreadsClient`](#classesthreadsclientmd)\<`TStateType`, `TUpdateType`\>

##### Inherited from

`BaseClient.constructor`

### Methods

#### copy()

> **copy**(`threadId`): `Promise`\<`Thread`\<`TStateType`\>\>

Defined in: [client.ts:566](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L566)

Copy an existing thread

##### Parameters

###### threadId

`string`

ID of the thread to be copied

##### Returns

`Promise`\<`Thread`\<`TStateType`\>\>

Newly copied thread

***

#### create()

> **create**(`payload`?): `Promise`\<`Thread`\<`TStateType`\>\>

Defined in: [client.ts:511](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L511)

Create a new thread.

##### Parameters

###### payload?

Payload for creating a thread.

####### graphId?

`string`

Graph ID to associate with the thread.

####### ifExists?

`OnConflictBehavior`

How to handle duplicate creation.

**Default**

```ts
"raise"
```

####### metadata?

`Metadata`

Metadata for the thread.

####### supersteps?

`object`[]

Apply a list of supersteps when creating a thread, each containing a sequence of updates.

Used for copying a thread between deployments.

####### threadId?

`string`

ID of the thread to create.

If not provided, a random UUID will be generated.

##### Returns

`Promise`\<`Thread`\<`TStateType`\>\>

The created thread.

***

#### delete()

> **delete**(`threadId`): `Promise`\<`void`\>

Defined in: [client.ts:599](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L599)

Delete a thread.

##### Parameters

###### threadId

`string`

ID of the thread.

##### Returns

`Promise`\<`void`\>

***

#### get()

> **get**\<`ValuesType`\>(`threadId`): `Promise`\<`Thread`\<`ValuesType`\>\>

Defined in: [client.ts:499](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L499)

Get a thread by ID.

##### Type Parameters

• **ValuesType** = `TStateType`

##### Parameters

###### threadId

`string`

ID of the thread.

##### Returns

`Promise`\<`Thread`\<`ValuesType`\>\>

The thread.

***

#### getHistory()

> **getHistory**\<`ValuesType`\>(`threadId`, `options`?): `Promise`\<`ThreadState`\<`ValuesType`\>[]\>

Defined in: [client.ts:752](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L752)

Get all past states for a thread.

##### Type Parameters

• **ValuesType** = `TStateType`

##### Parameters

###### threadId

`string`

ID of the thread.

###### options?

Additional options.

####### before?

`Config`

####### checkpoint?

`Partial`\<`Omit`\<`Checkpoint`, `"thread_id"`\>\>

####### limit?

`number`

####### metadata?

`Metadata`

##### Returns

`Promise`\<`ThreadState`\<`ValuesType`\>[]\>

List of thread states.

***

#### getState()

> **getState**\<`ValuesType`\>(`threadId`, `checkpoint`?, `options`?): `Promise`\<`ThreadState`\<`ValuesType`\>\>

Defined in: [client.ts:659](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L659)

Get state for a thread.

##### Type Parameters

• **ValuesType** = `TStateType`

##### Parameters

###### threadId

`string`

ID of the thread.

###### checkpoint?

`string` | `Checkpoint`

###### options?

####### subgraphs?

`boolean`

##### Returns

`Promise`\<`ThreadState`\<`ValuesType`\>\>

Thread state.

***

#### patchState()

> **patchState**(`threadIdOrConfig`, `metadata`): `Promise`\<`void`\>

Defined in: [client.ts:722](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L722)

Patch the metadata of a thread.

##### Parameters

###### threadIdOrConfig

Thread ID or config to patch the state of.

`string` | `Config`

###### metadata

`Metadata`

Metadata to patch the state with.

##### Returns

`Promise`\<`void`\>

***

#### search()

> **search**\<`ValuesType`\>(`query`?): `Promise`\<`Thread`\<`ValuesType`\>[]\>

Defined in: [client.ts:611](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L611)

List threads

##### Type Parameters

• **ValuesType** = `TStateType`

##### Parameters

###### query?

Query options

####### limit?

`number`

Maximum number of threads to return.
Defaults to 10

####### metadata?

`Metadata`

Metadata to filter threads by.

####### offset?

`number`

Offset to start from.

####### sortBy?

`ThreadSortBy`

Sort by.

####### sortOrder?

`SortOrder`

Sort order.
Must be one of 'asc' or 'desc'.

####### status?

`ThreadStatus`

Thread status to filter on.
Must be one of 'idle', 'busy', 'interrupted' or 'error'.

##### Returns

`Promise`\<`Thread`\<`ValuesType`\>[]\>

List of threads

***

#### update()

> **update**(`threadId`, `payload`?): `Promise`\<`Thread`\<`DefaultValues`\>\>

Defined in: [client.ts:579](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L579)

Update a thread.

##### Parameters

###### threadId

`string`

ID of the thread.

###### payload?

Payload for updating the thread.

####### metadata?

`Metadata`

Metadata for the thread.

##### Returns

`Promise`\<`Thread`\<`DefaultValues`\>\>

The updated thread.

***

#### updateState()

> **updateState**\<`ValuesType`\>(`threadId`, `options`): `Promise`\<`Pick`\<`Config`, `"configurable"`\>\>

Defined in: [client.ts:693](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L693)

Add state to a thread.

##### Type Parameters

• **ValuesType** = `TUpdateType`

##### Parameters

###### threadId

`string`

The ID of the thread.

###### options

####### asNode?

`string`

####### checkpoint?

`Checkpoint`

####### checkpointId?

`string`

####### values

`ValuesType`

##### Returns

`Promise`\<`Pick`\<`Config`, `"configurable"`\>\>


<a name="functionsgetapikeymd"></a>

[**@langchain/langgraph-sdk**](#readmemd)

***

[@langchain/langgraph-sdk](#readmemd) / getApiKey

## Function: getApiKey()

> **getApiKey**(`apiKey`?): `undefined` \| `string`

Defined in: [client.ts:53](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L53)

Get the API key from the environment.
Precedence:
  1. explicit argument
  2. LANGGRAPH_API_KEY
  3. LANGSMITH_API_KEY
  4. LANGCHAIN_API_KEY

### Parameters

#### apiKey?

`string`

Optional API key provided as an argument

### Returns

`undefined` \| `string`

The API key if found, otherwise undefined


<a name="interfacesclientconfigmd"></a>

[**@langchain/langgraph-sdk**](#readmemd)

***

[@langchain/langgraph-sdk](#readmemd) / ClientConfig

## Interface: ClientConfig

Defined in: [client.ts:71](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L71)

### Properties

#### apiKey?

> `optional` **apiKey**: `string`

Defined in: [client.ts:73](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L73)

***

#### apiUrl?

> `optional` **apiUrl**: `string`

Defined in: [client.ts:72](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L72)

***

#### callerOptions?

> `optional` **callerOptions**: `AsyncCallerParams`

Defined in: [client.ts:74](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L74)

***

#### defaultHeaders?

> `optional` **defaultHeaders**: `Record`\<`string`, `undefined` \| `null` \| `string`\>

Defined in: [client.ts:76](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L76)

***

#### timeoutMs?

> `optional` **timeoutMs**: `number`

Defined in: [client.ts:75](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/client.ts#L75)


<a name="reactreadmemd"></a>

**@langchain/langgraph-sdk**

***

## @langchain/langgraph-sdk/react

### Interfaces

- [UseStream](#reactinterfacesusestreammd)
- [UseStreamOptions](#reactinterfacesusestreamoptionsmd)

### Type Aliases

- [MessageMetadata](#reacttype-aliasesmessagemetadatamd)

### Functions

- [useStream](#reactfunctionsusestreammd)


<a name="reactfunctionsusestreammd"></a>

[**@langchain/langgraph-sdk**](#reactreadmemd)

***

[@langchain/langgraph-sdk](#reactreadmemd) / useStream

## Function: useStream()

> **useStream**\<`StateType`, `Bag`\>(`options`): [`UseStream`](#reactinterfacesusestreammd)\<`StateType`, `Bag`\>

Defined in: [react/stream.tsx:618](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L618)

### Type Parameters

• **StateType** *extends* `Record`\<`string`, `unknown`\> = `Record`\<`string`, `unknown`\>

• **Bag** *extends* `object` = `BagTemplate`

### Parameters

#### options

[`UseStreamOptions`](#reactinterfacesusestreamoptionsmd)\<`StateType`, `Bag`\>

### Returns

[`UseStream`](#reactinterfacesusestreammd)\<`StateType`, `Bag`\>


<a name="reactinterfacesusestreammd"></a>

[**@langchain/langgraph-sdk**](#reactreadmemd)

***

[@langchain/langgraph-sdk](#reactreadmemd) / UseStream

## Interface: UseStream\<StateType, Bag\>

Defined in: [react/stream.tsx:507](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L507)

### Type Parameters

• **StateType** *extends* `Record`\<`string`, `unknown`\> = `Record`\<`string`, `unknown`\>

• **Bag** *extends* `BagTemplate` = `BagTemplate`

### Properties

#### assistantId

> **assistantId**: `string`

Defined in: [react/stream.tsx:592](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L592)

The ID of the assistant to use.

***

#### branch

> **branch**: `string`

Defined in: [react/stream.tsx:542](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L542)

The current branch of the thread.

***

#### client

> **client**: `Client`

Defined in: [react/stream.tsx:587](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L587)

LangGraph SDK client used to send request and receive responses.

***

#### error

> **error**: `unknown`

Defined in: [react/stream.tsx:519](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L519)

Last seen error from the thread or during streaming.

***

#### experimental\_branchTree

> **experimental\_branchTree**: `Sequence`\<`StateType`\>

Defined in: [react/stream.tsx:558](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L558)

**`Experimental`**

Tree of all branches for the thread.

***

#### getMessagesMetadata()

> **getMessagesMetadata**: (`message`, `index`?) => `undefined` \| [`MessageMetadata`](#reacttype-aliasesmessagemetadatamd)\<`StateType`\>

Defined in: [react/stream.tsx:579](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L579)

Get the metadata for a message, such as first thread state the message
was seen in and branch information.

##### Parameters

###### message

`Message`

The message to get the metadata for.

###### index?

`number`

The index of the message in the thread.

##### Returns

`undefined` \| [`MessageMetadata`](#reacttype-aliasesmessagemetadatamd)\<`StateType`\>

The metadata for the message.

***

#### history

> **history**: `ThreadState`\<`StateType`\>[]

Defined in: [react/stream.tsx:552](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L552)

Flattened history of thread states of a thread.

***

#### interrupt

> **interrupt**: `undefined` \| `Interrupt`\<`GetInterruptType`\<`Bag`\>\>

Defined in: [react/stream.tsx:563](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L563)

Get the interrupt value for the stream if interrupted.

***

#### isLoading

> **isLoading**: `boolean`

Defined in: [react/stream.tsx:524](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L524)

Whether the stream is currently running.

***

#### messages

> **messages**: `Message`[]

Defined in: [react/stream.tsx:569](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L569)

Messages inferred from the thread.
Will automatically update with incoming message chunks.

***

#### setBranch()

> **setBranch**: (`branch`) => `void`

Defined in: [react/stream.tsx:547](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L547)

Set the branch of the thread.

##### Parameters

###### branch

`string`

##### Returns

`void`

***

#### stop()

> **stop**: () => `void`

Defined in: [react/stream.tsx:529](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L529)

Stops the stream.

##### Returns

`void`

***

#### submit()

> **submit**: (`values`, `options`?) => `void`

Defined in: [react/stream.tsx:534](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L534)

Create and stream a run to the thread.

##### Parameters

###### values

`undefined` | `null` | `GetUpdateType`\<`Bag`, `StateType`\>

###### options?

`SubmitOptions`\<`StateType`, `GetConfigurableType`\<`Bag`\>\>

##### Returns

`void`

***

#### values

> **values**: `StateType`

Defined in: [react/stream.tsx:514](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L514)

The current values of the thread.


<a name="reactinterfacesusestreamoptionsmd"></a>

[**@langchain/langgraph-sdk**](#reactreadmemd)

***

[@langchain/langgraph-sdk](#reactreadmemd) / UseStreamOptions

## Interface: UseStreamOptions\<StateType, Bag\>

Defined in: [react/stream.tsx:408](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L408)

### Type Parameters

• **StateType** *extends* `Record`\<`string`, `unknown`\> = `Record`\<`string`, `unknown`\>

• **Bag** *extends* `BagTemplate` = `BagTemplate`

### Properties

#### apiKey?

> `optional` **apiKey**: `string`

Defined in: [react/stream.tsx:430](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L430)

The API key to use.

***

#### apiUrl?

> `optional` **apiUrl**: `string`

Defined in: [react/stream.tsx:425](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L425)

The URL of the API to use.

***

#### assistantId

> **assistantId**: `string`

Defined in: [react/stream.tsx:415](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L415)

The ID of the assistant to use.

***

#### callerOptions?

> `optional` **callerOptions**: `AsyncCallerParams`

Defined in: [react/stream.tsx:435](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L435)

Custom call options, such as custom fetch implementation.

***

#### client?

> `optional` **client**: `Client`\<`DefaultValues`, `DefaultValues`, `unknown`\>

Defined in: [react/stream.tsx:420](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L420)

Client used to send requests.

***

#### defaultHeaders?

> `optional` **defaultHeaders**: `Record`\<`string`, `undefined` \| `null` \| `string`\>

Defined in: [react/stream.tsx:440](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L440)

Default headers to send with requests.

***

#### messagesKey?

> `optional` **messagesKey**: `string`

Defined in: [react/stream.tsx:448](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L448)

Specify the key within the state that contains messages.
Defaults to "messages".

##### Default

```ts
"messages"
```

***

#### onCustomEvent()?

> `optional` **onCustomEvent**: (`data`, `options`) => `void`

Defined in: [react/stream.tsx:470](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L470)

Callback that is called when a custom event is received.

##### Parameters

###### data

`GetCustomEventType`\<`Bag`\>

###### options

####### mutate

(`update`) => `void`

##### Returns

`void`

***

#### onDebugEvent()?

> `optional` **onDebugEvent**: (`data`) => `void`

Defined in: [react/stream.tsx:494](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L494)

**`Internal`**

Callback that is called when a debug event is received.
 This API is experimental and subject to change.

##### Parameters

###### data

`unknown`

##### Returns

`void`

***

#### onError()?

> `optional` **onError**: (`error`) => `void`

Defined in: [react/stream.tsx:453](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L453)

Callback that is called when an error occurs.

##### Parameters

###### error

`unknown`

##### Returns

`void`

***

#### onFinish()?

> `optional` **onFinish**: (`state`) => `void`

Defined in: [react/stream.tsx:458](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L458)

Callback that is called when the stream is finished.

##### Parameters

###### state

`ThreadState`\<`StateType`\>

##### Returns

`void`

***

#### onLangChainEvent()?

> `optional` **onLangChainEvent**: (`data`) => `void`

Defined in: [react/stream.tsx:488](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L488)

Callback that is called when a LangChain event is received.

##### Parameters

###### data

####### data

`unknown`

####### event

`string` & `object` \| `"on_tool_start"` \| `"on_tool_stream"` \| `"on_tool_end"` \| `"on_chat_model_start"` \| `"on_chat_model_stream"` \| `"on_chat_model_end"` \| `"on_llm_start"` \| `"on_llm_stream"` \| `"on_llm_end"` \| `"on_chain_start"` \| `"on_chain_stream"` \| `"on_chain_end"` \| `"on_retriever_start"` \| `"on_retriever_stream"` \| `"on_retriever_end"` \| `"on_prompt_start"` \| `"on_prompt_stream"` \| `"on_prompt_end"`

####### metadata

`Record`\<`string`, `unknown`\>

####### name

`string`

####### parent_ids

`string`[]

####### run_id

`string`

####### tags

`string`[]

##### Returns

`void`

##### See

https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_events/#stream-graph-in-events-mode for more details.

***

#### onMetadataEvent()?

> `optional` **onMetadataEvent**: (`data`) => `void`

Defined in: [react/stream.tsx:482](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L482)

Callback that is called when a metadata event is received.

##### Parameters

###### data

####### run_id

`string`

####### thread_id

`string`

##### Returns

`void`

***

#### onThreadId()?

> `optional` **onThreadId**: (`threadId`) => `void`

Defined in: [react/stream.tsx:504](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L504)

Callback that is called when the thread ID is updated (ie when a new thread is created).

##### Parameters

###### threadId

`string`

##### Returns

`void`

***

#### onUpdateEvent()?

> `optional` **onUpdateEvent**: (`data`) => `void`

Defined in: [react/stream.tsx:463](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L463)

Callback that is called when an update event is received.

##### Parameters

###### data

##### Returns

`void`

***

#### threadId?

> `optional` **threadId**: `null` \| `string`

Defined in: [react/stream.tsx:499](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L499)

The ID of the thread to fetch history and current values from.


<a name="reacttype-aliasesmessagemetadatamd"></a>

[**@langchain/langgraph-sdk**](#reactreadmemd)

***

[@langchain/langgraph-sdk](#reactreadmemd) / MessageMetadata

## Type Alias: MessageMetadata\<StateType\>

> **MessageMetadata**\<`StateType`\>: `object`

Defined in: [react/stream.tsx:169](https://github.com/langchain-ai/langgraph/blob/d4f644877db6264bd46d0b00fc4c37f174e502d5/libs/sdk-js/src/react/stream.tsx#L169)

### Type Parameters

• **StateType** *extends* `Record`\<`string`, `unknown`\>

### Type declaration

#### branch

> **branch**: `string` \| `undefined`

The branch of the message.

#### branchOptions

> **branchOptions**: `string`[] \| `undefined`

The list of branches this message is part of.
This is useful for displaying branching controls.

#### firstSeenState

> **firstSeenState**: `ThreadState`\<`StateType`\> \| `undefined`

The first thread state the message was seen in.

#### messageId

> **messageId**: `string`

The ID of the message used.
