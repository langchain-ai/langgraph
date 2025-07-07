[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / CronsClient

# Class: CronsClient

Defined in: [client.ts:334](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L334)

## Extends

- `BaseClient`

## Constructors

### new CronsClient()

> **new CronsClient**(`config`?): [`CronsClient`](CronsClient.md)

Defined in: [client.ts:183](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L183)

#### Parameters

##### config?

[`ClientConfig`](../interfaces/ClientConfig.md)

#### Returns

[`CronsClient`](CronsClient.md)

#### Inherited from

`BaseClient.constructor`

## Methods

### create()

> **create**(`assistantId`, `payload`?): `Promise`\<`CronCreateResponse`\>

Defined in: [client.ts:375](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L375)

#### Parameters

##### assistantId

`string`

Assistant ID to use for this cron job.

##### payload?

`CronsCreatePayload`

Payload for creating a cron job.

#### Returns

`Promise`\<`CronCreateResponse`\>

***

### createForThread()

> **createForThread**(`threadId`, `assistantId`, `payload`?): `Promise`\<`CronCreateForThreadResponse`\>

Defined in: [client.ts:342](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L342)

#### Parameters

##### threadId

`string`

The ID of the thread.

##### assistantId

`string`

Assistant ID to use for this cron job.

##### payload?

`CronsCreatePayload`

Payload for creating a cron job.

#### Returns

`Promise`\<`CronCreateForThreadResponse`\>

The created background run.

***

### delete()

> **delete**(`cronId`): `Promise`\<`void`\>

Defined in: [client.ts:402](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L402)

#### Parameters

##### cronId

`string`

Cron ID of Cron job to delete.

#### Returns

`Promise`\<`void`\>

***

### search()

> **search**(`query`?): `Promise`\<`Cron`[]\>

Defined in: [client.ts:413](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L413)

#### Parameters

##### query?

Query options.

###### assistantId?

`string`

###### limit?

`number`

###### offset?

`number`

###### threadId?

`string`

#### Returns

`Promise`\<`Cron`[]\>

List of crons.
