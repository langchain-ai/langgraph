[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / StoreClient

# Class: StoreClient

Defined in: [client.ts:1347](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1347)

## Extends

- `BaseClient`

## Constructors

### new StoreClient()

> **new StoreClient**(`config`?): [`StoreClient`](StoreClient.md)

Defined in: [client.ts:183](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L183)

#### Parameters

##### config?

[`ClientConfig`](../interfaces/ClientConfig.md)

#### Returns

[`StoreClient`](StoreClient.md)

#### Inherited from

`BaseClient.constructor`

## Methods

### deleteItem()

> **deleteItem**(`namespace`, `key`): `Promise`\<`void`\>

Defined in: [client.ts:1468](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1468)

Delete an item.

#### Parameters

##### namespace

`string`[]

A list of strings representing the namespace path.

##### key

`string`

The unique identifier for the item.

#### Returns

`Promise`\<`void`\>

Promise<void>

***

### getItem()

> **getItem**(`namespace`, `key`, `options`?): `Promise`\<`null` \| `Item`\>

Defined in: [client.ts:1424](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1424)

Retrieve a single item.

#### Parameters

##### namespace

`string`[]

A list of strings representing the namespace path.

##### key

`string`

The unique identifier for the item.

##### options?

###### refreshTtl?

`null` \| `boolean`

Whether to refresh the TTL on this read operation. If null, uses the store's default behavior.

#### Returns

`Promise`\<`null` \| `Item`\>

Promise<Item>

#### Example

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

### listNamespaces()

> **listNamespaces**(`options`?): `Promise`\<`ListNamespaceResponse`\>

Defined in: [client.ts:1564](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1564)

List namespaces with optional match conditions.

#### Parameters

##### options?

###### limit?

`number`

Maximum number of namespaces to return (default is 100).

###### maxDepth?

`number`

Optional integer specifying the maximum depth of namespaces to return.

###### offset?

`number`

Number of namespaces to skip before returning results (default is 0).

###### prefix?

`string`[]

Optional list of strings representing the prefix to filter namespaces.

###### suffix?

`string`[]

Optional list of strings representing the suffix to filter namespaces.

#### Returns

`Promise`\<`ListNamespaceResponse`\>

Promise<ListNamespaceResponse>

***

### putItem()

> **putItem**(`namespace`, `key`, `value`, `options`?): `Promise`\<`void`\>

Defined in: [client.ts:1368](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1368)

Store or update an item.

#### Parameters

##### namespace

`string`[]

A list of strings representing the namespace path.

##### key

`string`

The unique identifier for the item within the namespace.

##### value

`Record`\<`string`, `any`\>

A dictionary containing the item's data.

##### options?

###### index?

`null` \| `false` \| `string`[]

Controls search indexing - null (use defaults), false (disable), or list of field paths to index.

###### ttl?

`null` \| `number`

Optional time-to-live in minutes for the item, or null for no expiration.

#### Returns

`Promise`\<`void`\>

Promise<void>

#### Example

```typescript
await client.store.putItem(
  ["documents", "user123"],
  "item456",
  { title: "My Document", content: "Hello World" },
  { ttl: 60 } // expires in 60 minutes
);
```

***

### searchItems()

> **searchItems**(`namespacePrefix`, `options`?): `Promise`\<`SearchItemsResponse`\>

Defined in: [client.ts:1519](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L1519)

Search for items within a namespace prefix.

#### Parameters

##### namespacePrefix

`string`[]

List of strings representing the namespace prefix.

##### options?

###### filter?

`Record`\<`string`, `any`\>

Optional dictionary of key-value pairs to filter results.

###### limit?

`number`

Maximum number of items to return (default is 10).

###### offset?

`number`

Number of items to skip before returning results (default is 0).

###### query?

`string`

Optional search query.

###### refreshTtl?

`null` \| `boolean`

Whether to refresh the TTL on items returned by this search. If null, uses the store's default behavior.

#### Returns

`Promise`\<`SearchItemsResponse`\>

Promise<SearchItemsResponse>

#### Example

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
