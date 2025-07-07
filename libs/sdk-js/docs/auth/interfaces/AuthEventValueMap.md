[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / AuthEventValueMap

# Interface: AuthEventValueMap

Defined in: [src/auth/types.ts:218](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L218)

## Properties

### assistants:create

> **assistants:create**: `object`

Defined in: [src/auth/types.ts:226](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L226)

#### assistant\_id?

> `optional` **assistant\_id**: `Maybe`\<`string`\>

#### config?

> `optional` **config**: `Maybe`\<`AssistantConfig`\>

#### graph\_id

> **graph\_id**: `string`

#### if\_exists?

> `optional` **if\_exists**: `Maybe`\<`"raise"` \| `"do_nothing"`\>

#### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### name?

> `optional` **name**: `Maybe`\<`string`\>

***

### assistants:delete

> **assistants:delete**: `object`

Defined in: [src/auth/types.ts:229](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L229)

#### assistant\_id

> **assistant\_id**: `string`

***

### assistants:read

> **assistants:read**: `object`

Defined in: [src/auth/types.ts:227](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L227)

#### assistant\_id

> **assistant\_id**: `string`

#### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

***

### assistants:search

> **assistants:search**: `object`

Defined in: [src/auth/types.ts:230](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L230)

#### graph\_id?

> `optional` **graph\_id**: `Maybe`\<`string`\>

#### limit?

> `optional` **limit**: `Maybe`\<`number`\>

#### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### offset?

> `optional` **offset**: `Maybe`\<`number`\>

***

### assistants:update

> **assistants:update**: `object`

Defined in: [src/auth/types.ts:228](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L228)

#### assistant\_id

> **assistant\_id**: `string`

#### config?

> `optional` **config**: `Maybe`\<`AssistantConfig`\>

#### graph\_id?

> `optional` **graph\_id**: `Maybe`\<`string`\>

#### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### name?

> `optional` **name**: `Maybe`\<`string`\>

#### version?

> `optional` **version**: `Maybe`\<`number`\>

***

### crons:create

> **crons:create**: `object`

Defined in: [src/auth/types.ts:232](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L232)

#### cron\_id?

> `optional` **cron\_id**: `Maybe`\<`string`\>

#### end\_time?

> `optional` **end\_time**: `Maybe`\<`string`\>

#### payload?

> `optional` **payload**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### schedule

> **schedule**: `string`

#### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

#### user\_id?

> `optional` **user\_id**: `Maybe`\<`string`\>

***

### crons:delete

> **crons:delete**: `object`

Defined in: [src/auth/types.ts:235](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L235)

#### cron\_id

> **cron\_id**: `string`

***

### crons:read

> **crons:read**: `object`

Defined in: [src/auth/types.ts:233](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L233)

#### cron\_id

> **cron\_id**: `string`

***

### crons:search

> **crons:search**: `object`

Defined in: [src/auth/types.ts:236](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L236)

#### assistant\_id?

> `optional` **assistant\_id**: `Maybe`\<`string`\>

#### limit?

> `optional` **limit**: `Maybe`\<`number`\>

#### offset?

> `optional` **offset**: `Maybe`\<`number`\>

#### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

### crons:update

> **crons:update**: `object`

Defined in: [src/auth/types.ts:234](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L234)

#### cron\_id

> **cron\_id**: `string`

#### payload?

> `optional` **payload**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### schedule?

> `optional` **schedule**: `Maybe`\<`string`\>

***

### store:delete

> **store:delete**: `object`

Defined in: [src/auth/types.ts:242](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L242)

#### key

> **key**: `string`

#### namespace?

> `optional` **namespace**: `Maybe`\<`string`[]\>

***

### store:get

> **store:get**: `object`

Defined in: [src/auth/types.ts:239](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L239)

#### key

> **key**: `string`

#### namespace

> **namespace**: `Maybe`\<`string`[]\>

***

### store:list\_namespaces

> **store:list\_namespaces**: `object`

Defined in: [src/auth/types.ts:241](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L241)

#### limit?

> `optional` **limit**: `Maybe`\<`number`\>

#### max\_depth?

> `optional` **max\_depth**: `Maybe`\<`number`\>

#### namespace?

> `optional` **namespace**: `Maybe`\<`string`[]\>

#### offset?

> `optional` **offset**: `Maybe`\<`number`\>

#### suffix?

> `optional` **suffix**: `Maybe`\<`string`[]\>

***

### store:put

> **store:put**: `object`

Defined in: [src/auth/types.ts:238](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L238)

#### key

> **key**: `string`

#### namespace

> **namespace**: `string`[]

#### value

> **value**: `Record`\<`string`, `unknown`\>

***

### store:search

> **store:search**: `object`

Defined in: [src/auth/types.ts:240](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L240)

#### filter?

> `optional` **filter**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### limit?

> `optional` **limit**: `Maybe`\<`number`\>

#### namespace?

> `optional` **namespace**: `Maybe`\<`string`[]\>

#### offset?

> `optional` **offset**: `Maybe`\<`number`\>

#### query?

> `optional` **query**: `Maybe`\<`string`\>

***

### threads:create

> **threads:create**: `object`

Defined in: [src/auth/types.ts:219](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L219)

#### if\_exists?

> `optional` **if\_exists**: `Maybe`\<`"raise"` \| `"do_nothing"`\>

#### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

### threads:create\_run

> **threads:create\_run**: `object`

Defined in: [src/auth/types.ts:224](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L224)

#### after\_seconds?

> `optional` **after\_seconds**: `Maybe`\<`number`\>

#### assistant\_id

> **assistant\_id**: `string`

#### if\_not\_exists?

> `optional` **if\_not\_exists**: `Maybe`\<`"reject"` \| `"create"`\>

#### kwargs

> **kwargs**: `Record`\<`string`, `unknown`\>

#### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### multitask\_strategy?

> `optional` **multitask\_strategy**: `Maybe`\<`"reject"` \| `"interrupt"` \| `"rollback"` \| `"enqueue"`\>

#### prevent\_insert\_if\_inflight?

> `optional` **prevent\_insert\_if\_inflight**: `Maybe`\<`boolean`\>

#### run\_id

> **run\_id**: `string`

#### status

> **status**: `Maybe`\<`"pending"` \| `"running"` \| `"error"` \| `"success"` \| `"timeout"` \| `"interrupted"`\>

#### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

### threads:delete

> **threads:delete**: `object`

Defined in: [src/auth/types.ts:222](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L222)

#### run\_id?

> `optional` **run\_id**: `Maybe`\<`string`\>

#### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

### threads:read

> **threads:read**: `object`

Defined in: [src/auth/types.ts:220](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L220)

#### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

***

### threads:search

> **threads:search**: `object`

Defined in: [src/auth/types.ts:223](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L223)

#### limit?

> `optional` **limit**: `Maybe`\<`number`\>

#### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### offset?

> `optional` **offset**: `Maybe`\<`number`\>

#### status?

> `optional` **status**: `Maybe`\<`"error"` \| `"interrupted"` \| `"idle"` \| `"busy"` \| `string` & `object`\>

#### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>

#### values?

> `optional` **values**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

***

### threads:update

> **threads:update**: `object`

Defined in: [src/auth/types.ts:221](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/types.ts#L221)

#### action?

> `optional` **action**: `Maybe`\<`"interrupt"` \| `"rollback"`\>

#### metadata?

> `optional` **metadata**: `Maybe`\<`Record`\<`string`, `unknown`\>\>

#### thread\_id?

> `optional` **thread\_id**: `Maybe`\<`string`\>
