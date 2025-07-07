[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / Auth

# Class: Auth\<TExtra, TAuthReturn, TUser\>

Defined in: [src/auth/index.ts:11](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/index.ts#L11)

## Type Parameters

• **TExtra** = \{\}

• **TAuthReturn** *extends* `BaseAuthReturn` = `BaseAuthReturn`

• **TUser** *extends* `BaseUser` = `ToUserLike`\<`TAuthReturn`\>

## Constructors

### new Auth()

> **new Auth**\<`TExtra`, `TAuthReturn`, `TUser`\>(): [`Auth`](Auth.md)\<`TExtra`, `TAuthReturn`, `TUser`\>

#### Returns

[`Auth`](Auth.md)\<`TExtra`, `TAuthReturn`, `TUser`\>

## Methods

### authenticate()

> **authenticate**\<`T`\>(`cb`): [`Auth`](Auth.md)\<`TExtra`, `T`\>

Defined in: [src/auth/index.ts:25](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/index.ts#L25)

#### Type Parameters

• **T** *extends* `BaseAuthReturn`

#### Parameters

##### cb

`AuthenticateCallback`\<`T`\>

#### Returns

[`Auth`](Auth.md)\<`TExtra`, `T`\>

***

### on()

> **on**\<`T`\>(`event`, `callback`): `this`

Defined in: [src/auth/index.ts:32](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/auth/index.ts#L32)

#### Type Parameters

• **T** *extends* `CallbackEvent`

#### Parameters

##### event

`T`

##### callback

`OnCallback`\<`T`, `TUser`\>

#### Returns

`this`
