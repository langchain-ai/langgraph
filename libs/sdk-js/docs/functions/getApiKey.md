[**@langchain/langgraph-sdk**](../README.md)

***

[@langchain/langgraph-sdk](../README.md) / getApiKey

# Function: getApiKey()

> **getApiKey**(`apiKey`?): `undefined` \| `string`

Defined in: [client.ts:122](https://github.com/langchain-ai/langgraph/blob/a7ea5e44ce12e3618d1a766587163afbfe424e04/libs/sdk-js/src/client.ts#L122)

Get the API key from the environment.
Precedence:
  1. explicit argument
  2. LANGGRAPH_API_KEY
  3. LANGSMITH_API_KEY
  4. LANGCHAIN_API_KEY

## Parameters

### apiKey?

`string`

Optional API key provided as an argument

## Returns

`undefined` \| `string`

The API key if found, otherwise undefined
