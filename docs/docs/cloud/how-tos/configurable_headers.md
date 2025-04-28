# Configurable Headers

LangGraph allows runtime configuration to modify agent behavior and permissions dynamically. When using the [LangGraph Platform](../quick_start.md), you can pass this configuration in the request body (`config`) or specific request headers. This enables adjustments based on user identity or other request data (see the [configuration how-to](../../how-tos/configuration.ipynb) for more details on how to access within your graph).

For privacy, control which headers are passed to the runtime configuration via the `http.configurable_headers` section in your `langgraph.json` file.

Here's how to customize the included and excluded headers:

```json
{
  "http": {
    "configurable_headers": {
      "include": ["x-user-id", "x-organization-id", "my-prefix-*"],
      "exclude": ["authorization", "x-api-key"]
    }
  }
}
```

The `include` and `exclude` lists accept exact header names or patterns using `*` to match any number of characters. For your security, no other regex patterns are supported.

If you'd like to opt-out of configurable headers, you can simply set a wildcard pattern in the `exclude` list:

```json
{
  "http": {
    "configurable_headers": {
      "exclude": ["*"]
    }
  }
}
```

This will exclude all headers from being added to your run's configuration.

Note that exclusions take precedence over inclusions.