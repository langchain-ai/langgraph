# Models

LangGraph provides built-in support for [LLMs (language models)](https://python.langchain.com/docs/concepts/chat_models/) via the LangChain library. This makes it easy to integrate various LLMs into your agents and workflows.

## Initialize a model

:::python
Use [`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) to initialize models:

{% include-markdown "../../snippets/chat_model_tabs.md" %}
:::

:::js
Use model provider classes to initialize models:

=== "OpenAI"

    ```typescript
    import { ChatOpenAI } from "@langchain/openai";

    const model = new ChatOpenAI({
      model: "gpt-4o",
      temperature: 0,
    });
    ```

=== "Anthropic"

    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";

    const model = new ChatAnthropic({
      model: "claude-3-5-sonnet-20240620",
      temperature: 0,
      maxTokens: 2048,
    });
    ```

=== "Google"

    ```typescript
    import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-pro",
      temperature: 0,
    });
    ```

=== "Groq"

    ```typescript
    import { ChatGroq } from "@langchain/groq";

    const model = new ChatGroq({
      model: "llama-3.1-70b-versatile",
      temperature: 0,
    });
    ```

:::

:::python

### Instantiate a model directly

If a model provider is not available via `init_chat_model`, you can instantiate the provider's model class directly. The model must implement the [BaseChatModel interface](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html) and support tool calling:

```python
# Anthropic is already supported by `init_chat_model`,
# but you can also instantiate it directly.
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
  model="claude-3-7-sonnet-latest",
  temperature=0,
  max_tokens=2048
)
```

:::

!!! important "Tool calling support"

    If you are building an agent or workflow that requires the model to call external tools, ensure that the underlying
    language model supports [tool calling](../concepts/tools.md). Compatible models can be found in the [LangChain integrations directory](https://python.langchain.com/docs/integrations/chat/).

## Use in an agent

:::python
When using `create_react_agent` you can specify the model by its name string, which is a shorthand for initializing the model using `init_chat_model`. This allows you to use the model without needing to import or instantiate it directly.

=== "model name"

      ```python
      from langgraph.prebuilt import create_react_agent

      create_react_agent(
         # highlight-next-line
         model="anthropic:claude-3-7-sonnet-latest",
         # other parameters
      )
      ```

=== "model instance"

      ```python
      from langchain_anthropic import ChatAnthropic
      from langgraph.prebuilt import create_react_agent

      model = ChatAnthropic(
          model="claude-3-7-sonnet-latest",
          temperature=0,
          max_tokens=2048
      )
      # Alternatively
      # model = init_chat_model("anthropic:claude-3-7-sonnet-latest")

      agent = create_react_agent(
        # highlight-next-line
        model=model,
        # other parameters
      )
      ```

:::

:::js
When using `createReactAgent` you can pass the model instance directly:

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

const agent = createReactAgent({
  llm: model,
  tools: tools,
});
```

:::

:::python

### Dynamic model selection

Pass a callable function to `create_react_agent` to dynamically select the model at runtime. This is useful for scenarios where you want to choose a model based on user input, configuration settings, or other runtime conditions.

The selector function must return a chat model. If you're using tools, you must bind the tools to the model within the selector function.

  ```python
from dataclasses import dataclass
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime

@tool
def weather() -> str:
    """Returns the current weather conditions."""
    return "It's nice and sunny."


# Define the runtime context
@dataclass
class CustomContext:
    provider: Literal["anthropic", "openai"]

# Initialize models
openai_model = init_chat_model("openai:gpt-4o")
anthropic_model = init_chat_model("anthropic:claude-sonnet-4-20250514")


# Selector function for model choice
def select_model(state: AgentState, runtime: Runtime[CustomContext]) -> BaseChatModel:
    if runtime.context.provider == "anthropic":
        model = anthropic_model
    elif runtime.context.provider == "openai":
        model = openai_model
    else:
        raise ValueError(f"Unsupported provider: {runtime.context.provider}")

    # With dynamic model selection, you must bind tools explicitly
    return model.bind_tools([weather])


# Create agent with dynamic model selection
agent = create_react_agent(select_model, tools=[weather])

# Invoke with context to select model
output = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Which model is handling this?",
            }
        ]
    },
    context=CustomContext(provider="openai"),
)

print(output["messages"][-1].text())
```

!!! version-added "Added in version 0.6.0"

:::

## Advanced model configuration

### Disable streaming

:::python
To disable streaming of the individual LLM tokens, set `disable_streaming=True` when initializing the model:

=== "`init_chat_model`"

    ```python
    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        "anthropic:claude-3-7-sonnet-latest",
        # highlight-next-line
        disable_streaming=True
    )
    ```

=== "`ChatModel`"

    ```python
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(
        model="claude-3-7-sonnet-latest",
        # highlight-next-line
        disable_streaming=True
    )
    ```

Refer to the [API reference](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html#langchain_core.language_models.chat_models.BaseChatModel.disable_streaming) for more information on `disable_streaming`
:::

:::js
To disable streaming of the individual LLM tokens, set `streaming: false` when initializing the model:

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o",
  streaming: false,
});
```

:::

### Add model fallbacks

:::python
You can add a fallback to a different model or a different LLM provider using `model.with_fallbacks([...])`:

=== "`init_chat_model`"

    ```python
    from langchain.chat_models import init_chat_model

    model_with_fallbacks = (
        init_chat_model("anthropic:claude-3-5-haiku-latest")
        # highlight-next-line
        .with_fallbacks([
            init_chat_model("openai:gpt-4.1-mini"),
        ])
    )
    ```

=== "`ChatModel`"

    ```python
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    model_with_fallbacks = (
        ChatAnthropic(model="claude-3-5-haiku-latest")
        # highlight-next-line
        .with_fallbacks([
            ChatOpenAI(model="gpt-4.1-mini"),
        ])
    )
    ```

See this [guide](https://python.langchain.com/docs/how_to/fallbacks/#fallback-to-better-model) for more information on model fallbacks.
:::

:::js
You can add a fallback to a different model or a different LLM provider using `model.withFallbacks([...])`:

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";

const modelWithFallbacks = new ChatOpenAI({
  model: "gpt-4o",
}).withFallbacks([
  new ChatAnthropic({
    model: "claude-3-5-sonnet-20240620",
  }),
]);
```

See this [guide](https://js.langchain.com/docs/how_to/fallbacks/#fallback-to-better-model) for more information on model fallbacks.
:::

:::python

### Use the built-in rate limiter

Langchain includes a built-in in-memory rate limiter. This rate limiter is thread safe and can be shared by multiple threads in the same process.

```python
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_anthropic import ChatAnthropic

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

model = ChatAnthropic(
   model_name="claude-3-opus-20240229",
   rate_limiter=rate_limiter
)
```

See the LangChain docs for more information on how to [handle rate limiting](https://python.langchain.com/docs/how_to/chat_model_rate_limiting/).
:::

## Bring your own model

If your desired LLM isn't officially supported by LangChain, consider these options:

:::python

1. **Implement a custom LangChain chat model**: Create a model conforming to the [LangChain chat model interface](https://python.langchain.com/docs/how_to/custom_chat_model/). This enables full compatibility with LangGraph's agents and workflows but requires understanding of the LangChain framework.

   :::

:::js

1. **Implement a custom LangChain chat model**: Create a model conforming to the [LangChain chat model interface](https://js.langchain.com/docs/how_to/custom_chat/). This enables full compatibility with LangGraph's agents and workflows but requires understanding of the LangChain framework.

   :::

2. **Direct invocation with custom streaming**: Use your model directly by [adding custom streaming logic](../how-tos/streaming.md#use-with-any-llm) with `StreamWriter`.
   Refer to the [custom streaming documentation](../how-tos/streaming.md#use-with-any-llm) for guidance. This approach suits custom workflows where prebuilt agent integration is not necessary.

## Additional resources

:::python

- [Multimodal inputs](https://python.langchain.com/docs/how_to/multimodal_inputs/)
- [Structured outputs](https://python.langchain.com/docs/how_to/structured_output/)
- [Model integration directory](https://python.langchain.com/docs/integrations/chat/)
- [Force model to call a specific tool](https://python.langchain.com/docs/how_to/tool_choice/)
- [All chat model how-to guides](https://python.langchain.com/docs/how_to/#chat-models)
- [Chat model integrations](https://python.langchain.com/docs/integrations/chat/)

  :::

:::js

- [Multimodal inputs](https://js.langchain.com/docs/how_to/multimodal_inputs/)
- [Structured outputs](https://js.langchain.com/docs/how_to/structured_output/)
- [Model integration directory](https://js.langchain.com/docs/integrations/chat/)
- [Force model to call a specific tool](https://js.langchain.com/docs/how_to/tool_choice/)
- [All chat model how-to guides](https://js.langchain.com/docs/how_to/#chat-models)
- [Chat model integrations](https://js.langchain.com/docs/integrations/chat/)

  :::
