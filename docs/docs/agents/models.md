---
search:
  boost: 2
tags:
  - anthropic
  - openai
  - agent
hide:
  - tags
---

# Models

This page describes how to configure the chat model used by an agent.

## Tool calling support

To enable tool-calling agents, the underlying LLM must support [tool calling](https://python.langchain.com/docs/concepts/tool_calling/).

Compatible models can be found in the [LangChain integrations directory](https://python.langchain.com/docs/integrations/chat/).

## Specifying a model by name

You can configure an agent with a model name string:

=== "OpenAI"

    ```python
    import os
    from langgraph.prebuilt import create_react_agent

    os.environ["OPENAI_API_KEY"] = "sk-..."

    agent = create_react_agent(
        # highlight-next-line
        model="openai:gpt-4.1",
        # other parameters
    )
    ```

=== "Anthropic"

    ```python
    import os
    from langgraph.prebuilt import create_react_agent

    os.environ["ANTHROPIC_API_KEY"] = "sk-..."

    agent = create_react_agent(
        # highlight-next-line
        model="anthropic:claude-3-7-sonnet-latest",
        # other parameters
    )
    ```

=== "Azure"

    ```python
    import os
    from langgraph.prebuilt import create_react_agent

    os.environ["AZURE_OPENAI_API_KEY"] = "..."
    os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
    os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

    agent = create_react_agent(
        # highlight-next-line
        model="azure_openai:gpt-4.1",
        # other parameters
    )
    ```

=== "Google Gemini"

    ```python
    import os
    from langgraph.prebuilt import create_react_agent

    os.environ["GOOGLE_API_KEY"] = "..."

    agent = create_react_agent(
        # highlight-next-line
        model="google_genai:gemini-2.0-flash",
        # other parameters
    )
    ```

=== "AWS Bedrock"

    ```python
    from langgraph.prebuilt import create_react_agent

    # Follow the steps here to configure your credentials:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

    agent = create_react_agent(
        # highlight-next-line
        model="bedrock_converse:anthropic.claude-3-5-sonnet-20240620-v1:0",
        # other parameters
    )
    ```


## Using `init_chat_model`

The [`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) utility simplifies model initialization with configurable parameters:

=== "OpenAI"

    ```
    pip install -U "langchain[openai]"
    ```
    ```python
    import os
    from langchain.chat_models import init_chat_model

    os.environ["OPENAI_API_KEY"] = "sk-..."

    model = init_chat_model(
        "openai:gpt-4.1",
        temperature=0,
        # other parameters
    )
    ```

=== "Anthropic"

    ```
    pip install -U "langchain[anthropic]"
    ```
    ```python
    import os
    from langchain.chat_models import init_chat_model

    os.environ["ANTHROPIC_API_KEY"] = "sk-..."

    model = init_chat_model(
        "anthropic:claude-3-5-sonnet-latest",
        temperature=0,
        # other parameters
    )
    ```

=== "Azure"

    ```
    pip install -U "langchain[openai]"
    ```
    ```python
    import os
    from langchain.chat_models import init_chat_model

    os.environ["AZURE_OPENAI_API_KEY"] = "..."
    os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
    os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

    model = init_chat_model(
        "azure_openai:gpt-4.1",
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0,
        # other parameters
    )
    ```

=== "Google Gemini"

    ```
    pip install -U "langchain[google-genai]"
    ```
    ```python
    import os
    from langchain.chat_models import init_chat_model

    os.environ["GOOGLE_API_KEY"] = "..."

    model = init_chat_model(
        "google_genai:gemini-2.0-flash",
        temperature=0,
        # other parameters
    )
    ```

=== "AWS Bedrock"

    ```
    pip install -U "langchain[aws]"
    ```
    ```python
    from langchain.chat_models import init_chat_model

    # Follow the steps here to configure your credentials:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

    model = init_chat_model(
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_provider="bedrock_converse",
        temperature=0,
        # other parameters
    )
    ```


Refer to the [API reference](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) for advanced options.

## Using provider-specific LLMs 

If a model provider is not available via `init_chat_model`, you can instantiate the provider's model class directly. The model must implement the [BaseChatModel interface](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html) and support tool calling:

```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

model = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    temperature=0,
    max_tokens=2048
)

agent = create_react_agent(
    # highlight-next-line
    model=model,
    # other parameters
)
```

!!! note "Illustrative example" 

    The example above uses `ChatAnthropic`, which is already supported by `init_chat_model`. This pattern is shown to illustrate how to manually instantiate a model not available through init_chat_model.

## Disable streaming

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

## Adding model fallbacks

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

## Additional resources

- [Model integration directory](https://python.langchain.com/docs/integrations/chat/)
- [Universal initialization with `init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/)
