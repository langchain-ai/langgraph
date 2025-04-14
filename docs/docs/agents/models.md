# Models

To switch the model your agent uses, you can simply change the model name string. To control model configuration (temperature, max tokens, etc.), you can use `init_chat_model`:

=== "Model name"

    ```python
    anthropic_agent = create_react_agent(
        # highlight-next-line
        model="anthropic:claude-3-7-sonnet-latest",
        ...
    )

    openai_agent = create_react_agent(
        # highlight-next-line
        model="openai:gpt-4.1",
        ...
    )
    ```

=== "`init_chat_model`"

    ```python
    from langchain.chat_models import init_chat_model

    # highlight-next-line
    anthropic_model = init_chat_model(
        "anthropic:claude-3-7-sonnet-latest", temperature=0
    )
    # highlight-next-line
    openai_model = init_chat_model(
        "openai:gpt-4.1", max_tokens=2048
    )
    anthropic_agent = create_react_agent(
        # highlight-next-line
        model=anthropic_model
        ...
    )
    openai_agent = create_react_agent(
        # highlight-next-line
        model=openai_model
        ...
    )
    ```

To learn more about how to use `init_chat_model`, check out [this guide](https://python.langchain.com/docs/how_to/chat_models_universal_init/).

