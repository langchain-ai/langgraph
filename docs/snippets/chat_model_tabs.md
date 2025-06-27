=== "OpenAI"

    ```shell
    pip install -U "langchain[openai]"
    ```
    ```python
    import os
    from langchain.chat_models import init_chat_model

    os.environ["OPENAI_API_KEY"] = "sk-..."

    llm = init_chat_model("openai:gpt-4.1")
    ```

    👉 Read the [OpenAI integration docs](https://python.langchain.com/docs/integrations/chat/openai/)

=== "Anthropic"

    ```shell
    pip install -U "langchain[anthropic]"
    ```
    ```python
    import os
    from langchain.chat_models import init_chat_model

    os.environ["ANTHROPIC_API_KEY"] = "sk-..."

    llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
    ```

    👉 Read the [Anthropic integration docs](https://python.langchain.com/docs/integrations/chat/anthropic/)

=== "Azure"

    ```shell
    pip install -U "langchain[openai]"
    ```
    ```python
    import os
    from langchain.chat_models import init_chat_model

    os.environ["AZURE_OPENAI_API_KEY"] = "..."
    os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
    os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

    llm = init_chat_model(
        "azure_openai:gpt-4.1",
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    )
    ```
 
    👉 Read the [Azure integration docs](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)

=== "Google Gemini"

    ```shell
    pip install -U "langchain[google-genai]"
    ```
    ```python
    import os
    from langchain.chat_models import init_chat_model

    os.environ["GOOGLE_API_KEY"] = "..."

    llm = init_chat_model("google_genai:gemini-2.0-flash")
    ```

    👉 Read the [Google GenAI integration docs](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)

=== "AWS Bedrock"

    ```shell
    pip install -U "langchain[aws]"
    ```
    ```python
    from langchain.chat_models import init_chat_model

    # Follow the steps here to configure your credentials:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

    llm = init_chat_model(
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_provider="bedrock_converse",
    )
    ```

    👉 Read the [AWS Bedrock integration docs](https://python.langchain.com/docs/integrations/chat/bedrock/)
