try:
    import sys

    from langgraph_cli.cli import cli

except ImportError:
    # Provide more detailed error message with installation instructions
    error_message = (
        "\nError: langgraph_cli package not found.\n\n"
        "This could be due to one of the following reasons:\n"
        "1. You haven't installed the CLI package\n"
        "2. Your virtual environment doesn't have the package installed\n"
        "3. There's a path issue with your Python environment\n\n"
        "To fix this, try one of the following solutions:\n"
        "- Install the CLI package: pip install langgraph-cli\n"
        "- If you're using a virtual environment, activate it first\n"
        "- Use the standalone CLI directly by running: langgraph\n\n"
        "For more help, visit: https://github.com/langchain-ai/langgraph/tree/main/libs/cli"
    )
    raise ImportError(error_message)

try:
    cli()
except Exception as e:
    # Catch any exceptions that might occur when running the CLI
    print(f"\nError occurred while running langgraph CLI: {str(e)}")
    print(
        "If this problem persists, please report it at: https://github.com/langchain-ai/langgraph/issues"
    )
    sys.exit(1)
