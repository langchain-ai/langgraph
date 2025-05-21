from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.interrupt import HumanInterruptConfig, InterruptToolNode

from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:gpt-4.1", temperature=0)

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

agent = create_react_agent(
    llm,
    tools=[write_email],
    prompt="You are a helpful email assistant.",
    post_model_hook=InterruptToolNode(
        write_email=HumanInterruptConfig(
              allow_accept=True,
              allow_edit=True,
              allow_ignore=True,
              allow_respond=True,
          )
    ),
)