from langgraph.prebuilt.chat_agent_executor import make_agent_v2
from langgraph.prebuilt.handoff import handoff
from langgraph.graph import StateGraph
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


def test_foo() -> None:
    """Test foo."""
    bob_tool = handoff("bob", name="transfer_to_bob")
    alice_tool = handoff("alice", name="transfer_to_alice")
    charlie_tool = handoff("charlie", name="transfer_to_charlie")

    # State modifier now refers to the internal state of the agent (this is confusing).
    alice = make_agent_v2(model, [add, bob_tool], state_modifier="You're Alice.")
    bob = make_agent_v2(model, [add, alice_tool], state_modifier="You're Bob.")

    fake_chat_model = GenericFakeChatModel(
        messages=["Hello, I'm Alice.", "Hello, I'm Bob.", "Hello, I'm Charlie."]
    )
