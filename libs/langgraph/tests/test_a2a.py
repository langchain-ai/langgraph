"""Tests for the declarative A2A card attachment system."""

from __future__ import annotations

import sys
from typing import Annotated

import pytest

from langgraph.a2a.types import AgentCard, AgentSkill
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_graph():
    """Return a minimal compiled StateGraph."""

    class State(dict):
        messages: Annotated[list, lambda a, b: a + b]

    def noop(state: State) -> State:
        return state

    builder = StateGraph(State)
    builder.add_node("noop", noop)
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    return builder.compile()


@pytest.fixture
def simple_graph():
    return _make_graph()


# ---------------------------------------------------------------------------
# with_agent_card
# ---------------------------------------------------------------------------


def test_with_agent_card_attaches_card(simple_graph):
    card = AgentCard(name="Test", description="desc", url="http://localhost")
    g = simple_graph.with_agent_card(card)
    assert g.agent_card is card


def test_with_agent_card_returns_self(simple_graph):
    card = AgentCard(name="Test", description="desc", url="http://localhost")
    assert simple_graph.with_agent_card(card) is simple_graph


# ---------------------------------------------------------------------------
# serve_a2a pre-condition
# ---------------------------------------------------------------------------


def test_serve_a2a_raises_without_card(simple_graph):
    with pytest.raises(RuntimeError, match="with_agent_card"):
        simple_graph.serve_a2a()


# ---------------------------------------------------------------------------
# AgentCard.to_dict
# ---------------------------------------------------------------------------


def test_agent_card_to_dict_shape():
    skill = AgentSkill(
        id="s1",
        name="Search",
        description="Searches web",
        tags=["search"],
        examples=["find X"],
    )
    card = AgentCard(
        name="MyAgent",
        description="Does things",
        url="https://agent.example.com",
        skills=[skill],
    )
    d = card.to_dict()
    assert d["name"] == "MyAgent"
    assert d["capabilities"]["streaming"] is True
    assert len(d["skills"]) == 1
    assert d["skills"][0]["id"] == "s1"
    assert "/.well-known" not in str(d)


def test_agent_card_no_auth():
    card = AgentCard(name="X", description="Y", url="Z", auth_scheme="none")
    d = card.to_dict()
    assert "securitySchemes" not in d
    assert "security" not in d


def test_agent_card_bearer_auth():
    card = AgentCard(name="X", description="Y", url="Z", auth_scheme="bearer")
    d = card.to_dict()
    assert "bearer" in d["securitySchemes"]
    assert d["security"] == [{"bearer": []}]


def test_agent_card_apikey_auth():
    card = AgentCard(name="X", description="Y", url="Z", auth_scheme="apiKey")
    d = card.to_dict()
    assert "apiKey" in d["securitySchemes"]
    assert d["security"] == [{"apiKey": []}]


def test_agent_card_provider_present():
    card = AgentCard(
        name="X",
        description="Y",
        url="Z",
        org="Acme",
        org_url="https://acme.com",
    )
    d = card.to_dict()
    assert d["provider"]["organization"] == "Acme"
    assert d["provider"]["url"] == "https://acme.com"


def test_agent_card_provider_absent():
    card = AgentCard(name="X", description="Y", url="Z")
    d = card.to_dict()
    assert "provider" not in d


def test_state_transition_history_flag():
    card = AgentCard(
        name="X",
        description="Y",
        url="Z",
        state_transition_history=True,
    )
    d = card.to_dict()
    assert d["capabilities"]["stateTransitionHistory"] is True


def test_default_io_modes():
    card = AgentCard(name="X", description="Y", url="Z")
    d = card.to_dict()
    assert "text/plain" in d["defaultInputModes"]
    assert "application/json" in d["defaultOutputModes"]


# ---------------------------------------------------------------------------
# serve_a2a import error
# ---------------------------------------------------------------------------


def test_serve_a2a_import_error(simple_graph, monkeypatch):
    card = AgentCard(name="X", description="Y", url="Z")
    simple_graph.with_agent_card(card)
    monkeypatch.setitem(sys.modules, "langgraph.a2a.server", None)
    with pytest.raises(ImportError, match=r"pip install langgraph\[a2a\]"):
        simple_graph.serve_a2a()
