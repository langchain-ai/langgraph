import hashlib
import hmac
import logging
import operator
import os
import secrets
import time as _time
import uuid
from datetime import datetime, timezone
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.types import Send

logger = logging.getLogger(__name__)

_AUDIT_LOG: list[dict] = []

_SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
_AGENT_SECRET = os.environ.get("AGENT_SECRET", secrets.token_hex(32))
_SESSION_TTL = int(os.environ.get("SESSION_TTL", "3600"))


def _sign_token(token: str) -> str:
    return hmac.new(_SESSION_SECRET.encode(), token.encode(), hashlib.sha256).hexdigest()


def _create_session_token(principal: str = "bench") -> dict:
    token_id = str(uuid.uuid4())
    issued_at = int(_time.time())
    expires_at = issued_at + _SESSION_TTL
    payload = f"{token_id}:{principal}:{issued_at}:{expires_at}"
    signature = _sign_token(payload)
    return {
        "token_id": token_id,
        "principal": principal,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "signature": signature,
        "payload": payload,
    }


def _verify_session_token(session: dict) -> bool:
    payload = session.get("payload", "")
    expected_sig = _sign_token(payload)
    if not hmac.compare_digest(expected_sig, session.get("signature", "")):
        raise ValueError("Session token signature verification failed.")
    if int(_time.time()) > session.get("expires_at", 0):
        raise ValueError("Session token has expired.")
    return True


def _sign_agent_message(node: str, payload: dict) -> str:
    msg = f"{node}:{sorted(payload.items())}"
    return hmac.new(_AGENT_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()


def _verify_agent_message(node: str, payload: dict, signature: str) -> bool:
    expected = _sign_agent_message(node, payload)
    if not hmac.compare_digest(expected, signature):
        raise ValueError(f"Inter-agent message authentication failed for node '{node}'.")
    return True


def _audit(event: str, node: str, input_data: object, output_data: object, principal: str, trace_id: str) -> None:
    input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()
    output_hash = hashlib.sha256(str(output_data).encode()).hexdigest()
    record = {
        "event": event,
        "node": node,
        "principal": principal,
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": input_hash,
        "output_hash": output_hash,
    }
    _AUDIT_LOG.append(record)
    logger.info("AUDIT: %s", record)


def fanout_to_subgraph() -> StateGraph:
    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    async def continue_to_jokes(state: OverallState):
        sends = []
        for s in state["subjects"]:
            payload = {"subject": s}
            sig = _sign_agent_message("generate_joke", payload)
            payload["_agent_sig"] = sig
            sends.append(Send("generate_joke", payload))
        return sends

    class JokeInput(TypedDict):
        subject: str

    class JokeOutput(TypedDict):
        jokes: list[str]

    class JokeState(JokeInput, JokeOutput): ...

    async def bump(state: JokeOutput):
        _audit("node_execute", "bump", state, None, "system", "n/a")
        result = {"jokes": [state["jokes"][0] + " a"]}
        _audit("node_complete", "bump", state, result, "system", "n/a")
        return result

    async def generate(state: JokeInput):
        _audit("node_execute", "generate", state, None, "system", "n/a")
        result = {"jokes": [f"Joke about {state['subject']}"]}
        _audit("node_complete", "generate", state, result, "system", "n/a")
        return result

    async def edit(state: JokeInput):
        sig = state.get("_agent_sig")
        verify_payload = {k: v for k, v in state.items() if k != "_agent_sig"}
        if sig:
            _verify_agent_message("generate_joke", verify_payload, sig)
        _audit("node_execute", "edit", state, None, "system", "n/a")
        subject = state["subject"]
        result = {"subject": f"{subject} - hohoho"}
        _audit("node_complete", "edit", state, result, "system", "n/a")
        return result

    async def bump_loop(state: JokeOutput):
        _audit("node_execute", "bump_loop", state, None, "system", "n/a")
        decision = END if state["jokes"][0].endswith(" a" * 10) else "bump"
        _audit("node_complete", "bump_loop", state, {"decision": decision}, "system", "n/a")
        return decision

    # subgraph
    subgraph = StateGraph(JokeState, input_schema=JokeInput, output_schema=JokeOutput)
    subgraph.add_node("edit", edit)
    subgraph.add_node("generate", generate)
    subgraph.add_node("bump", bump)
    subgraph.set_entry_point("edit")
    subgraph.add_edge("edit", "generate")
    subgraph.add_edge("generate", "bump")
    subgraph.add_conditional_edges("bump", bump_loop)
    subgraph.set_finish_point("generate")
    subgraphc = subgraph.compile()

    # parent graph
    builder = StateGraph(OverallState)
    builder.add_node("generate_joke", subgraphc)
    builder.add_conditional_edges(START, continue_to_jokes)
    builder.add_edge("generate_joke", END)

    return builder


def fanout_to_subgraph_sync() -> StateGraph:
    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    def continue_to_jokes(state: OverallState):
        sends = []
        for s in state["subjects"]:
            payload = {"subject": s}
            sig = _sign_agent_message("generate_joke", payload)
            payload["_agent_sig"] = sig
            sends.append(Send("generate_joke", payload))
        return sends

    class JokeInput(TypedDict):
        subject: str

    class JokeOutput(TypedDict):
        jokes: list[str]

    class JokeState(JokeInput, JokeOutput): ...

    def bump(state: JokeOutput):
        _audit("node_execute", "bump", state, None, "system", "n/a")
        result = {"jokes": [state["jokes"][0] + " a"]}
        _audit("node_complete", "bump", state, result, "system", "n/a")
        return result

    def generate(state: JokeInput):
        _audit("node_execute", "generate", state, None, "system", "n/a")
        result = {"jokes": [f"Joke about {state['subject']}"]}
        _audit("node_complete", "generate", state, result, "system", "n/a")
        return result

    def edit(state: JokeInput):
        sig = state.get("_agent_sig")
        verify_payload = {k: v for k, v in state.items() if k != "_agent_sig"}
        if sig:
            _verify_agent_message("generate_joke", verify_payload, sig)
        _audit("node_execute", "edit", state, None, "system", "n/a")
        subject = state["subject"]
        result = {"subject": f"{subject} - hohoho"}
        _audit("node_complete", "edit", state, result, "system", "n/a")
        return result

    def bump_loop(state: JokeOutput):
        _audit("node_execute", "bump_loop", state, None, "system", "n/a")
        decision = END if state["jokes"][0].endswith(" a" * 10) else "bump"
        _audit("node_complete", "bump_loop", state, {"decision": decision}, "system", "n/a")
        return decision

    # subgraph
    subgraph = StateGraph(JokeState, input_schema=JokeInput, output_schema=JokeOutput)
    subgraph.add_node("edit", edit)
    subgraph.add_node("generate", generate)
    subgraph.add_node("bump", bump)
    subgraph.set_entry_point("edit")
    subgraph.add_edge("edit", "generate")
    subgraph.add_edge("generate", "bump")
    subgraph.add_conditional_edges("bump", bump_loop)
    subgraph.set_finish_point("generate")
    subgraphc = subgraph.compile()

    # parent graph
    builder = StateGraph(OverallState)
    builder.add_node("generate_joke", subgraphc)
    builder.add_conditional_edges(START, continue_to_jokes)
    builder.add_edge("generate_joke", END)

    return builder


if __name__ == "__main__":
    import asyncio
    import random

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    session = _create_session_token(principal="bench")
    _verify_session_token(session)
    trace_id = str(uuid.uuid4())
    thread_id = secrets.token_urlsafe(32)

    logger.info(
        "AUDIT: workflow_start trace_id=%s thread_id=%s principal=%s session_token_id=%s",
        trace_id,
        thread_id,
        session["principal"],
        session["token_id"],
    )

    graph = fanout_to_subgraph().compile(checkpointer=InMemorySaver())
    input = {
        "subjects": [
            random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(1000)
        ]
    }
    config = {"configurable": {"thread_id": thread_id}}

    async def run():
        count = len([c async for c in graph.astream(input, config=config)])
        _audit(
            "workflow_complete",
            "fanout_to_subgraph",
            {"input_subjects_count": len(input["subjects"])},
            {"stream_events": count},
            session["principal"],
            trace_id,
        )
        return count

    uvloop.install()
    start = _time.time()
    asyncio.run(run())
    end = _time.time()
    logger.info(
        "AUDIT: workflow_end trace_id=%s thread_id=%s elapsed=%.4f",
        trace_id,
        thread_id,
        end - start,
    )
    print(f"Time taken: {end - start:.4f} seconds")