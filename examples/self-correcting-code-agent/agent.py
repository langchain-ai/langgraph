"""
Self-Correcting Code Agent using LangGraph

A production-ready pattern where an LLM writes code, executes it in a sandbox,
reads the error output, and iteratively fixes it — with a human approval gate
before final execution. Demonstrates:

  - LangGraph state + cyclic graph (write → execute → fix → loop)
  - Human-in-the-loop (interrupt before execution)
  - Conditional edges based on execution result
  - Max-retry guard to prevent infinite loops
  - Structured state with TypedDict

Ideal for: coding assistants, data pipeline generators, test generators.
"""

from __future__ import annotations

import ast
import io
import sys
import textwrap
import traceback
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MAX_RETRIES = 4


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
    code: str                                 # latest generated code
    execution_output: str                     # stdout from last run
    execution_error: str                      # stderr / exception from last run
    retry_count: int                          # how many fix attempts so far
    task: str                                 # original task description
    approved: bool                            # human approved the code?


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_WRITE = SystemMessage(content=textwrap.dedent("""\
    You are an expert Python programmer.
    When given a task, output ONLY a valid Python code block — no prose, no markdown fences.
    The code must be self-contained and print its result to stdout.
"""))

SYSTEM_FIX = SystemMessage(content=textwrap.dedent("""\
    You are an expert Python debugger.
    You will be given code that raised an error and the full traceback.
    Output ONLY the corrected Python code — no prose, no markdown fences.
    Preserve the original intent of the code.
"""))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def write_code(state: AgentState) -> AgentState:
    """Generate initial code from the task description."""
    response = llm.invoke([
        SYSTEM_WRITE,
        HumanMessage(content=f"Task: {state['task']}"),
    ])
    code = _extract_code(response.content)
    return {
        "messages": [AIMessage(content=f"```python\n{code}\n```")],
        "code": code,
        "execution_output": "",
        "execution_error": "",
    }


def human_approval(state: AgentState) -> AgentState:
    """
    Pause graph and show the code to a human.
    Resume by calling graph.invoke(..., Command(resume=True/False)).
    """
    approved = interrupt({
        "question": "Approve this code for execution?",
        "code": state["code"],
        "retry_count": state["retry_count"],
    })
    return {"approved": approved}


def execute_code(state: AgentState) -> AgentState:
    """Run the code in a restricted in-process sandbox and capture output."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    error_msg = ""
    try:
        # Compile first so syntax errors surface cleanly
        compiled = compile(state["code"], "<agent_code>", "exec")
        exec(compiled, {})  # noqa: S102  (intentional sandbox exec)
    except Exception:
        error_msg = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    output = stdout_buf.getvalue()
    stderr_out = stderr_buf.getvalue()
    combined_error = "\n".join(filter(None, [stderr_out, error_msg]))

    return {
        "execution_output": output,
        "execution_error": combined_error,
        "messages": [AIMessage(content=_format_execution_result(output, combined_error))],
    }


def fix_code(state: AgentState) -> AgentState:
    """Ask the LLM to fix the code given the error traceback."""
    response = llm.invoke([
        SYSTEM_FIX,
        HumanMessage(content=textwrap.dedent(f"""\
            Code:
            {state['code']}

            Error:
            {state['execution_error']}

            Fix the code.
        """)),
    ])
    fixed_code = _extract_code(response.content)
    return {
        "messages": [AIMessage(content=f"Fixed (attempt {state['retry_count'] + 1}):\n```python\n{fixed_code}\n```")],
        "code": fixed_code,
        "retry_count": state["retry_count"] + 1,
        "execution_output": "",
        "execution_error": "",
    }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_approval(state: AgentState) -> Literal["execute_code", END]:
    if not state.get("approved", False):
        return END
    return "execute_code"


def route_after_execution(state: AgentState) -> Literal["fix_code", END]:
    if state["execution_error"]:
        if state["retry_count"] >= MAX_RETRIES:
            return END  # give up after MAX_RETRIES attempts
        return "fix_code"
    return END  # success


def route_after_fix(state: AgentState) -> Literal["human_approval", END]:
    if state["retry_count"] >= MAX_RETRIES:
        return END
    return "human_approval"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("write_code", write_code)
    builder.add_node("human_approval", human_approval)
    builder.add_node("execute_code", execute_code)
    builder.add_node("fix_code", fix_code)

    builder.add_edge(START, "write_code")
    builder.add_edge("write_code", "human_approval")
    builder.add_conditional_edges("human_approval", route_after_approval)
    builder.add_conditional_edges("execute_code", route_after_execution)
    builder.add_conditional_edges("fix_code", route_after_fix)

    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["human_approval"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Strip markdown fences if the LLM wrapped the code anyway."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence line
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner).strip()
    return text


def _format_execution_result(output: str, error: str) -> str:
    if error:
        return f"Execution failed:\n{error}"
    return f"Execution succeeded. Output:\n{output or '(no output)'}"


# ---------------------------------------------------------------------------
# Quick demo — run `python agent.py` to see the graph in action
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langgraph.types import Command

    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-1"}}

    task = (
        "Write a Python function that returns the first N Fibonacci numbers "
        "and prints them. Call it with N=10."
    )

    print("=== STEP 1: Generate code ===")
    # Run until the graph hits the interrupt (human_approval node)
    for event in graph.stream(
        {"task": task, "retry_count": 0, "approved": False},
        config,
        stream_mode="values",
    ):
        if "code" in event and event["code"]:
            print("\nGenerated code:\n")
            print(event["code"])

    print("\n=== STEP 2: Human approves ===")
    # Resume with approval = True
    for event in graph.stream(
        Command(resume=True),
        config,
        stream_mode="values",
    ):
        if "execution_output" in event and event["execution_output"]:
            print("\nExecution output:", event["execution_output"])
        if "execution_error" in event and event["execution_error"]:
            print("\nExecution error:", event["execution_error"])

    final = graph.get_state(config)
    print("\n=== FINAL STATE ===")
    print("Output:", final.values.get("execution_output", "(none)"))
    print("Error:", final.values.get("execution_error", "(none)"))
    print("Retries used:", final.values.get("retry_count", 0))
