"""Messages in an append-only log, and what happens to existing graphs.

Run from `libs/langgraph`:

    uv run python examples/messages_log_demo.py

1. Graphs on `add_messages` don't change and keep working. The four message
   edit ops (`RemoveMessage`, `REMOVE_ALL_MESSAGES`, replace by id,
   `Overwrite`) are deprecated and warn.
2. On `MessagesLog` the same edits are errors: stored history is append-only.
3. The same bot without edits: history only grows, and the bot builds the
   messages it sees (a summary, a window) when it's called.
4. An `add_messages` thread switching to `MessagesLog` keeps its history.
5. Reading messages a page at a time, without the graph.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.log import MessagesLog, read_log
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import Overwrite
from langgraph.warnings import LangGraphDeprecationWarning


class OldState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class LogState(TypedDict):
    messages: Annotated[list[AnyMessage], MessagesLog()]


def build(state_schema: type, bot: Any, saver: InMemorySaver):
    return (
        StateGraph(state_schema)
        .add_node("bot", bot)
        .add_edge(START, "bot")
        .add_edge("bot", END)
        .compile(checkpointer=saver)
    )


def say(app, thread: dict, text: str, id: str) -> None:
    app.invoke({"messages": [HumanMessage(text, id=id)]}, thread)


def contents(messages: list[AnyMessage]) -> str:
    return " | ".join(str(m.content) for m in messages)


def heading(text: str) -> None:
    print(f"\n== {text}")


saver = InMemorySaver()


# -- today's bot: edits stored history -----------------------------------------


def editing_bot(state: dict) -> dict:
    messages = state["messages"]
    last = messages[-1]
    if last.content == "/compact":
        # summarize: replace everything with a summary and the last exchange
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                SystemMessage(f"summary of {len(messages) - 1}", id=f"sum-{last.id}"),
                *messages[-3:-1],
            ]
        }
    update: list[Any] = [AIMessage(f"echo: {last.content}", id=f"ai-{last.id}")]
    if len(messages) > 6:
        # trim the oldest message
        update.append(RemoveMessage(id=messages[0].id))
    return {"messages": update}


# -- 1 -------------------------------------------------------------------------

heading("1. add_messages graphs keep working; message edit ops warn")
old_app = build(OldState, editing_bot, saver)
old_thread = {"configurable": {"thread_id": "add_messages"}}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", LangGraphDeprecationWarning)
    say(old_app, old_thread, "hi", "h1")
    say(old_app, old_thread, "my name is Bob", "h2")
    # correct a message by id
    old_app.update_state(
        old_thread, {"messages": [HumanMessage("my name is Rob", id="h2")]}
    )
    say(old_app, old_thread, "/compact", "h3")
    for i in range(4, 8):  # long enough for the bot to trim
        say(old_app, old_thread, f"turn {i}", f"h{i}")
    keep = old_app.get_state(old_thread).values["messages"][-3:]
    old_app.update_state(old_thread, {"messages": Overwrite(keep)})
for text in dict.fromkeys(str(w.message).split(" is deprecated")[0] for w in caught):
    print(f"  warns: {text} is deprecated ...")
print(f"  e.g.: {caught[0].message}")
print(f"  messages: {contents(old_app.get_state(old_thread).values['messages'])}")


# -- 2 -------------------------------------------------------------------------

heading("2. On MessagesLog the same edits are errors")
edit_on_log = build(LogState, editing_bot, saver)
attempts = {
    "replace by id": lambda t: (
        say(edit_on_log, t, "my name is Bob", "h2"),
        edit_on_log.update_state(
            t, {"messages": [HumanMessage("my name is Rob", id="h2")]}
        ),
    ),
    "REMOVE_ALL_MESSAGES": lambda t: say(edit_on_log, t, "/compact", "h3"),
    "Overwrite": lambda t: (
        say(edit_on_log, t, "hi", "h1"),
        edit_on_log.update_state(t, {"messages": Overwrite([])}),
    ),
}
for label, attempt in attempts.items():
    try:
        attempt({"configurable": {"thread_id": f"log-{label}"}})
        print(f"  {label}: accepted (unexpected)")
    except InvalidUpdateError as e:
        print(f"  {label}: {str(e).splitlines()[0]}")


# -- 3 -------------------------------------------------------------------------


def model_view(messages: list[AnyMessage], window: int = 6) -> list[AnyMessage]:
    """What the bot sees: the latest summary, then the newest messages.

    Built from the history when the bot is called; the history never changes.
    """
    summaries = [i for i, m in enumerate(messages) if str(m.id).startswith("sum-")]
    if not summaries:
        return messages[-window:]
    last = summaries[-1]
    return [messages[last], *messages[last + 1 :][-window:]]


def appending_bot(state: dict) -> dict:
    messages = state["messages"]
    last = messages[-1]
    if last.content == "/compact":
        # record the summary; don't delete what it summarizes
        summary = SystemMessage(f"summary of {len(messages) - 1}", id=f"sum-{last.id}")
        return {"messages": [summary]}
    view = model_view(messages)  # what would be sent to the model
    return {
        "messages": [
            AIMessage(f"echo: {last.content} (saw {len(view)})", id=f"ai-{last.id}")
        ]
    }


heading("3. The same bot without edits: history grows, the bot builds its view")
log_app = build(LogState, appending_bot, saver)
log_thread = {"configurable": {"thread_id": "MessagesLog"}}
say(log_app, log_thread, "hi", "h1")
say(log_app, log_thread, "my name is Bob", "h2")
# a correction is a new message, not an edit of an old one
say(log_app, log_thread, "sorry, my name is Rob", "h2b")
say(log_app, log_thread, "/compact", "h3")
for i in range(4, 8):
    say(log_app, log_thread, f"turn {i}", f"h{i}")
history = log_app.get_state(log_thread).values["messages"]
print(f"  stored history ({len(history)}): {contents(history)}")
print(
    f"  what the bot sees ({len(model_view(history))}): {contents(model_view(history))}"
)


# -- 4 -------------------------------------------------------------------------

heading("4. An add_messages thread switching to MessagesLog")
# the thread from section 1, written by the editing bot on add_messages
before = old_app.get_state(old_thread)
print(f"  on add_messages: {contents(before.values['messages'])}")
switched = build(LogState, appending_bot, saver)
say(switched, old_thread, "still there?", "s1")
print(
    f"  next turn on MessagesLog: {contents(switched.get_state(old_thread).values['messages'])}"
)
print(
    "  edits made before the switch are already in the stored list; after it, "
    "the graph code must not edit (section 2)"
)


# -- 5 -------------------------------------------------------------------------

heading("5. Reading a page at a time, with only the checkpointer")
page = read_log(saver, log_thread, limit=3)
print(f"  last 3 of {page.length}:")
for i, m in page.entries:
    print(f"    [{i}] {m.type}: {m.content}")
page = read_log(saver, log_thread, before=page.entries[0][0], limit=2)
print("  the 2 before those:")
for i, m in page.entries:
    print(f"    [{i}] {m.type}: {m.content}")
