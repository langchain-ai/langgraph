import os
import re
import sqlite3
from datetime import datetime
from typing import Annotated, Optional

import numpy as np
import openai
import pandas as pd
import pytest
import requests
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ensure_config
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import unit

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
db = local_file
if not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)

# Convert the flights to present time for our tutorial
conn = sqlite3.connect(local_file)
cursor = conn.cursor()

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';", conn
).name.tolist()
tdf = {}
for t in tables:
    tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

example_time = pd.to_datetime(
    tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
).max()
current_time = pd.to_datetime("now").tz_localize(example_time.tz)
time_diff = current_time - example_time

tdf["bookings"]["book_date"] = (
    pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT)) + time_diff
)

datetime_columns = [
    "scheduled_departure",
    "scheduled_arrival",
    "actual_departure",
    "actual_arrival",
]
for column in datetime_columns:
    tdf["flights"][column] = (
        pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
    )

for table_name, df in tdf.items():
    df.to_sql(table_name, conn, if_exists="replace", index=False)
conn.commit()
conn.close()


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    config = ensure_config()
    configuration = config.get("configurable", {})
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)

    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)

    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)

    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)
    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def fetch_user_flight_information() -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def update_ticket_to_new_flight(ticket_no: str, new_flight_id: int) -> str:
    """Update the user's ticket to a new valid flight."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "Invalid new flight ID provided."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    cursor.execute(
        "SELECT departure_airport, arrival_airport FROM flights WHERE flight_id = ?",
        (current_flight[0],),
    )
    current_airports = cursor.fetchone()
    if new_flight != current_airports:
        cursor.close()
        conn.close()
        return "New flight does not match the original departure and arrival airports."

    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully updated to new flight."


response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


tools = [TavilySearchResults(max_results=1), search_flights, lookup_policy]
sensitive_tools = [update_ticket_to_new_flight]
sensitive_tool_names = {t.name for t in sensitive_tools}

llm = ChatAnthropic(model="claude-3-haiku-20240307")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Where appropriate, use the provided tools to assist the user in managing their flights."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>",
        ),
        ("placeholder", "{messages}"),
    ]
)

assistant_runnable = prompt | llm.bind_tools(tools + sensitive_tools)


def assistant(state: State):
    return {"messages": assistant_runnable.invoke(state)}


builder = StateGraph(State)
builder.add_node("fetch_user_info", user_info)
builder.set_entry_point("fetch_user_info")
builder.add_node("assistant", assistant)
builder.add_node("safe_tools", ToolNode(tools))
builder.add_node("sensitive_tools", ToolNode(tools + sensitive_tools))
builder.add_edge("sensitive_tools", "fetch_user_info")
builder.add_edge("fetch_user_info", "assistant")
# Fetch the user info again in case it was updated
builder.add_edge("safe_tools", "fetch_user_info")


def route_tools(state: State):
    selected = tools_condition(state)
    if selected == "action":
        # Check if we need to get user confirmation
        ai_message = state["messages"][-1]
        tools_called = [t["name"] for t in ai_message.tool_calls]
        if any(name in sensitive_tool_names for name in tools_called):
            return "sensitive_tools"
    return selected


builder.add_conditional_edges(
    "assistant",
    route_tools,
    {"action": "safe_tools", END: END, "sensitive_tools": "sensitive_tools"},
)
checkpointer = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(interrupt_before=["sensitive_tools"], checkpointer=checkpointer)


questions = [
    ["Hi there, what time is my flight?"],
    [
        "Am i allowed to update my flight to something sooner? I want to leave later today.",
        "Okkk, what options are there for me to change my flight?",
        "Update my flight to something later actually.",
    ],
]


@pytest.fixture
def passenger_id():
    return "3442 587242"

@unit
@pytest.mark.parametrize("thread_num, questions", enumerate(questions))
def test_customer_support_bot(thread_num: int, questions: list[str], passenger_id: str):
    config = {
        "configurable": {
            "passenger_id": passenger_id,
            "thread_id": f"test-thread-id-{thread_num}",
        }
    }
    for question in questions:
        results = graph.invoke(
            {"messages": [("user", question)]},
            config,
        )
    return results
