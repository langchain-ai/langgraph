# Build a Customer Support Bot

Customer support bots can free up teams' time by handling routine issues, but it can be hard to build a bot that reliably handles diverse tasks in a way that doesn't leave the user pulling their hair out.

In this tutorial, you will build a customer support bot for an airline to help users research and make travel arrangements. You'll learn to use LangGraph's interrupts and checkpointers and more complex state to organize your assistant's tools and manage a user's flight bookings, hotel reservations, car rentals, and excursions. It assumes you are familiar with the concepts presented in the [LangGraph introductory tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/).

By the end, you'll have built a working bot and gained an understanding of  LangGraph's key concepts and architectures. You'll be able to apply these design patterns to your other AI projects.

Your final chat bot will look something like the following diagram:

<img src="../img/part-4-diagram.png" src="../img/part-4-diagram.png">

Let's start!

## Prerequisites

First, set up your environment. We'll install this tutorial's prerequisites, download the test DB, and define the tools we will reuse in each section.

We'll be using Claude as our LLM and define a number of custom tools. While most of our tools will connect to a local sqlite database (and require no additional dependencies), we will also provide a general web search to the agent using Tavily.


```
%%capture --no-stderr
%pip install -U langgraph langchain-community langchain-anthropic tavily-python pandas openai
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

#### Populate the database

Run the next script to fetch a `sqlite` DB we've prepared for this tutorial and update it to look like it's current. The details are unimportant.


```python
import os
import shutil
import sqlite3

import pandas as pd
import requests

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
# The backup lets us restart for each tutorial section
backup_file = "travel2.backup.sqlite"
overwrite = False
if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(local_file, backup_file)


# Convert the flights to present time for our tutorial
def update_dates(file):
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
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
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
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
    del df
    del tdf
    conn.commit()
    conn.close()

    return file


db = update_dates(local_file)
```

## Tools

Next, define our assistant's tools to search the airline's policy manual and search and manage reservations for flights, hotels, car rentals, and excursions. We will reuse these tools throughout the tutorial. The exact implementations
aren't important, so feel free to run the code below and jump to [Part 1](#part-1-zero-shot.md).

#### Lookup Company Policies

The assistant retrieve policy information to answer user questions. Note that _enforcement_ of these policies still must be done within the tools/APIs themselves, since the LLM can always ignore this.


```python
import re

import numpy as np
import openai
from langchain_core.tools import tool

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
```

#### Flights

Define the (`fetch_user_flight_information`) tool to let the agent see the current user's flight information.  Then define tools to search for flights and manage the passenger's bookings stored in the SQL database.

We then can [access the RunnableConfig](https://python.langchain.com/docs/how_to/tool_configure/#inferring-by-parameter-type) for a given run to check the `passenger_id` of the user accessing this application. The LLM never has to provide these explicitly, they are provided for a given invocation of the graph so that each user cannot access other passengers' booking information.

<div class="admonition warning">
    <p class="admonition-title">Compatibility</p>
    <p>
        This tutorial expects `langchain-core>=0.2.16` to use the injected RunnableConfig. Prior to that, you'd use `ensure_config` to collect the config from context.
    </p>
</div> 



```python
import sqlite3
from datetime import date, datetime
from typing import Optional

import pytz
from langchain_core.runnables import RunnableConfig


@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
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
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
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
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "Invalid new flight ID provided."
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    # In a real application, you'd likely add additional checks here to enforce business logic,
    # like "does the new departure airport match the current ticket", etc.
    # While it's best to try to be *proactive* in 'type-hinting' policies to the LLM
    # it's inevitably going to get things wrong, so you **also** need to ensure your
    # API enforces valid behavior
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully updated to new flight."


@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully cancelled."
```

#### Car Rental Tools

Once a user books a flight, they likely will want to organize transportation. Define some "car rental" tools to let the user search for and reserve a car at their destination.


```python
from datetime import date, datetime
from typing import Optional, Union


@tool
def search_car_rentals(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for car rentals based on location, name, price tier, start date, and end date.

    Args:
        location (Optional[str]): The location of the car rental. Defaults to None.
        name (Optional[str]): The name of the car rental company. Defaults to None.
        price_tier (Optional[str]): The price tier of the car rental. Defaults to None.
        start_date (Optional[Union[datetime, date]]): The start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The end date of the car rental. Defaults to None.

    Returns:
        list[dict]: A list of car rental dictionaries matching the search criteria.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM car_rentals WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # For our tutorial, we will let you match on any dates and price tier.
    # (since our toy dataset doesn't have much data)
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_car_rental(rental_id: int) -> str:
    """
    Book a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to book.

    Returns:
        str: A message indicating whether the car rental was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully booked."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."


@tool
def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a car rental's start and end dates by its ID.

    Args:
        rental_id (int): The ID of the car rental to update.
        start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

    Returns:
        str: A message indicating whether the car rental was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if start_date:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_date, rental_id),
        )
    if end_date:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?", (end_date, rental_id)
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully updated."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."


@tool
def cancel_car_rental(rental_id: int) -> str:
    """
    Cancel a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to cancel.

    Returns:
        str: A message indicating whether the car rental was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully cancelled."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."
```

#### Hotels

The user has to sleep! Define some tools to search for and manage hotel reservations.


```python
@tool
def search_hotels(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for hotels based on location, name, price tier, check-in date, and check-out date.

    Args:
        location (Optional[str]): The location of the hotel. Defaults to None.
        name (Optional[str]): The name of the hotel. Defaults to None.
        price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
        checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

    Returns:
        list[dict]: A list of hotel dictionaries matching the search criteria.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM hotels WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # For the sake of this tutorial, we will let you match on any dates and price tier.
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_hotel(hotel_id: int) -> str:
    """
    Book a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to book.

    Returns:
        str: A message indicating whether the hotel was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully booked."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."


@tool
def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a hotel's check-in and check-out dates by its ID.

    Args:
        hotel_id (int): The ID of the hotel to update.
        checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

    Returns:
        str: A message indicating whether the hotel was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if checkin_date:
        cursor.execute(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?", (checkin_date, hotel_id)
        )
    if checkout_date:
        cursor.execute(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (checkout_date, hotel_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully updated."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."


@tool
def cancel_hotel(hotel_id: int) -> str:
    """
    Cancel a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to cancel.

    Returns:
        str: A message indicating whether the hotel was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully cancelled."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."
```

#### Excursions

Finally, define some tools to let the user search for things to do (and make reservations) once they arrive.


```python
@tool
def search_trip_recommendations(
    location: Optional[str] = None,
    name: Optional[str] = None,
    keywords: Optional[str] = None,
) -> list[dict]:
    """
    Search for trip recommendations based on location, name, and keywords.

    Args:
        location (Optional[str]): The location of the trip recommendation. Defaults to None.
        name (Optional[str]): The name of the trip recommendation. Defaults to None.
        keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

    Returns:
        list[dict]: A list of trip recommendation dictionaries matching the search criteria.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if keywords:
        keyword_list = keywords.split(",")
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_excursion(recommendation_id: int) -> str:
    """
    Book a excursion by its recommendation ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to book.

    Returns:
        str: A message indicating whether the trip recommendation was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully booked."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."


@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
    Update a trip recommendation's details by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to update.
        details (str): The new details of the trip recommendation.

    Returns:
        str: A message indicating whether the trip recommendation was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully updated."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."


@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
    Cancel a trip recommendation by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to cancel.

    Returns:
        str: A message indicating whether the trip recommendation was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully cancelled."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."
```

#### Utilities

Define helper functions to pretty print the messages in the graph while we debug it and to give our tool node error handling (by adding the error to the chat history).


```python
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
```

## Part 1: Zero-shot Agent

When building, it's best to start with the simplest working implementation and use an [evaluation tool like LangSmith](https://docs.smith.langchain.com/evaluation) to measure its efficacy. All else equal, prefer simple, scalable solutions to complicated ones. In this case, the single-graph approach has limitations. The bot may take undesired actions without user confirmation, struggle with complex queries, and lack focus in its responses. We'll address these issues later. 

In this section, we will define a simple Zero-shot agent as the assistant, give the agent **all** of our tools, and prompt it to use them judiciously to assist the user.

The simple 2-node graph will look like the following:

<img src="../img/part-1-diagram.png" src="../img/part-1-diagram.png">

Start by defining the state.

#### State

Define our `StateGraph`'s state as a typed dictionary containing an append-only list of messages. These messages form the chat history, which is all the state our simple assistant needs.


```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

#### Agent

Next, define the assistant function. This function takes the graph state, formats it into a prompt, and then calls an LLM for it to predict the best response.


```python
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Haiku is faster and cheaper, but less accurate
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# You could swap LLMs, though you will likely want to update the prompts when
# doing so!
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4-turbo-preview")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

part_1_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)
```

#### Define Graph

Now, create the graph. The graph is the final assistant for this section.


```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)
```


```python
from IPython.display import Image, display

try:
    display(Image(part_1_graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAD5ANYDASIAAhEBAxEB/8QAHQABAAIDAQEBAQAAAAAAAAAAAAUGAwQHCAECCf/EAFEQAAEEAQIDAgYLDAcGBwAAAAEAAgMEBQYRBxIhEzEVFiJBUZQIFBcyVVZhdNHS0yM1NlRxdYGRk5WytCU3QkNSgpIYJGRylqEzNFNiscHw/8QAGwEBAQADAQEBAAAAAAAAAAAAAAECAwUEBgf/xAAzEQEAAQIBCQUJAQADAAAAAAAAAQIRAwQSITFBUVKR0RQzYXGhBRMVI2KSscHhgSLw8f/aAAwDAQACEQMRAD8A/qmiIgIiICIiAsNq5XpR89ieOuz/ABSvDR+sqDu37uevz47FTGlVrnkt5NrQ5zX/APpQhwLS4d7nuBa3cNAc4u5Ptbh/p+F5llxcF+ydua1fb7ZmcR5y9+5/V0W+KKae8n/IW29u+NWF+F6HrLPpTxqwvwxQ9ZZ9KeKuF+B6HqzPoTxVwvwPQ9WZ9CvyfH0XQeNWF+GKHrLPpTxqwvwxQ9ZZ9KeKuF+B6HqzPoTxVwvwPQ9WZ9CfJ8fQ0HjVhfhih6yz6U8asL8MUPWWfSnirhfgeh6sz6E8VcL8D0PVmfQnyfH0NB41YX4Yoess+lblTIVb7S6rZhstHeYZA4D9S0/FXC/A9D1Zn0LUtaB05bkErsNThnad22K0QhmafkkZs4foKfJnbPp/E0J9FWI7NzSM8MN+1NksPK4RsvT8va1XE7NbKQAHMPQB+24O3NvuXCzrXXRm+MEwIiLWgiIgIiICIiAiIgIiICIiAojV2Yfp/S+VyMQDpq1Z8kTXdxft5IP6dlLqvcQqct7ROZjhaZJm13SsY0blzmeWAB6SW7LbgxE4lMVarwsa0hp/Dx4DDVKEZ5uxZ5cnnkkJ3e8/K5xc4n0kqRWGnaivVILMDueGZjZGO9LSNwf1FZlhVMzVM1a0FUuIHFbS3C6LHv1JkzSfkJHRVIIa01madzW8z+SKFj3kNHUnbYbjchW1cU9krQqPg07k48frBupMc+zJiM5o7HG7NQldG0OZNEA4Ojl6Atc0tPL1LehWI2cp7JjT+N4q6b0m2tetUc3hfC8OTq463ODzyQthaGxwu8lzZHOdISAzZodylwVgtcftBUdct0hZz3tfOvtNotilpzthNhw3bCJzH2XaHcbN59zuBsuUx5fWendd8Ltfax0nlrtuxpGzicxDp6g+4+neklrTDnij3LWu7J43G4aehPnVA4t4/Wep5tTDMYbX+W1Bj9VwW8fUxsEwwsOJguRSRyRtjIjsSGJpJGz5ec9GgDoHpi3x20TT1je0ocpYsahozR17VCnjbVh8DpI2yMLzHE4NYWvb5ZPLuSN9wQIvgLx7xvHPBWblWjdx1yvYsxyV56VlkYjZYkijc2aSJjHuc1gc5jSSwktcAQtbhLp+7jOMXGnJWsbYqQZLLY91W3NA5jbUbMdA0ljiNnta/nb03APMO/dRfsY7GQ0vh8poTMaezWNyWLymUte3rFF7aFmGW9JLG6GxtyPLmzNPKDuOV24GyDuCIiDXyFCvlaFmlbibPVsxuhlif3PY4bOB/KCVEaGvz39Nwi1L29upLNRmlO+8j4ZXRF53/wAXJzfpU+qzw8b2mn5Lg35L921cj5htvHJO90Z2+VnKf0r0U9zVffH7XYsyIi86CIiAiIgIiICIiAiIgIiICIiCqU52aDeaNvaLAOeXU7fXkqbncwynuY3cnkf0btsw7EN7THqvhFobX+RjyWo9JYTP3mxCFlrIUYp5BGCSGhzgTy7ucdvlKtr2NkY5j2h7HDYtcNwR6Cq0/h9joSTjbOQwoP8AdY62+OIejaI7xt/Q0f8AYL0TVRiaa5tPO/8A3/WWiVePsbeFBaG+5vpblBJA8EwbA+f+z8gVm0fw70tw9hsxaY09jNPxWXNdOzG1GQCUjcAuDQN9tz3+lYfEmx8as9+2h+yTxJsfGrPftofsk93h8fpKWjetCKr+JNj41Z79tD9kqnex2Wr8VcHp5mqcx4OuYW/flJlh7TtYZ6bGbfc/e8tiTfp38vUed7vD4/SS0b3VFC6s0XgNd4xuO1HhaGdx7ZBM2rka7Z4w8AgO5XAjcBxG/wApWj4k2PjVnv20P2SeJNj41Z79tD9knu8Pj9JLRvQDfY3cKWBwbw40u0PGzgMTB1G4Ox8n0gfqUnpngroDRmXiyuA0XgcNk4g5sdyjj4oZWhw2cA5rQRuCQVueJNj41Z79tD9kvviBTsO/pDIZXKs337G1deIj+VjOVrh8jgQmZhxrr5R/4Wh+crkPG7t8Nipeeo/mhyGRhd5ELOodFG4d8p7unvBu4kHla6ywQR1oI4YWNiijaGMYwbBrQNgAPMF8q1YaVeOvXhjrwRtDWRRNDWtA7gAOgCyrCuuJjNp1QSIiLUgiIgIiICIiAiIgIiICIiAiIgIiICIiAufZYt937SwJPN4sZfYebb21jd/P+TzfpHn6Cuf5Xf3ftLdW7eLGX6EDf/zWN7vPt+Tp3b+ZB0BERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAXPcsB/tA6VPM0HxXzHk7dT/veM677d36fOP0dCXPctt/tBaV6nm8V8xsOX/i8Z5/8A9/2QdCREQEREBERAREQEREBERAREQEREBERAREQERaeXy1fB46a7aLhDEBuGNLnOJIDWtA7ySQAPOSFYiaptGsbiKlP1Dquby4cVia7HdRHYuyOkaP8A3cse2/pAJHylfnw7rD8Qwfrc32a9fZa98c4Wy7oqR4d1h+IYP1ub7NPDusPxDB+tzfZp2WvfHOCy7rwHrH2e2V097IivibXCud2ocTHc06MfFmA7t5Z7FZzXsd7X35T7XG2w8oPB8wXsXw7rD8Qwfrc32a5BnvY/zah9kHh+LVjH4YZnHVexNQWJDFPM0csU7j2e/Oxp2H/Kz/D1dlr3xzgs9LIqR4d1h+IYP1ub7NPDusPxDB+tzfZp2WvfHOCy7oqR4d1h+IYP1ub7NPDusPxDB+tzfZp2WvfHOCy7oqUzPaua7d+NwsjR3tbdmaT+nsjt+pWPAZyHP0PbEbHwSMeYpq8u3PDI33zHbdOnpG4IIIJBBWqvArw4zp1eE3LJJERaEEREBERAREQEREBERAREQFUuJh2wVEeY5ahuD85jVtVR4m/eKh+dqH8zGvTk3f0ecMqdcNtERepiIiICKJy2qsXgsthsbesmG7mJn16MXZvd2r2RukcNwCG7Ma47uIHTbv6KRt24KFWazZmjr1oWOklmlcGsY0DcucT0AAG5JUGVFr43I1cxjqt+lPHapWomTwTxO5mSRuAc1zT5wQQR+VbCoItXKZWng8bayORtQ0aFWJ009mw8MjijaN3Oc49AAASSVmrzx2oI5oXiSKRoex7e5zSNwQgyLR0Af6V1kPMMszYAf8DVK3lo6A++2s/zvH/I1VZ7uvy/cMo1SuKIi5bEREQEREBERAREQEREBERAVR4m/eKh+dqH8zGrcqjxN+8VD87UP5mNenJu/o84ZU64bapHGvU1PSPDDOZG7NlIIuSOux2EkbHddLLI2KJsTndGuc97W8x6DffzK7qK1TpbFa10/dwecpR5HFXGdnPWl32eNwR1BBBBAIIIIIBBBC9M6mLzLpStxPZkOJvD+nl7uIzEunamSw5y2ddl5aU0kk0bh7adG1zecRjps4MPVpO6zxRal1Nw/vYHS1rWdfUWAz1eTUun8rqD+k3V3QbmCpf3I5H7tla7mbzbOG7AQF2Cn7HXh9RZkBFgXOfkaRx92aW/ZkltQl7X8skjpC55BY3lc4lzQNmkAkL432OfD5mAfhm4KVtR91uQfK3I2hadYawxtkNjte1JDCWjd/QEha82RzHEanhzWq+BGS03qLU8mOyFrK421WzN6UvkMNS04stRc3LJJHKzbmIJ8huzj0Kr+GqZbFaV17ozXuX1W/XE+mb1508uZfNjclCwnexU5SDAQSxrotmbNdts4EleiMZwk0jhYdMQ0MNHUi00+aTFMhlkaK75Y3xyu995Zc2R+5fzHdxPf1WnovgZofh9dtW8HgmVrFisab3z2JrPLXJ5jCwSvcGRk7Esbs07Dp0VzZHG8fVq6V9jvw0wuOvatv5fVUdD2lXx+oJYZnymkJHsFmQuNes1jHOLY9tthyjqVWqWqtaxcOsjp/IagymPyWN4lY7AMuw5Q27UVSZ9ZzojZdG0zbdu8cz2dRsCDsu8wexv4eVdPHBw4KWPGCyy5FE3JWg6tKwODHQP7Xmg2D3DaMtGziNtlt47gHoLEV5IKWAbWhkyFTKvjjtThr7dYh0M5HP1eCAXE+/I8vmUzZHCeKWPt4vTXsgdFSZ3OZLCUtKVszT9v5OaeeCR7LPaR9s5xe6JxgYSxxLdi4bbOIXonhXputpfQmIq1bmQvRSV45+1yV+W5Ju5jTsHyucQ30NB2HmC3Z9AaftZjN5SfGxz3M1RjxuQdK5z2WKzO05Y3MJ5dvusm+wBPN136L8aD4eYHhnhXYnTtSaljzJ2vYy25rHKeVrdmmV7i1oa1oDQQBt0CyiLSLGtHQH321n+d4/5Gqt5aOgPvtrP87x/yNVbJ7uvy/cMo1SuKIi5bEREQEREBERAREQEREBERAVR4m/eKh+dqH8zGrcorU2D8YcPLTbN7WmD45oZuXm7OWN4ewkbjcczRuNxuNxuN1vwKooxaaqtUTCxoloooZ9/UVfyJdJ2rEg6OfSuVnRH5WmSRjtvytB+RanjPmDfbTbo3LvmLXOcWTVHMZy8m4e8TcrXESNIaSCRuQCGkjoZn1R90dSyyIoTwtnviZlfWqX26eFs98TMr61S+3TM+qPujqtk2ihPC2e+JmV9apfbqr3eMdbH8Qsfoexg78WqshUfdrY4z1eaSFm/M7m7blHc47E7kNJA2BTM+qPujqWdDRQnhbPfEzK+tUvt08LZ74mZX1ql9umZ9UfdHUsm0UJ4Wz3xMyvrVL7dPC2e+JmV9apfbpmfVH3R1LJtaOgPvtrP87x/yNVRGP1RlcpI+GHSmRgsNBJiuWK0TmgPczmLe1Lw0ljtncpDgNwSCFbdKYObC0rDrcrJb92c2rJi37Nry1rQ1m/Xla1jW7nbfbfYb7DXiTFGHVEzGnRomJ2xOzyNUJtERcxiIiICIiAiIgIiICIiAiIgIvjnBjS5xDWgbknuCgY32NT2GyRyTUsRBOfeiNzcpGYuhDtyWxczz3crnOiBB7M/dA/M+Qs6lE1bEyy06ZjhlZnIuykilBk8uOEbkl3I07vLeUdowt5yHBstjcVTw8MkNGrFUikmksPbEwNDpJHl8jzt3uc5xJPnJKzVq0NKtFXrxMggiYI44omhrWNA2DQB0AA6bLKgIiIC/njxB9jLxuz3suqmsq2otK1c/OZszi43XbRigqVJYIhA8iv5xYjBABB3fufT/Q5c/wAhyzcfMByhpdX0zkec7nmaJLVHl6d2x7J3+n8qDoCIiAiIgis3p2vmWPla99DJivJWr5WqyP21Va8tLuzc9rhtzMjcWuBa4sbzNcBstV+opcRekhzcUNKpLahq0L0cjntsukb0bIOUdi/nBYASWu5o9ncz+Rs+iAirIqy6Jqh1NktrT9WCxNNWHbWrjHc3aNEI3c57QC9oiAJADGsGwDVYoJ47MLJoniSJ7Q5rm9xB7igyIiICIiAiIgIiICIiAiLFan9q1ppuR8vZsL+SMbudsN9gPOUEBZEOsr1zHu5J8JUdJTyVK5j+eO690bHBjXv8l0bQ883K1wL9m8wMcjDZFA6Dj5NF4R3a5SYyVI5i/Nn/AH3d7Q4iYDoHjm2LR0BGw6AKeQEREBERAXPuHBOq9Q6g1xvzUciIsdiHb7h9GAvInHXbaWWWZwI99G2E+jb96ltS8QsrY0pjJnR4iu8Mz+Qhc5ruXYO9pROHdI8Edo4Hdkbths+RrmXqvXiqQRwQRshhiaGMjjaGtY0DYAAdwA8yDIiIgIiICIiAoG7RfgbdrK0Ws7CeT2xkoXNlke8Nj5eeJrOby+VrByhp5+UDoepnkQa2OyNXMY+rfo2I7dK1E2eCxC4OZLG4BzXNI6EEEEH5Vsqv4WWSjqTMYuR+UtMcGZGGzbiBrxtlLmmvFKO8sdEXlrurRMzYkbBtgQEREBERAREQERQuY1tp7T9oVsnnMdj7JHN2Nm0xj9vTyk77LOmiqubUxeVtdNIqt7qWjvjTiPXY/pVZ4l3+G3FfQmZ0ln9R4qbFZSDsZQy/G17SCHMe07++a9rXDfpu0bgjotvZ8bgnlK5s7kjoXiBpeGWpow6k31NSdLSGKzuQidmJxCXDtnx83O8PjYJWv28qNzXnvKvy/nF7CngvR4K+yJ1ff1Hm8XJj8PTNbE5T2ywRXDM4fdIzvtuI2uDh3tL9j8vvT3UtHfGnEeux/SnZ8bgnlJmzuWlFVvdS0d8acR67H9Ke6lo7404j12P6U7PjcE8pM2dy0qm57O5DUGXk05puXsJIi0ZXM8vM3HsI37KLccr7Lm9zTuImuEjwd445ojJcRqus86zS+ls5UgfLHz28vFPG50LCPeVmu3Esx9OxZGOrtzysdesHg6Gm8XDjsbWbVpw8xbG0kkuc4ue9zjuXOc5znOc4lznOJJJJK1VUVUTauLJaz5gcDQ0xiK2MxlcVqVcEMZzFxJJLnOc5xLnvc4lznuJc5ziSSSSpBEWCCIiAiIgIiICIiCu2yG8Q8UN8yS/F3OkX3tHLNW/8b0Tnm+5+lgn9CsS45k/ZFcKq/EbFQy8T8LE9mNvtfEzO1Bjw4TVBtP8AdOk469mP8Ptj0LsaAiIgIiICIiDSzVx2Pw960wAvggklaD6WtJH/AMKo6SqR1sBSkA5p7MTJ55ndXzSOaC57iepJJ/R3dwVn1V+DGY+ZzfwFV7TX4OYr5pF/AF0MDRhT5rsSSIizQREQEREGrksbWy1OStajEkT/AJdi0jqHNI6tcDsQ4dQQCOq39B5SfNaLwd60/tbM9OJ8sm23O7lG7tvNueu3yrEsPCz+rnTnzGL+FY4unBnwmPxPRdi0oiLnIIiICIq3rrWcGisQLDoxZuTv7KrV5uXtX95JPma0bkn0DYbkgHZh4dWLXFFEXmRM5PLUcJUdbyNyvQqt99PalbGwflc4gKsS8YdHQvLTnIXEdN445Hj9YaQuH5O1azuR8IZWw6/e68skg8mIb+9jb3Mb0HQdTsCST1WNfW4XsPDin5tc38P7cvDuPuzaN+Gm+ry/UT3ZtG/DTfV5fqLhyLd8Dybiq5x0Lw4FxI9jppPVPsxsdqSvcjPD3JSeGMq4RSBsdhh3fBy7c33V/Keg2Ae70L3d7s2jfhpvq8v1Fw5E+B5NxVc46F4dx92bRvw031eX6i+s4yaNe7bw3G35XwyNH6y1cNRPgeTcVXOOheHpbD6gxmoa7p8XkKuQiaeVzq0rZA0+g7HofkKkF5YgMlK9HepTyUb8fvLVchr2/IehDh0HkuBB26gruvDfXw1jSmr22sgy9MNE8bPeytPdKweZpIII72kEdRsTxcu9l1ZLT7yib0+sLr1LkiIuEiL1V+DGY+ZzfwFV7TX4OYr5pF/AFYdVfgxmPmc38BVe01+DmK+aRfwBdHB7mfP9Lsb1h0jIJHQsbLMGksY53KHO26AnY7dfPsV524W8etUYzgrmNZ68xUVivUvW4Ks2Puiazdn8ISV46wh7GNrNnckbXcx5gOYhvVejV57h4Baul0DqXQU+RwsWAdfmy+By0Jldchsm8LkTZ4i0M5WvLmkteSRt0Ck32IsDfZCT6WtZmpxD0wdIWqGFlz8XtXINyEdmtE4Nla14YzaVrnMHJtsecbOIWCvxvzs9iriNT6Om0dNqDF27WEsx5Ntpz3xQ9q6KUNY0wyhh5wAXDyXeVuFG5ngRqji5kM3e4i3MNRdPp2xp+hU086WaOHt3NdJZe+VrCXbxx7MA2AB3J71u47hRrrV+qtNZHX9/BMqaap2oajMCZnvuWJ4DXdPL2jWiMCMv2Y3m6vPldAp/yEHpLjjmNNcMOC2MixbtV6o1XhGTNnyuWFRkj4oInSc072vL5XmQbN2Jds4kjZehMfNPZoVprNY07MkTXy1y8P7J5AJZzDodjuNx0Oy8/WOC2vncEMDw9sUdC6ir4+pJjpJMr7ZaOzY1rKtiPlY4smaA4uA8+3K8Ltmg9P29KaJwGFv5KTMXsdQgqT5CbfnsvZGGukO5J3cQT1JPXqSrTfaJ1YeFn9XOnPmMX8KzLDws/q5058xi/hVxe5nzj8SuxaURFzkEREBcC4s5J2S4iWIHOJixtWOCNp7muk+6PI/KOyB/5Au+rgXFnGuxnEOedzSIsnVjnjee5z4/ubwPyDsj/nC73sXN7Vp12m3p+rrslVkWvkb8WLoz25xKYYWF7xDC+V+w9DGAucfkAJVVHFvT5/us5/07kPsF9vViUUaKpiGtcnODWkkgAdST5lxOl7KDD3chUeyDHnCW7bKkU7M1A695T+RsjqY8sMLiD74uDTuWhXtnFHT997avY5o9uez2fp++xp36dXGAADr3k7KvcPtCau0HFj9Ptfp+9pmhI5sV6Zsovur7ktYWAcnMNwOfm7h73deTErrrqp9zVo22tO637Vin43X68OUyUmli3T2LzMmHuX/CDe0aW2BCJWRcnlN3c0kFzSNyBzAbnX4mcUMxNh9c0dL4Sa5BhaM8V3NNvisas5gL9oRsS98bXNcdi3Y9Ad1nyPCbL2+HWsMAyzSFzMZ2bJ13ue/s2xPtsmAeeTcO5WkbAEb+fzrBqHhprCv484/TlnCyYTVQmmkGTdMyarYlgEUhbyNIe13K09dtj6fPoqnKM2030x4X2/wdH0XPLa0dgpppHzTSUIHvkkcXOc4xtJJJ7yT51MKi4/W+K0bjKGDvtykl3H1oa0zqeFvTxFzY2glsjIS1w+UFZ/dd08f7rO/9O5D7Be2nFw4iImqL+aLmpbRWSdh9e4CyxxaJpzSlA/tslaQB/rEbv8qreFzVbP46O7UFhsDyQBarS15Oh2O7JGtcO7zjqrJonGuzOvcBWY3mbBObspH9hkbSQf8AWYx/mUyiaJwK5q1Wn8Mqdb0giIvzBUXqr8GMx8zm/gKr2mvwcxXzSL+AK05mm7I4i9UYQHzwSRAnzFzSP/tVDSVyOxgacIPJZrQsgsQO6Phka0BzHA9QQf1jYjoQuhgacKY8V2JhERZoIiICIiAsPCz+rnTnzGL+FY8nlK2IqPs2pRHG3oB3ue49A1rR1c4kgBo3JJAHUqQ0Ji58JozCUbTOzswU4mSx778j+Ubt38+x6b/IscXRgz4zH4nquxOoiLnIIiICrmudGQa1w4rPkFa3C/tatrl5jE/u6jpu0jcEb9x6EEAixotmHiVYVcV0TaYHl3K1LWn8h7Qy1c4+515WvO7JR/ijf3PHd3dRuNw09FjXpzJYulmaj6t+pBerP99DZibIw/laQQqxLwg0dK4uOBrtJ67RuewfqBAX1uF7cw5p+bRN/D+locKRdy9xvRvwHF+1k+snuN6N+A4v2sn1lu+OZNw1co6locNRdy9xvRvwHF+1k+snuN6N+A4v2sn1k+OZNw1co6locNRdy9xvRvwHF+1k+svrODujWO38BQO+R73uH6i7ZPjmTcNXKOpaN7hdYS5C8yjRgkv33+9q1wHPPynrs0dR5TiAN+pXduHGgho2jNPaeyfL2+UzyM95G0e9iYe8tBJO56uJJ2A2a2xYjBY3AVzBjKFbHwk7llaJsYcfSdh1Pylb64mXe1Ksrp93RFqfWV1ahERcNBQuY0Vp/UNgWMpg8bkZwOUS2qkcjwPRu4E7KaRZU11UTembSalW9yvRnxTwn7vi+qnuV6M+KeE/d8X1VaUW7tGNxzzlbzvVb3K9GfFPCfu+L6qe5Xoz4p4T93xfVVpRO0Y3HPOS871W9yvRnxTwn7vi+qnuV6M+KeE/d8X1VaUTtGNxzzkvO9B4rQ2nMFZbZx2AxlCw3flmrVI43t379iBuN1OIi1VV1VzeqbprERFgCIiAiIgIiICIiAiIgIiICIiAiIg//9k=)

#### Example Conversation

Now it's time to try out our mighty chatbot! Let's run it over the following list of dialog turns. If it hits a "RecursionLimit", that means the agent wasn't able to get an answer in the allocated number of steps. That's OK! We have more tricks up our sleeve in later sections of this tutorial.


```python
import shutil
import uuid

# Let's create an example conversation a user might have with the assistant
tutorial_questions = [
    "Hi there, what time is my flight?",
    "Am i allowed to update my flight to something sooner? I want to leave later today.",
    "Update my flight to sometime next week then",
    "The next available option is great",
    "what about lodging and transportation?",
    "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
    "OK could you place a reservation for your recommended hotel? It sounds nice.",
    "yes go ahead and book anything that's moderate expense and has availability.",
    "Now for a car, what are my options?",
    "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
    "Cool so now what recommendations do you have on excursions?",
    "Are they available while I'm there?",
    "interesting - i like the museums, what options are there? ",
    "OK great pick one and book it for my second day there.",
]

# Update with the backup file so we can restart from the original place in each section
db = update_dates(db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}


_printed = set()
for question in tutorial_questions:
    events = part_1_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
```

#### Part 1 Review

Our simple assistant is not bad! It was able to respond reasonably well for all the questions, quickly respond in-context, and successfully execute all our tasks. You can (check out an example LangSmith trace)[https://smith.langchain.com/public/f9e77b80-80ec-4837-98a8-254415cb49a1/r/26146720-d3f9-44b6-9bb9-9158cde61f9d] to get a better sense of how the LLM is prompted throughout the interactions above.

If this were a simple Q&A bot, we'd probably be happy with the results above. Since our customer support bot is taking actions on behalf of the user, some of its behavior above is a bit concerning:

1. The assistant booked a car when we were focusing on lodging, then had to cancel and rebook later on: oops! The user should have final say before booking to avoid unwanted feeds.
2. The assistant struggled to search for recommendations. We could improve this by adding more verbose instructions and examples using the tool, but doing this for every tool can lead to a large prompt and overwhelmed agent.
3. The assistant had to do an explicit search just to get the user's relevant information. We can save a lot of time by fetching the user's relevant travel details immediately so the assistant can directly respond.

In the next section, we will address the first two of these issues.

## Part 2: Add Confirmation

When an assistant takes actions on behalf of the user, the user should (almost) always have the final say on whether to follow through with the actions. Otherwise, any small mistake the assistant makes (or any prompt injection it succombs to) can cause real damage to the user.

In this section, we will use `interrupt_before` to pause the graph and return control to the user **before** executing any of the tools.

Your graph will look something like the following:

<img src="../img/part-2-diagram.png" src="../img/part-2-diagram.png">

As before, start by defining the state:

#### State & Assistant

Our graph state and LLM calling is nearly identical to Part 1 except Exception:

- We've added a `user_info` field that will be eagerly populated by our graph
- We can use the state directly in the `Assistant` object rather than using the configurable params


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Haiku is faster and cheaper, but less accurate
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# You could also use OpenAI or another model, though you will likely have
# to adapt the prompts
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4-turbo-preview")

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

part_2_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]
part_2_assistant_runnable = assistant_prompt | llm.bind_tools(part_2_tools)
```

#### Define Graph

Now, create the graph. Make 2 changes from part 1 to address our previous concerns.

1. Add an interrupt before using a tool
2. Explicitly populate the user state within the first node so the assistant doesn't have to use a tool just to learn about the user.


```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


# NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
# having to take an action
builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")
builder.add_node("assistant", Assistant(part_2_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))
builder.add_edge("fetch_user_info", "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
part_2_graph = builder.compile(
    checkpointer=memory,
    # NEW: The graph will always halt before executing the "tools" node.
    # The user can approve or reject (or even alter the request) before
    # the assistant continues
    interrupt_before=["tools"],
)
```


```python
from IPython.display import Image, display

try:
    display(Image(part_2_graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/jpg;base64,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)

#### Example Conversation

Now it's time to try out our newly revised chatbot! Let's run it over the following list of dialog turns.


```python
import shutil
import uuid

# Update with the backup file so we can restart from the original place in each section
db = update_dates(db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}


_printed = set()
# We can reuse the tutorial questions from part 1 to see how it does.
for question in tutorial_questions:
    events = part_2_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
    snapshot = part_2_graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        try:
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            # Just continue
            result = part_2_graph.invoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = part_2_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = part_2_graph.get_state(config)
```

#### Part 2 Review

Now our assistant was able to save a step to respond with our flight details. We also completely controlled which actions were performed. This all worked using LangGraph's `interrupts` and `checkpointers`. The interrupt pauses graph execution, its state safely persisted using your configured checkpointer. The user can then start it up at any time by running it with the right config.

See an [example LangSmith trace](https://smith.langchain.com/public/b3c71814-c366-476d-be6a-f6f3056caaec/r) to get a better sense of how the graph is running. Note [from this trace](https://smith.langchain.com/public/a077f4be-6baa-4e97-89f7-0dabc65c0fd0/r) that you typically **resume** a flow by invoking the graph with `(None, config)`. The state is loaded from the checkpoint as if it never was interrupted.

This graph worked pretty well! We *didn't really* need to be involved in *EVERY* assistant action, though...

In the next section, we will reorganize our graph so that we can interrupt only on the "sensitive" actions that actually write to the database.

## Part 3: Conditional Interrupt

In this section, we'll refine our interrupt strategy by categorizing tools as safe (read-only) or sensitive (data-modifying). We'll apply interrupts to the sensitive tools only, allowing the bot to handle simple queries autonomously.

This balances user control and conversational flow, but as we add more tools, our single graph may grow too complex for this "flat" structure. We'll address that in the next section. 

Your graph for Part 3 will look something like the following diagram.

<img src="../img/part-3-diagram.png" src="../img/part-3-diagram.png">

#### State

As always, start by defining the graph state. Our state and LLM calling **are identical to** part 2. 



```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Haiku is faster and cheaper, but less accurate
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# You can update the LLMs, though you may need to update the prompts
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4-turbo-preview")

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


# "Read"-only tools (such as retrievers) don't need a user confirmation to use
part_3_safe_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    search_car_rentals,
    search_hotels,
    search_trip_recommendations,
]

# These tools all change the user's reservations.
# The user has the right to control what decisions are made
part_3_sensitive_tools = [
    update_ticket_to_new_flight,
    cancel_ticket,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    book_hotel,
    update_hotel,
    cancel_hotel,
    book_excursion,
    update_excursion,
    cancel_excursion,
]
sensitive_tool_names = {t.name for t in part_3_sensitive_tools}
# Our LLM doesn't have to know which nodes it has to route to. In its 'mind', it's just invoking functions.
part_3_assistant_runnable = assistant_prompt | llm.bind_tools(
    part_3_safe_tools + part_3_sensitive_tools
)
```

#### Define Graph

Now, create the graph. Our graph is almost identical to part 2 **except** we split out the tools into 2 separate nodes. We only interrupt before the tools that are actually making changes to the user's bookings.


```python
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


# NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
# having to take an action
builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")
builder.add_node("assistant", Assistant(part_3_assistant_runnable))
builder.add_node("safe_tools", create_tool_node_with_fallback(part_3_safe_tools))
builder.add_node(
    "sensitive_tools", create_tool_node_with_fallback(part_3_sensitive_tools)
)
# Define logic
builder.add_edge("fetch_user_info", "assistant")


def route_tools(state: State):
    next_node = tools_condition(state)
    # If no tools are invoked, return to the user
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
    # This assumes single tool calls. To handle parallel tool calling, you'd want to
    # use an ANY condition
    first_tool_call = ai_message.tool_calls[0]
    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"


builder.add_conditional_edges(
    "assistant", route_tools, ["safe_tools", "sensitive_tools", END]
)
builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")

memory = MemorySaver()
part_3_graph = builder.compile(
    checkpointer=memory,
    # NEW: The graph will always halt before executing the "tools" node.
    # The user can approve or reject (or even alter the request) before
    # the assistant continues
    interrupt_before=["sensitive_tools"],
)
```


```python
from IPython.display import Image, display

try:
    display(Image(part_3_graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/jpg;base64,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xOyPykiIi8aCIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAqDxubzaHrDbm/4/wZ25d/8ApWp9B/8AL+0d4vy19xzZz6Fqjlc7/lBgjs1u56Zaod+8fr830oNgoiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiIC17x2IGg6vMdh74cD3NB6+N6m3f/wCfmWwlQOOIedDVeQyA+P8AB/7MbnbxtU3/ALNt9/o3QX9ERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEURmdXYPT0rYspmKOPlc3nEdmwyNxbvtvsTvtv03Ub8KWjvSnEeux/xW6nBxaovTTMxwlbStKKrfClo70pxHrsf8U+FLR3pTiPXY/4rLR8byTylcmdy0rUfuitc6ZwOmKeNy2oMVjsg7MYS22ncuxRTGFuVrOMvI57TyARvJd3Dkd37EK5/Clo70pxHrsf8V4r/AOEX4a4bi5V01q3SOVx2V1FSe3FWqta2xz5Kz3l0b9uboI3ufufkkJPRpTR8byTykyZ3PdOndU4XV1F93BZehmqbJDE6xjrLLEbXgAlpcwkb7OB2+kfKpRab4GR6A4J8K9P6Po6owr/F9cCxOy5GO3sO8qWTv36vJ237hsPMr38KWjvSnEeux/xTR8byTykyZ3LSiq3wpaO9KcR67H/FPhS0d6U4j12P+KaPjeSeUmTO5aUVXHFHR7iANUYgk9ABdj6/4qfx2Tp5inHboW4LtWTqyetIJGO/M4EgrCvCxKIvXTMcYS0w7KIi1IIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgLgnYErlcO+KfzINeaB5bWlcdlHAPuZSvHfszuHlyySMDiSevQbhoG+zWta0bAAKwqucNvwdaW/RVX/JarGuzj97VHrKztERFpQREQEREBERAUVjy3GcQKkVcCKPJUrElmNo2bJJE6EMkPm5g17mk7bkcu58kBSqiB+EfAf1C99aus6dcVR6T9plYXpERclBERAREQEREBERAREQE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saga1PKaNrNxjMPby9GDdtWzRfGT2W/kMka97SHNB23G7SG77tJ5R9e+DI+iec/Zr+2XbxIzlc10zFpnfDKYvKcRQfvgyPonnP2a/tk98GR9E85+zX9stebnfHOOqWlOIoP3wZH0Tzn7Nf2ye+DI+iec/Zr+2TNzvjnHUtKcRQfvgyPonnP2a/tk98GR9E85+zX9smbnfHOOpaU4ig/fBkfRPOfs1/bJ74Mj6J5z9mv7ZM3O+OcdS0pxFB++DI+iec/Zr+2T3wZH0Tzn7Nf2yZud8c46lpTiKD98GR9E85+zX9snvgyPonnP2a/tkzc745x1LSnEUH74Mj6J5z9mv7ZPfBkfRPOfs1/bJm53xzjqWlOIoP3wZH0Tzn7Nf2ye+DI+iec/Zr+2TNzvjnHUtKcUdo7789U/0Kf1HrqDP5N3Ruks253mBFZu/wDaZgB/ap7SWEs483shfDI7+Qe174I3czYWNbysZzec95JHTdx26Dc44lsPDqiZjXFtsT4xP4XYsKIi5TEREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQf/2Q==)

#### Example Conversation

Now it's time to try out our newly revised chatbot! Let's run it over the following list of dialog turns. This time, we'll have many fewer confirmations.


```python
import shutil
import uuid

# Update with the backup file so we can restart from the original place in each section
db = update_dates(db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

tutorial_questions = [
    "Hi there, what time is my flight?",
    "Am i allowed to update my flight to something sooner? I want to leave later today.",
    "Update my flight to sometime next week then",
    "The next available option is great",
    "what about lodging and transportation?",
    "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
    "OK could you place a reservation for your recommended hotel? It sounds nice.",
    "yes go ahead and book anything that's moderate expense and has availability.",
    "Now for a car, what are my options?",
    "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
    "Cool so now what recommendations do you have on excursions?",
    "Are they available while I'm there?",
    "interesting - i like the museums, what options are there? ",
    "OK great pick one and book it for my second day there.",
]


_printed = set()
# We can reuse the tutorial questions from part 1 to see how it does.
for question in tutorial_questions:
    events = part_3_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
    snapshot = part_3_graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        try:
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            # Just continue
            result = part_3_graph.invoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = part_3_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = part_3_graph.get_state(config)
```

#### Part 3 Review

Much better! Our agent is now working well - [check out a LangSmith trace](https://smith.langchain.com/public/a0d64d8b-1714-4cfe-a239-e170ca45e81a/r) of our latest run to inspect its work! You may be satisfied with this design. The code is contained, and it's behaving as desired. 

One problem with this design is that we're putting a lot of pressure on a single prompt. If we want to add more tools, or if each tool gets more complicated (more filters, more business logic constraining behavior, etc), it's likely the tool usage and overall behavior of the bot will start to suffer. 

In the next section, we show how you can take more control over different user experiences by routing to specialist agents or sub-graphs based on the user's intent.

## Part 4: Specialized Workflows

In the previous sections, we saw how "wide" chat-bots, relying on a single prompt and LLM to handle various user intents, can get us far. However, it's difficult to create **predictably great** user experiences for known intents with this approach.

Alternatively, your graph can detect userintent and select the appropriate workflow or "skill" to satisfy the user's needs. Each workflow can focus on its domain, allowing for isolated improvements without degrading the overall assistant.

In this section, we'll split user experiences into separate sub-graphs, resulting in a structure like this:

<img src="../img/part-4-diagram.png" src="../img/part-4-diagram.png">

In the diagram above, each square wraps an agentic, focused workflow. The primary assistant fields the user's initial queries, and the graph routes to the appropriate "expert" based on the query content.

#### State

We want to keep track of which sub-graph is in control at any given moment. While we _could_ do this through some arithmetic on the message list, it's easier to track as a dedicated **stack**. 

Add a `dialog_state` list to the `State` below. Any time a `node` is run and returns a value for `dialog_state`, the `update_dialog_stack` function will be called to determine how to apply the update.


```python
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]
```

#### Assistants

This time we will create an assistant **for every workflow**. That means:

1. Flight booking assistant
2. Hotel booking assistant
3. Car rental assistant
4. Excursion assistant
5. and finally, a "primary assistant" to route between these

If you're paying attention, you may recognize this as an example of the **supervisor** design pattern from our Multi-agent examples.

Below, define the `Runnable` objects to power each assistant.
Each `Runnable` has a prompt, LLM, and schemas for the tools scoped to that assistant.
Each *specialized* / delegated assistant additionally can call the `CompleteOrEscalate` tool to indicate that the control flow should be passed back to the primary assistant. This happens if it has successfully completed its work or if the user has changed their mind or needs assistance on something that beyond the scope of that particular workflow.

<div class="admonition note">
    <p class="admonition-title">Using Pydantic with LangChain</p>
    <p>
        This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.
    </p>
</div>


```python
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from pydantic import BaseModel, Field


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


# Flight booking assistant

flight_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling flight updates. "
            " The primary assistant delegates work to you whenever the user needs help updating their bookings. "
            "Confirm the updated flight details with the customer and inform them of any additional fees. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.',
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

update_flight_safe_tools = [search_flights]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools
update_flight_runnable = flight_booking_prompt | llm.bind_tools(
    update_flight_tools + [CompleteOrEscalate]
)

# Hotel Booking Assistant
book_hotel_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling hotel bookings. "
            "The primary assistant delegates work to you whenever the user needs help booking a hotel. "
            "Search for available hotels based on the user's preferences and confirm the booking details with the customer. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.'
            " Do not waste the user's time. Do not make up invalid tools or functions."
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'what's the weather like this time of year?'\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'i need to figure out transportation while i'm there'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Hotel booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_hotel_safe_tools = [search_hotels]
book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools
book_hotel_runnable = book_hotel_prompt | llm.bind_tools(
    book_hotel_tools + [CompleteOrEscalate]
)

# Car Rental Assistant
book_car_rental_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling car rental bookings. "
            "The primary assistant delegates work to you whenever the user needs help booking a car rental. "
            "Search for available car rentals based on the user's preferences and confirm the booking details with the customer. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
            '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'what's the weather like this time of year?'\n"
            " - 'What flights are available?'\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Car rental booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_car_rental_safe_tools = [search_car_rentals]
book_car_rental_sensitive_tools = [
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
]
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools
book_car_rental_runnable = book_car_rental_prompt | llm.bind_tools(
    book_car_rental_tools + [CompleteOrEscalate]
)

# Excursion Assistant

book_excursion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling trip recommendations. "
            "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
            "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'i need to figure out transportation while i'm there'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Excursion booking confirmed!'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_excursion_safe_tools = [search_trip_recommendations]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools
book_excursion_runnable = book_excursion_prompt | llm.bind_tools(
    book_excursion_tools + [CompleteOrEscalate]
)


# Primary Assistant
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    )


class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""

    location: str = Field(
        description="The location where the user wants to rent a car."
    )
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the car rental."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Basel",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "I need a compact car with automatic transmission.",
            }
        }


class ToHotelBookingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""

    location: str = Field(
        description="The location where the user wants to book a hotel."
    )
    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the hotel booking."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Zurich",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "I prefer a hotel near the city center with a room that has a view.",
            }
        }


class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""

    location: str = Field(
        description="The location where the user wants to book a recommended trip."
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the trip recommendation."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Lucerne",
                "request": "The user is interested in outdoor activities and scenic views.",
            }
        }


# The top-level assistant performs general Q&A and delegates specialized tasks to other assistants.
# The task delegation is a simple form of semantic routing / does simple intent detection
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Your primary role is to search for flight information and company policies to answer customer queries. "
            "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)
primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToFlightBookingAssistant,
        ToBookCarRental,
        ToHotelBookingAssistant,
        ToBookExcursion,
    ]
)
```

#### Create Assistant

We're about ready to create the graph. In the previous section, we made the design decision to have a shared `messages` state between all the nodes. This is powerful in that each delegated assistant can see the entire user journey and have a shared context. This, however, means that weaker LLMs can easily get mixed up about there specific scope. To mark the "handoff" between the primary assistant and one of the delegated workflows (and complete the tool call from the router), we will add a `ToolMessage` to the state.


#### Utility

Create a function to make an "entry" node for each workflow, stating "the current assistant is `assistant_name`".


```python
from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node
```

#### Define Graph

Now it's time to start building our graph. As before, we'll start with a node to pre-populate the state with the user's current information.


```python
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")
```

Now we'll start adding our specialized workflows. Each mini-workflow looks very similar to our full graph in [Part 3](#part-3-conditional-interrupt.md), employing 5 nodes:

1. `enter_*`: use the `create_entry_node` utility you defined above to add a ToolMessage signaling that the new specialized assistant is at the helm
2. Assistant: the prompt + llm combo that takes in the current state and either uses a tool, asks a question of the user, or ends the workflow (return to the primary assistant)
3. `*_safe_tools`: "read-only" tools the assistant can use without user confirmation.
4. `*_sensitive_tools`: tools with "write" access that require user confirmation (and will be assigned an `interrupt_before` when we compile the graph)
5. `leave_skill`: _pop_ the `dialog_state` to signal that the *primary assistant* is back in control

Because of their similarities, we _could_ define a factory function to generate these. Since this is a tutorial, we'll define them each explicitly.

First, make the **flight booking assistant** dedicated to managing the user journey for updating and canceling flights.


```python
# Flight booking assistant
builder.add_node(
    "enter_update_flight",
    create_entry_node("Flight Updates & Booking Assistant", "update_flight"),
)
builder.add_node("update_flight", Assistant(update_flight_runnable))
builder.add_edge("enter_update_flight", "update_flight")
builder.add_node(
    "update_flight_sensitive_tools",
    create_tool_node_with_fallback(update_flight_sensitive_tools),
)
builder.add_node(
    "update_flight_safe_tools",
    create_tool_node_with_fallback(update_flight_safe_tools),
)


def route_update_flight(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in update_flight_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "update_flight_safe_tools"
    return "update_flight_sensitive_tools"


builder.add_edge("update_flight_sensitive_tools", "update_flight")
builder.add_edge("update_flight_safe_tools", "update_flight")
builder.add_conditional_edges(
    "update_flight",
    route_update_flight,
    ["update_flight_sensitive_tools", "update_flight_safe_tools", "leave_skill", END],
)


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")
```

Next, create the **car rental assistant** graph to own all car rental needs.


```python
# Car rental assistant

builder.add_node(
    "enter_book_car_rental",
    create_entry_node("Car Rental Assistant", "book_car_rental"),
)
builder.add_node("book_car_rental", Assistant(book_car_rental_runnable))
builder.add_edge("enter_book_car_rental", "book_car_rental")
builder.add_node(
    "book_car_rental_safe_tools",
    create_tool_node_with_fallback(book_car_rental_safe_tools),
)
builder.add_node(
    "book_car_rental_sensitive_tools",
    create_tool_node_with_fallback(book_car_rental_sensitive_tools),
)


def route_book_car_rental(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in book_car_rental_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "book_car_rental_safe_tools"
    return "book_car_rental_sensitive_tools"


builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
builder.add_edge("book_car_rental_safe_tools", "book_car_rental")
builder.add_conditional_edges(
    "book_car_rental",
    route_book_car_rental,
    [
        "book_car_rental_safe_tools",
        "book_car_rental_sensitive_tools",
        "leave_skill",
        END,
    ],
)
```

Then define the **hotel booking** workflow.


```python
# Hotel booking assistant
builder.add_node(
    "enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel")
)
builder.add_node("book_hotel", Assistant(book_hotel_runnable))
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_node(
    "book_hotel_safe_tools",
    create_tool_node_with_fallback(book_hotel_safe_tools),
)
builder.add_node(
    "book_hotel_sensitive_tools",
    create_tool_node_with_fallback(book_hotel_sensitive_tools),
)


def route_book_hotel(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_hotel_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_hotel_safe_tools"
    return "book_hotel_sensitive_tools"


builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
builder.add_edge("book_hotel_safe_tools", "book_hotel")
builder.add_conditional_edges(
    "book_hotel",
    route_book_hotel,
    ["leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", END],
)
```

After that, define the **excursion assistant**.


```python
# Excursion assistant
builder.add_node(
    "enter_book_excursion",
    create_entry_node("Trip Recommendation Assistant", "book_excursion"),
)
builder.add_node("book_excursion", Assistant(book_excursion_runnable))
builder.add_edge("enter_book_excursion", "book_excursion")
builder.add_node(
    "book_excursion_safe_tools",
    create_tool_node_with_fallback(book_excursion_safe_tools),
)
builder.add_node(
    "book_excursion_sensitive_tools",
    create_tool_node_with_fallback(book_excursion_sensitive_tools),
)


def route_book_excursion(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_excursion_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_excursion_safe_tools"
    return "book_excursion_sensitive_tools"


builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
builder.add_edge("book_excursion_safe_tools", "book_excursion")
builder.add_conditional_edges(
    "book_excursion",
    route_book_excursion,
    ["book_excursion_safe_tools", "book_excursion_sensitive_tools", "leave_skill", END],
)
```

Finally, create the **primary assistant**.


```python
# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)


def route_primary_assistant(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return "enter_book_car_rental"
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_update_flight",
        "enter_book_car_rental",
        "enter_book_hotel",
        "enter_book_excursion",
        "primary_assistant_tools",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "update_flight",
    "book_car_rental",
    "book_hotel",
    "book_excursion",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# Compile graph
memory = MemorySaver()
part_4_graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ],
)
```


```python
from IPython.display import Image, display

try:
    display(Image(part_4_graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAGMB3ADASIAAhEBAxEB/8QAHQABAAIDAQEBAQAAAAAAAAAAAAUGAwQHAQIICf/EAF4QAAEDAwIDAgcHDwkGBQMCBwEAAgMEBREGEhMhMQciFBYyQVGU0xUjU1VhcdIIM0JDUlRWZHJ0gZGVo9EkNDU2YqGxsrMXJXN1orRjgoOSkyZEwUXCw4SF1OHwpP/EABsBAQEBAQEBAQEAAAAAAAAAAAABBAMCBQYH/8QAMxEBAAEBBgMGBgMBAQADAAAAAAECAxETIZHhUVKhEhQxQWHRBDIzcbHBIoHwFUIFI/H/2gAMAwEAAhEDEQA/AP6poiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIixVNQylp5Z5DiONhe4+gAZKeI17jd6e2BglLnzSZ4cETS+STHXDRzOPOeg860fd+ufzj07cXM8xdJTtJ/QZc/rX3pylLqRtyqG5r65jZZXEc2NIy2IehrQcY853O6uKmF2ns0Tddf/vRfBCe7tx/Byv8A/mpvap7u3H8HK/8A+am9qvu8awsOnrhRUF0vduttdXbvBaasq44pajb5XDa4guxnnjOFXbH25aBv+mbbf4NXWaC2XGXwemlqq+KLfNgHhYc765gtOzrhw5c1O3Tyx19y9P8Au7cfwcr/AP5qb2qe7tx/Byv/APmpvaqH1h2yaK0HHXG86mtdJPQvjjqaQ1kfHiL3RBu6PduaPfonEkDDXtPQhW2irae5UcFXSTxVVLPG2WGeF4eyRjhlrmuHIggggjqnbp5Y6+5eivGcU3OvttbbYxzdNM1j42j0ucxzto+U4A+ZTLHtkY17HBzXDIcDkEelfSg6Bos99dbY+VJUwvqoI/NEWua2Rrf7J3tcB5iXeYgC3U1xN0XSeKcREXFBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQFrXKjFxt1VSOOGzxPiJ9AcCP/wArZRWJum+BF2CvNVY6eSXInijEdRGAS5kjRh7cD5QenXkR1Cgx2p2UgnwLUfLn/Vi5/wD9upyts8nhT6231Ao6x4AkD2b4psDA3syOYHRwIPIA5AwsHG1Gzl4Ha5v7fhUkef8Ay8N2P1rr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u3Z1xgvy75tvPqgl0URPeK+Di7bFWThnF28KaDL9oBbjdIMb8kDOMFp3bRgpNfamESk2O4vDOJjZwTv2tDhj3z7IktGcc2nOBgkJdFETahMAlLrXcSI+J5EAdu2NDuWDz3bsN9JBSbUsFOJS+juOI+JnZQyvzsaHHG1pzkOw3HlEEDJCCXRREuqaGASmRlawR8TcTb5/sGB7sdznycMY8o5AyQQEmq7XCJTJUPjEXE3l8MgxsYJH9W+Zrgf7hzBCCXRRD9W2aLicS5U8XD3h/Eft27IxI/OfQwhx9AK+hquyFzm+7FBuaXBw8JZkFsYkcDz5YjIefQ0g9OaCVRaLL7bZDhtwpHHOMCdp+w3+n7g7vm59FnZXU0hwyoiceXkvB6jcP7ufzIM6LwEOGQQR6QvUBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBEWrW3WitrS6srKelaACTPK1gALg0HmfS5o+cgedBtIoh2qrcd3BfNVlu7lS08kvNsgjcMtaRkOOMegOPRpIOvdXIH+D2WtkI3YdK6KNpIk2Y5vzzGXg4wWjrkgEJdFEOnvkpeI6Ohp25cGukqHvJxIACWhg6x5d15OwOYyV46hvU28PulPACHBvg9H3m++AtOXPcDhg2nlzJLuXIIJhFEPsM02/jXi4SB27uteyPaDIHgAsYD3QNgOc7c5yTlHaUt0u/jMnqQ/eHCoqpZAQ6QSEYc4jG4DA8wG0YHJBI1FZT0jS6eeOFowcyPDR1A8/ykD9IUe/VlmZuAudNIW5y2KQSEYk4R5NyeUh2n0Hl5llh05aqdznRWyjje4uLnNgaCS6TiOyced/eP8Aa59VINaGDDQGj0AIIh2qKUh5ip6+oLQ44joZcHbIIyAS0DOTnGebQXDI5pJfKs8QQWOulLeJhznQsa4teGgc5M94EuBxjDTnBIBmEQREtZenGUQWylbjiBjqisLdxDwGHDY3YDm7nekEAYOSQljv0plEc9uph74I3Ohkmx3xwyRv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1zAS3LiSQCCRk8s46AKyqsaV/pu/flw/6as6w/EfUn+vxBIiIs6CIiAiIgIiICIiAiIgKI1O/h2+ndv2fy6kGd8jc5qIxj3sEnOcYPdPR2G5Kl1D6peI7ZCS8R/y6jGTLJH1qYhjLATz6YPdOcOw0koJhERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREGvW2+muMQjqqeOoYCSBI0HBwRkeg4JGflKjjb620tLrfM6qgY3+ZVL8nDWNa1rJDzHk5O7dkuPMKZRBhpallXFxIw4N3OYQ9paQWuLTyPyg8+h6jIWZQ8sYotS08kUe1tbE9k5ZGwbns2ljnOyHHDd4wA7qPJxzmEBERAREQEREBERAREQEREBERAREQEREBERAREQEREGtWWykuAb4VSw1Gxwc0yMDi0gEAgnocOdzHpPpWg2xTUDWC2180DI2BjaeoPGiIbGWMHe7w57XHDsnb8pKmEQRtJdJfCG0tdTOpal27Y5mXwyBoZkh+MNOX4DXYcdriAQCVJLDV0kVdTvgnjbJE/q1wBHpB5+cHB/QtHTdY6stTBJOKmankkpZJeKyRz3RvLC5xY1rQ47ckBowSRgYQSiIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAojUBIfasHH8uZnvRDPdd931+Zne9HnUuojUGd9qxn+fMzgxdNrvhP/wBne9HnQS6IiAiIgIiICLHJURREB8jGEkABzgMk9B/cVo+MtoDo2+6tFukLGsHhDMuLw4sA589wa4j07TjoUEkiiIdW2mpERgrG1DZeHsdA10gdva57DkA8i1pOen6wkGp6Wq4Rhp694k4RDjQTMAEjS5pJc0YwG977kkA4JCCXRREF/lqeCWWa4hshiyZGRs2B4cSXBzwe7gBwGTlwwDzwp7rcqjhF1klpg7hFwnqI8sDt28HYXAlmG9Dg7uR5FBLooenqr7KIjNbqCnB4RkArXyFoO7ige9AEt7u3n3snO3Az7Ay/PMRmlt0Q97MjY4pH55u4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)

#### Conversation

That was a lot! Let's run it over the following list of dialog turns. This time, we'll have many fewer confirmations.


```python
import shutil
import uuid

# Update with the backup file so we can restart from the original place in each section
db = update_dates(db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

_printed = set()
# We can reuse the tutorial questions from part 1 to see how it does.
for question in tutorial_questions:
    events = part_4_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
    snapshot = part_4_graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        try:
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            # Just continue
            result = part_4_graph.invoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = part_4_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = part_4_graph.get_state(config)
```

#### Conclusion:

You've now developed a customer support bot that handles diverse tasks using focused workflows.
More importantly, you've learned to use some of LangGraph's core features to design and refactor an application based on your product needs.

The above examples are by no means optimized for your unique needs - LLMs make mistakes, and each flow can be made more reliable through better prompts and experimentation. Once you've created your initial support bot, the next step would be to start [adding evaluations](https://docs.smith.langchain.com/evaluation) so you can confidently improve your system. Check out those docs and our other tutorials to learn more!
