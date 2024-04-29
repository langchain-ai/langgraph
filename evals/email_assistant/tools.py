import sqlite3
from datetime import date, datetime
from typing import Optional

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool

from evals.email_assistant.db import db_file


@tool
async def search_emails(
    queries: Optional[list[str] | str] = None,
    sender: Optional[str] = None,
    recipient: Optional[str] = None,
    start_date: Optional[date | datetime] = None,
    end_date: Optional[date | datetime] = None,
    thread_id: Optional[int] = None,
) -> dict:
    """Search emails based on the given queries and optional filters.
    queries: The search queries to match against the subject or body of the email. The queries are joined with an OR condition.
    sender: A substring to match against the sender email address.
        If you don't know the email domain, just search by username.
    recipient: A substring to match against the recipient email address.
        If you don't know the email domain, just search by username.
    start_date: The start date for the search range (inclusive, but be lenient on bounds).
    end_date: The end date for the search range (exclusive)
    thread_id: The thread ID of the email thread."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    conditions = []
    params = []

    if queries:
        if isinstance(queries, str):
            queries = [queries]
        query_conditions = []
        for query in queries:
            for exact in query.split('"'):
                for tok in exact.split():
                    tok = tok.replace("*", "%")
                    query_conditions.append(
                        "(subject LIKE ? OR body LIKE ? OR sender LIKE ? OR recipient LIKE ?)"
                    )
                    params.extend([f"%{tok}%", f"%{tok}%", f"%{tok}%", f"%{tok}%"])
        conditions.append(f"({' OR '.join(query_conditions)})")

    if sender:
        conditions.append("sender LIKE ?")
        params.append(f"%{sender}%")
    if recipient:
        conditions.append("recipient LIKE ?")
        params.append(f"%{recipient}%")
    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("timestamp < ?")
        params.append(end_date)
    if thread_id:
        conditions.append("thread_id = ?")
        params.append(thread_id)

    where_clause = " AND ".join(conditions)
    if where_clause:
        where_clause = " WHERE " + where_clause
    cursor.execute(f"SELECT * FROM emails{where_clause}", params)
    results = cursor.fetchall()
    conn.close()
    keys = [column[0] for column in cursor.description]
    return {
        "results": [dict(zip(keys, row)) for row in results],
        "count": len(results),
    }


@tool
async def search_calendar_events(
    queries: Optional[list[str] | str] = None,
    start_date: Optional[datetime | date] = None,
    end_date: Optional[datetime | date] = None,
) -> dict:
    """Search calendar events based on the given queries and optional date range.

    Args:
        queries (list[str] | str | null): The search queries to match against the title or description of the event. The queries are joined with an OR condition.
            Examples:
                - queries: ["standup", "walk the dog"] (search for events containing either of these terms)
                - queries: "team offsite" (search for events containing the exact phrase "team offsite")
            Importantly, is NOT a comma separated list.

        start_date (datetime | date | null): The start time for the search range (inclusive).
        end_date (datetime | date | null): The end time for the search range (exclusive).
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    conditions = []
    params = []

    if queries:
        if isinstance(queries, str):
            queries = [queries]
        query_conditions = []
        for query in queries:
            for exact in query.split('"'):
                for tok in exact.split():
                    tok = tok.replace("*", "%")
                    query_conditions.append("(title LIKE ? OR description LIKE ?)")
                    params.extend([f"%{tok}%", f"%{tok}%"])
        conditions.append(f"({' OR '.join(query_conditions)})")

    if start_date:
        conditions.append("start_time >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("end_time < ?")
        params.append(end_date)

    where_clause = " AND ".join(conditions)
    if where_clause:
        where_clause = " WHERE " + where_clause
    cursor.execute(
        f"SELECT title, description, start_time, end_time FROM calendar_events{where_clause}",
        params,
    )
    results = cursor.fetchall()
    conn.close()
    keys = [column[0] for column in cursor.description]
    return {
        "results": [dict(zip(keys, row)) for row in results],
        "count": len(results),
    }


@tool
async def create_calendar_event(
    title: str, description: str, start_time: datetime, end_time: datetime
) -> str:
    """Create a calendar event with the given details."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO calendar_events (title, description, start_time, end_time) VALUES (?, ?, ?, ?)",
        (title, description, start_time, end_time),
    )
    conn.commit()
    conn.close()
    return f"Calendar event created: {title} from {start_time} to {end_time}."


@tool
async def send_email(
    to: str, subject: str, body: str, thread_id: Optional[int] = None
) -> str:
    """Send an email with the given details."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    config = ensure_config()
    user_id = config["configurable"]["user_id"]
    if thread_id is None:
        # Fetch the maximum thread ID and increment it by 1
        cursor.execute("SELECT COALESCE(MAX(thread_id), 0) + 1 FROM emails")
        thread_id = cursor.fetchone()[0]
    cursor.execute(
        "INSERT INTO emails (sender, recipient, subject, body, thread_id) VALUES (?, ?, ?, ?, ?)",
        (user_id, to, subject, body, thread_id),
    )
    conn.commit()
    conn.close()
    return f"Email sent to {to} with subject '{subject}'."
