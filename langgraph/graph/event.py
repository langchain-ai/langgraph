import uuid
from typing import Annotated, Generic, NotRequired, Sequence, TypedDict, TypeVar, Union

from langgraph.graph.state import StateGraph


class Event(TypedDict):
    id: NotRequired[str]


M_co = TypeVar("M_co", bound=Event, covariant=True)


def add_events(
    left: Union[M_co, Sequence[M_co]], right: Union[M_co, Sequence[M_co]]
) -> Sequence[M_co]:
    # coerce to list
    if not isinstance(left, Sequence):
        left = [left]
    if not isinstance(right, Sequence):
        right = [right]
    # assign missing ids
    for m in left:
        if m.get("id") is None:
            m["id"] = str(uuid.uuid4())
    for m in right:
        if m.get("id") is None:
            m["id"] = str(uuid.uuid4())
    # merge
    left_idx_by_id = {m["id"]: i for i, m in enumerate(left)}
    merged = list(left)
    for m in right:
        if (existing_idx := left_idx_by_id.get(m["id"])) is not None:
            merged[existing_idx] = m
        else:
            merged.append(m)
    return merged


class EventGraph(StateGraph, Generic[M_co]):
    """A StateGraph where every node
    - receives a list of events as input
    - returns one or more events as output.

    Events are dictionaries with an optional "id" field.
    The "id" field is auto-assigned if not provided, and used to dedupe events.
    """

    def __init__(self) -> None:
        super().__init__(Annotated[Sequence[M_co], add_events])
