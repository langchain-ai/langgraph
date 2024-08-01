import json
from typing import Any, Dict, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base.utils import get_checkpoint_id


def _metadata_predicate(
    metadata_filter: Dict[str, Any],
) -> Tuple[Sequence[str], Sequence[Any]]:
    """Return WHERE clause predicates for (a)search() given metadata filter.

    This method returns a tuple of a string and a tuple of values. The string
    is the parametered WHERE clause predicate (excluding the WHERE keyword):
    "column1 = ? AND column2 IS ?". The tuple of values contains the values
    for each of the corresponding parameters.
    """

    def _where_value(query_value: Any) -> Tuple[str, Any]:
        """Return tuple of operator and value for WHERE clause predicate."""
        if query_value is None:
            return ("IS ?", None)
        elif (
            isinstance(query_value, str)
            or isinstance(query_value, int)
            or isinstance(query_value, float)
        ):
            return ("= ?", query_value)
        elif isinstance(query_value, bool):
            return ("= ?", 1 if query_value else 0)
        elif isinstance(query_value, dict) or isinstance(query_value, list):
            # query value for JSON object cannot have trailing space after separators (, :)
            # SQLite json_extract() returns JSON string without whitespace
            return ("= ?", json.dumps(query_value, separators=(",", ":")))
        else:
            return ("= ?", str(query_value))

    predicates = []
    param_values = []

    # process metadata query
    for query_key, query_value in metadata_filter.items():
        operator, param_value = _where_value(query_value)
        predicates.append(
            f"json_extract(CAST(metadata AS TEXT), '$.{query_key}') {operator}"
        )
        param_values.append(param_value)

    return (predicates, param_values)


def search_where(
    config: Optional[RunnableConfig],
    filter: Optional[Dict[str, Any]],
    before: Optional[RunnableConfig] = None,
) -> Tuple[str, Sequence[Any]]:
    """Return WHERE clause predicates for (a)search() given metadata filter
    and `before` config.

    This method returns a tuple of a string and a tuple of values. The string
    is the parametered WHERE clause predicate (including the WHERE keyword):
    "WHERE column1 = ? AND column2 IS ?". The tuple of values contains the
    values for each of the corresponding parameters.
    """
    wheres = []
    param_values = []

    # construct predicate for config filter
    if config is not None:
        wheres.append("thread_id = ?")
        param_values.append(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        wheres.append("checkpoint_ns = ?")
        param_values.append(checkpoint_ns)

    # construct predicate for metadata filter
    if filter:
        metadata_predicates, metadata_values = _metadata_predicate(filter)
        wheres.extend(metadata_predicates)
        param_values.extend(metadata_values)

    # construct predicate for `before`
    if before is not None:
        wheres.append("checkpoint_id < ?")
        param_values.append(get_checkpoint_id(before))

    return ("WHERE " + " AND ".join(wheres) if wheres else "", param_values)
