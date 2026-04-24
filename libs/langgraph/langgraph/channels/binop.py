from collections.abc import Callable
from typing import Generic

from langgraph.channels.aggregate import (
    AggregateChannel,
)
from langgraph.channels.aggregate import (
    _get_overwrite as _get_overwrite,
)
from langgraph.channels.aggregate import (
    _strip_extras as _strip_extras,
)
from langgraph.channels.base import Value

__all__ = ("BinaryOperatorAggregate",)


class BinaryOperatorAggregate(AggregateChannel[Value], Generic[Value]):
    """Stores the result of applying a binary operator to the current value and each new value.

    ```python
    import operator

    total = Channels.BinaryOperatorAggregate(int, operator.add)
    ```

    Equivalent to `AggregateChannel(operator, typ=typ, snapshot_frequency=1)`.
    Preserved as a distinct subclass so existing `isinstance(x, BinaryOperatorAggregate)`
    checks and `_is_field_binop` detection continue to work. New code should
    prefer `AggregateChannel` directly — especially when a non-unit
    `snapshot_frequency` is wanted.
    """

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]):
        super().__init__(operator, typ=typ, snapshot_frequency=1)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, BinaryOperatorAggregate) and (
            value.operator is self.operator
            if value.operator.__name__ != "<lambda>"
            and self.operator.__name__ != "<lambda>"
            else True
        )
