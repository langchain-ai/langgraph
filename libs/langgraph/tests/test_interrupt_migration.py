import warnings

import pytest

from langgraph.types import Interrupt
from langgraph.warnings import LangGraphDeprecatedSinceV10


@pytest.mark.filterwarnings("ignore:LangGraphDeprecatedSinceV10")
def test_interrupt_legacy_ns() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)

        old_interrupt = Interrupt(
            value="abc", resumable=True, when="during", ns=["a:b", "c:d"]
        )

        new_interrupt = Interrupt.from_ns(value="abc", ns="a:b|c:d")
        assert new_interrupt.value == old_interrupt.value
        assert new_interrupt.id == old_interrupt.id
