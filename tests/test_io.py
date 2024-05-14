from typing import Iterator

from langgraph.pregel.io import single


def test_single() -> None:
    closed = False

    def myiter() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            nonlocal closed
            closed = True

    assert single(myiter()) == 1
    assert closed
