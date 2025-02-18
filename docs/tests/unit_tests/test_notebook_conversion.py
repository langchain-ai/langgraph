import pytest

from _scripts.notebook_convert import (
    _convert_links_in_markdown,
    _has_output,
)


def test_has_output() -> None:
    """Test if a given code block is expected to have output."""
    assert _has_output("print('Hello, world!')") is True
    assert _has_output("print_stream(some_iterable)") is True
    assert _has_output("foo.y") is True
    assert _has_output("display(x)") is False
    assert _has_output("assert 1 == 1") is False
    assert _has_output("def foo(): pass") is False
    assert _has_output("import foobar") is False


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "This is a [link](https://example.com).",
            "This is a [link](https://example.com).",
        ),
        ("This is a [link](../foo).", "This is a [link](foo.md)."),
        ("This is a [link](../foo#hello).", "This is a [link](foo.md#hello)."),
        ("This is a [link](../foo/#hello).", "This is a [link](foo.md#hello)."),
    ],
)
def test_link_conversion(source: str, expected: str) -> None:
    """Test logic to convert links in markdown cells."""
    assert _convert_links_in_markdown(source) == expected
