"""Unit tests for langgraph.warnings module.

These tests cover the LangGraphDeprecationWarning class hierarchy,
its __init__ logic, __str__ formatting, and version-specific subclasses.
"""

import warnings

import pytest

from langgraph.warnings import (
    LangGraphDeprecatedSinceV05,
    LangGraphDeprecatedSinceV10,
    LangGraphDeprecationWarning,
)

# --- LangGraphDeprecationWarning tests ---


def test_deprecation_warning_is_deprecation_warning() -> None:
    """Test that LangGraphDeprecationWarning is a DeprecationWarning subclass."""
    assert issubclass(LangGraphDeprecationWarning, DeprecationWarning)


def test_deprecation_warning_basic_init() -> None:
    """Test basic initialization with message and since."""
    w = LangGraphDeprecationWarning("Use new_func instead", since=(1, 0))
    assert w.message == "Use new_func instead"
    assert w.since == (1, 0)


def test_deprecation_warning_strips_trailing_dot() -> None:
    """Test that trailing dot is stripped from message."""
    w = LangGraphDeprecationWarning("This is deprecated.", since=(0, 5))
    assert w.message == "This is deprecated"


def test_deprecation_warning_no_trailing_dot() -> None:
    """Test that message without trailing dot is unchanged."""
    w = LangGraphDeprecationWarning("No trailing dot", since=(1, 0))
    assert w.message == "No trailing dot"


def test_deprecation_warning_default_expected_removal() -> None:
    """Test that expected_removal defaults to next major version."""
    w = LangGraphDeprecationWarning("test", since=(1, 0))
    assert w.expected_removal == (2, 0)

    w2 = LangGraphDeprecationWarning("test", since=(0, 5))
    assert w2.expected_removal == (1, 0)

    w3 = LangGraphDeprecationWarning("test", since=(3, 2))
    assert w3.expected_removal == (4, 0)


def test_deprecation_warning_custom_expected_removal() -> None:
    """Test that custom expected_removal overrides default."""
    w = LangGraphDeprecationWarning("test", since=(1, 0), expected_removal=(1, 5))
    assert w.expected_removal == (1, 5)


def test_deprecation_warning_str_format() -> None:
    """Test __str__ produces expected format with version info."""
    w = LangGraphDeprecationWarning("Use X instead", since=(1, 0))
    result = str(w)
    assert result == (
        "Use X instead. Deprecated in LangGraph V1.0 to be removed in V2.0."
    )


def test_deprecation_warning_str_format_with_custom_removal() -> None:
    """Test __str__ with custom expected_removal version."""
    w = LangGraphDeprecationWarning("Old API", since=(0, 5), expected_removal=(2, 0))
    result = str(w)
    assert result == ("Old API. Deprecated in LangGraph V0.5 to be removed in V2.0.")


def test_deprecation_warning_str_strips_dot_before_formatting() -> None:
    """Test __str__ when message has trailing dot (stripped before formatting)."""
    w = LangGraphDeprecationWarning("Deprecated feature.", since=(1, 0))
    result = str(w)
    # Should not have double dots
    assert ".." not in result
    assert result.startswith("Deprecated feature.")


def test_deprecation_warning_with_extra_args() -> None:
    """Test that extra positional args are passed to base Exception."""
    w = LangGraphDeprecationWarning("msg", "extra1", "extra2", since=(1, 0))
    assert w.args == ("msg", "extra1", "extra2")
    assert w.message == "msg"


def test_deprecation_warning_can_be_raised() -> None:
    """Test that LangGraphDeprecationWarning can be raised and caught."""
    with pytest.raises(LangGraphDeprecationWarning):
        raise LangGraphDeprecationWarning("test", since=(1, 0))


def test_deprecation_warning_can_be_warned() -> None:
    """Test that LangGraphDeprecationWarning works with warnings.warn."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.warn(
            LangGraphDeprecationWarning("test warning", since=(1, 0)),
            stacklevel=1,
        )
        assert len(w) == 1
        assert issubclass(w[0].category, LangGraphDeprecationWarning)
        assert "test warning" in str(w[0].message)


# --- LangGraphDeprecatedSinceV05 tests ---


def test_v05_is_deprecation_warning_subclass() -> None:
    """Test LangGraphDeprecatedSinceV05 inherits from LangGraphDeprecationWarning."""
    assert issubclass(LangGraphDeprecatedSinceV05, LangGraphDeprecationWarning)
    assert issubclass(LangGraphDeprecatedSinceV05, DeprecationWarning)


def test_v05_sets_correct_versions() -> None:
    """Test LangGraphDeprecatedSinceV05 uses since=(0, 5) and removal=(2, 0)."""
    w = LangGraphDeprecatedSinceV05("Use new API")
    assert w.since == (0, 5)
    assert w.expected_removal == (2, 0)


def test_v05_str_format() -> None:
    """Test LangGraphDeprecatedSinceV05 __str__ output."""
    w = LangGraphDeprecatedSinceV05("Old function")
    assert str(w) == (
        "Old function. Deprecated in LangGraph V0.5 to be removed in V2.0."
    )


def test_v05_with_extra_args() -> None:
    """Test LangGraphDeprecatedSinceV05 with extra args."""
    w = LangGraphDeprecatedSinceV05("msg", "extra")
    assert w.args == ("msg", "extra")


# --- LangGraphDeprecatedSinceV10 tests ---


def test_v10_is_deprecation_warning_subclass() -> None:
    """Test LangGraphDeprecatedSinceV10 inherits from LangGraphDeprecationWarning."""
    assert issubclass(LangGraphDeprecatedSinceV10, LangGraphDeprecationWarning)
    assert issubclass(LangGraphDeprecatedSinceV10, DeprecationWarning)


def test_v10_sets_correct_versions() -> None:
    """Test LangGraphDeprecatedSinceV10 uses since=(1, 0) and removal=(2, 0)."""
    w = LangGraphDeprecatedSinceV10("Use new API")
    assert w.since == (1, 0)
    assert w.expected_removal == (2, 0)


def test_v10_str_format() -> None:
    """Test LangGraphDeprecatedSinceV10 __str__ output."""
    w = LangGraphDeprecatedSinceV10("Old function")
    assert str(w) == (
        "Old function. Deprecated in LangGraph V1.0 to be removed in V2.0."
    )


def test_v10_with_extra_args() -> None:
    """Test LangGraphDeprecatedSinceV10 with extra args."""
    w = LangGraphDeprecatedSinceV10("msg", "extra")
    assert w.args == ("msg", "extra")


# --- Cross-class tests ---


def test_v05_and_v10_are_different_classes() -> None:
    """Test V05 and V10 are distinct classes."""
    assert LangGraphDeprecatedSinceV05 is not LangGraphDeprecatedSinceV10


def test_isinstance_hierarchy() -> None:
    """Test isinstance checks work across the hierarchy."""
    v05 = LangGraphDeprecatedSinceV05("test")
    v10 = LangGraphDeprecatedSinceV10("test")

    # Both are LangGraphDeprecationWarning
    assert isinstance(v05, LangGraphDeprecationWarning)
    assert isinstance(v10, LangGraphDeprecationWarning)

    # Both are DeprecationWarning
    assert isinstance(v05, DeprecationWarning)
    assert isinstance(v10, DeprecationWarning)

    # But each is only its own subclass
    assert isinstance(v05, LangGraphDeprecatedSinceV05)
    assert not isinstance(v05, LangGraphDeprecatedSinceV10)
    assert isinstance(v10, LangGraphDeprecatedSinceV10)
    assert not isinstance(v10, LangGraphDeprecatedSinceV05)
