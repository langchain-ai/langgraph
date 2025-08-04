"""Unit tests for cross-reference preprocessing functionality."""

from unittest.mock import patch

import pytest

from _scripts.handle_auto_links import _transform_link, _replace_autolinks


@pytest.fixture
def mock_link_maps():
    """Fixture providing mock link maps for testing."""
    mock_scope_maps = {
        "python": {"py-link": "https://example.com/python"},
        "js": {"js-link": "https://example.com/js"},
    }

    with patch("_scripts.handle_auto_links.SCOPE_LINK_MAPS", mock_scope_maps):
        yield mock_scope_maps


def test_transform_link_basic(mock_link_maps) -> None:
    """Test basic link transformation."""
    # Test with a known link
    result = _transform_link("py-link", "python", "test.md", 1)
    assert result == "[py-link](https://example.com/python)"

    # Test with an unknown link (returns None)
    result = _transform_link("unknown-link", "global", "test.md", 1)
    assert result is None


def test_transform_link_with_custom_title(mock_link_maps) -> None:
    """Test link transformation with custom title."""
    # Test with a known link and custom title
    result = _transform_link("py-link", "python", "test.md", 1, "Custom Python Link")
    assert result == "[Custom Python Link](https://example.com/python)"

    # Test with unknown link and custom title (should still return None)
    result = _transform_link("unknown-link", "python", "test.md", 1, "Custom Title")
    assert result is None


def test_no_cross_refs(mock_link_maps) -> None:
    """Test markdown with no @[references]."""
    lines = ["# Title\n", "Regular text.\n"]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join(["# Title\n", "Regular text.\n"])
    assert result == expected


def test_global_cross_refs(mock_link_maps) -> None:
    """Test @[references] in global scope (no conditional blocks)."""
    lines = ["@[global-link]\n", "Text with @[unknown-link].\n"]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join(["@[global-link]\n", "Text with @[unknown-link].\n"])
    assert result == expected


def test_python_conditional_block(mock_link_maps) -> None:
    """Test @[references] inside Python conditional block."""
    lines = [":::python\n", "@[py-link]\n", ":::\n"]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join(
        [":::python\n", "[py-link](https://example.com/python)\n", ":::\n"]
    )
    assert result == expected


def test_js_conditional_block(mock_link_maps) -> None:
    """Test @[references] inside JavaScript conditional block."""
    lines = [":::js\n", "@[js-link]\n", ":::\n"]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join([":::js\n", "[js-link](https://example.com/js)\n", ":::\n"])
    assert result == expected


def test_all_scopes(mock_link_maps) -> None:
    """Test @[references] in global, Python, and JavaScript scopes."""
    lines = [
        "@[global-link]\n",
        ":::python\n",
        "@[py-link]\n",
        ":::\n",
        "@[global-link]\n",
        ":::js\n",
        "@[js-link]\n",
        ":::\n",
        "@[global-link]\n",
    ]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join(
        [
            "@[global-link]\n",
            ":::python\n",
            "[py-link](https://example.com/python)\n",
            ":::\n",
            "@[global-link]\n",
            ":::js\n",
            "[js-link](https://example.com/js)\n",
            ":::\n",
            "@[global-link]\n",
        ]
    )
    assert result == expected


def test_fence_resets_to_global(mock_link_maps) -> None:
    """Test that closing fence resets scope to global."""
    lines = [":::python\n", "@[py-link]\n", ":::\n", "@[global-link]\n"]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join(
        [
            ":::python\n",
            "[py-link](https://example.com/python)\n",
            ":::\n",
            "@[global-link]\n",
        ]
    )
    assert result == expected


def test_indented_conditional_fences(mock_link_maps) -> None:
    """Test @[references] inside indented conditional fences (e.g., in tabs or admonitions)."""
    lines = [
        "@[global-link]\n",
        "    :::python\n",
        "    @[py-link]\n",
        "    :::\n",
        "@[global-link]\n",
        "\t\t:::js\n",
        "\t\t@[js-link]\n",
        "\t\t:::\n",
        "@[global-link]\n",
    ]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join(
        [
            "@[global-link]\n",
            "    :::python\n",
            "    [py-link](https://example.com/python)\n",
            "    :::\n",
            "@[global-link]\n",
            "\t\t:::js\n",
            "\t\t[js-link](https://example.com/js)\n",
            "\t\t:::\n",
            "@[global-link]\n",
        ]
    )
    assert result == expected


def test_custom_title_syntax(mock_link_maps) -> None:
    """Test @[title][ref] syntax with custom titles."""
    lines = [
        ":::python\n",
        "@[Custom Python Title][py-link]\n",
        ":::\n",
        ":::js\n", 
        "@[Custom JS Title][js-link]\n",
        ":::\n"
    ]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join([
        ":::python\n",
        "[Custom Python Title](https://example.com/python)\n",
        ":::\n",
        ":::js\n",
        "[Custom JS Title](https://example.com/js)\n",
        ":::\n"
    ])
    assert result == expected


def test_mixed_syntax_compatibility(mock_link_maps) -> None:
    """Test that both @[ref] and @[title][ref] syntax work together."""
    lines = [
        ":::python\n",
        "@[py-link]\n",  # Old syntax
        "@[Custom Title][py-link]\n",  # New syntax
        ":::\n"
    ]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join([
        ":::python\n",
        "[py-link](https://example.com/python)\n",
        "[Custom Title](https://example.com/python)\n",
        ":::\n"
    ])
    assert result == expected


def test_custom_title_with_unknown_link(mock_link_maps) -> None:
    """Test @[title][ref] syntax with unknown reference."""
    lines = [
        ":::python\n",
        "@[Custom Title][unknown-link]\n",
        ":::\n"
    ]
    markdown = "".join(lines)
    result = _replace_autolinks(markdown, "test.md")
    expected = "".join([
        ":::python\n",
        "@[Custom Title][unknown-link]\n",  # Should remain unchanged
        ":::\n"
    ])
    assert result == expected
