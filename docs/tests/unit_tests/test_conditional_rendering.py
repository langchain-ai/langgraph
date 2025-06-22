from _scripts.notebook_hooks import _apply_conditional_rendering


CONDITIONAL_RENDERING = """
above
:::js
js-content
:::
between
:::python
python-content
:::
below
"""


def test_conditional_rendering() -> None:
    """Test logic for conditional rendering of content."""
    output = _apply_conditional_rendering(CONDITIONAL_RENDERING, "js")
    assert output.strip() == "above\njs-content\n\nbetween\n\nbelow"
    output = _apply_conditional_rendering(CONDITIONAL_RENDERING, "python")
    assert output.strip() == "above\n\nbetween\npython-content\n\nbelow"
