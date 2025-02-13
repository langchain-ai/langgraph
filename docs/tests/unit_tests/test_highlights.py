from _scripts.notebook_hooks import _highlight_code_blocks


NO_OP_INPUT_1 = """\
This is a plain text without any code blocks.

```python
print("Hello, World!")
```
"""

NO_OP_INPUT_2 = """\

=== "Python"

    ```python
        def foo():
            pass
    print("Hello, World!")
    ```
"""


def test_highlight_code_blocks_no_op() -> None:
    assert _highlight_code_blocks(NO_OP_INPUT_1) == NO_OP_INPUT_1
    assert _highlight_code_blocks(NO_OP_INPUT_2) == NO_OP_INPUT_2


# Examples are written in multiline style to make sure that whitespace
# is easy to interpret.
INPUT_HIGHLIGHT_1 = """\
This is a plain text without any code blocks.

```python
# highlight-next-line
print("Hello, World!")
```
"""

EXPECTED_HIGHLIGHT_1 = """\
This is a plain text without any code blocks.

```python hl_lines="1"
print("Hello, World!")
```
"""

INPUT_HIGHLIGHT_2 = """\
This is a plain text without any code blocks.

```python
# highlight-next-line
print("Hello, World!")

x = 5

# highlight-next-line
print("Hello, World!")

```
"""

EXPECTED_HIGHLIGHT_2 = """\
This is a plain text without any code blocks.

```python hl_lines="1 5"
print("Hello, World!")

x = 5

print("Hello, World!")

```
"""

def test_highlight_code_blocks() -> None:
    """Test that code blocks are highlighted correctly."""
    assert _highlight_code_blocks(INPUT_HIGHLIGHT_1) == EXPECTED_HIGHLIGHT_1
    assert _highlight_code_blocks(INPUT_HIGHLIGHT_2) == EXPECTED_HIGHLIGHT_2
