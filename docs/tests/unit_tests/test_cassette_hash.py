import re
from textwrap import dedent

import pytest
from _scripts.hook_state import hook_state
from _scripts.notebook_hooks import handle_vcr_setup
from _scripts.setup_vcr import extract_code_blocks_for_session, get_hash_for_session

INITIAL_DOCUMENT_CONTENT = dedent(
    """
    Blah blah blah

    ```python exec="on" source="above" session="1" result="ansi"
    print("FIRST_CODE_BLOCK")
    ```

    Blah blah blah!

    ```python exec="on" source="above" session="1" result="ansi"
    print("SECOND_CODE_BLOCK")
    ```
    
    more blah blah blah
    """
)


@pytest.mark.parametrize(
    "replace_string",
    [
        "FIRST_CODE_BLOCK",
        "SECOND_CODE_BLOCK",
    ],
)
def test_changing_block_in_session_invalidates_hash(replace_string: str):
    hook_state['document_filename'] = 'test.md'
    hook_state['document_content'] = INITIAL_DOCUMENT_CONTENT

    code = "print('Hello, world!')"

    result1 = handle_vcr_setup(
        formatter=lambda **kwargs: None,
        language="python",
        session="1",
        id="test",
        code=code,
        md=None,
        extra={},
    )

    cassette_init_expr = re.compile(r"^_cassette = HashedCassette\('[^']+', '(?P<hash>[^']+)'\)$")
    assert result1['transform_source']
    execute_source, display_source = result1['transform_source'](code)
    assert display_source == code
    cassette_init_line = [line for line in execute_source.splitlines() if line.startswith("_cassette = HashedCassette(")][0]
    assert cassette_init_line
    match = cassette_init_expr.match(cassette_init_line)
    assert match
    hash_ = str(match.group('hash'))

    # change the content of the second block of code
    hook_state['document_content'] = INITIAL_DOCUMENT_CONTENT.replace(replace_string, "world")
    assert hook_state['document_content'] != INITIAL_DOCUMENT_CONTENT

    result2 = handle_vcr_setup(
        formatter=lambda **kwargs: None,
        language="python",
        session="1",
        id="test",
        code=code,
        md=None,
        extra={},
    )

    assert result2['transform_source']
    execute_source, display_source = result2['transform_source'](code)
    assert display_source == code
    cassette_init_line = [line for line in execute_source.splitlines() if line.startswith("_cassette = HashedCassette(")][0]
    assert cassette_init_line
    match = cassette_init_expr.match(cassette_init_line)
    assert match

    # this is the important part
    assert str(match.group('hash')) != hash_

@pytest.mark.parametrize(
    "replace_string",
    [
        "FIRST_CODE_BLOCK",
        "SECOND_CODE_BLOCK",
    ],
)
def test_get_hash_for_session(replace_string: str):
    hash_ = get_hash_for_session(
        language="python",
        session="1",
        content=INITIAL_DOCUMENT_CONTENT,
    )
    
    content = INITIAL_DOCUMENT_CONTENT.replace(replace_string, "world")
    assert content != INITIAL_DOCUMENT_CONTENT
    assert get_hash_for_session(
        language="python",
        session="1",
        content=INITIAL_DOCUMENT_CONTENT.replace(replace_string, "world"),
    ) != hash_

def test_get_code_blocks_for_session():
    code_blocks = extract_code_blocks_for_session(
        language="python",
        session="1",
        content=INITIAL_DOCUMENT_CONTENT,
    )
    assert code_blocks
    assert len(code_blocks) == 2
    assert code_blocks[0] == 'print("FIRST_CODE_BLOCK")'
    assert code_blocks[1] == 'print("SECOND_CODE_BLOCK")'
