# A list of patterns that, if found in a code block, will cause us to leave that block unchanged.
import hashlib
import json
import os
import re
from textwrap import dedent, indent

from mistune import BlockParser, BlockState, Markdown, create_markdown
from mistune.renderers.markdown import MarkdownRenderer

preambles = {
    "python": "vcr_setup_preamble.py",
    "typescript": "nock_setup_preamble.ts",
}


def _get_python_cassette_init(cassette_name: str, hash_: str) -> str:
    return dedent(
        f"""
        _cassette = HashedCassette('{cassette_name}', '{hash_}')
        _cassette.__enter__()
        """
    )


def _get_typescript_cassette_init(cassette_name: str, hash_: str) -> str:
    return dedent(
        f"""
        const _cassette = new HashedCassette("{cassette_name}", "{hash_}");
        await _cassette.enter();
        """
    )


def _get_python_cassette_cleanup() -> str:
    return "_cassette.__exit__()"


def _get_typescript_cassette_cleanup() -> str:
    return "await _cassette.exit();"


preamble_inits = {
    "python": _get_python_cassette_init,
    "py": _get_python_cassette_init,
    "typescript": _get_typescript_cassette_init,
    "ts": _get_typescript_cassette_init,
}

preamble_cleanups = {
    "python": _get_python_cassette_cleanup,
    "py": _get_python_cassette_cleanup,
    "typescript": _get_typescript_cassette_cleanup,
    "ts": _get_typescript_cassette_cleanup,
}


def load_preamble(language: str, hash_: str, cassette_name: str) -> str:
    """Load the source code for the preamble for a given language."""
    _assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

    preamble_path = os.path.join(_assets_dir, preambles[language])
    with open(preamble_path, "r") as f:
        lines = f.readlines()
        lines.append(preamble_inits[language](cassette_name, hash_))
        return "\n".join(lines).strip()


def load_postamble(language: str) -> str:
    """Load the source code for the postamble for a given language."""
    return preamble_cleanups[language]()


def _hash_string(input_string: str) -> str:
    # Encode the input string to bytes
    encoded_string = input_string.encode("utf-8")
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256(encoded_string)
    # Get the hexadecimal digest of the hash
    return sha256_hash.hexdigest()


def extract_code_blocks_for_session(language: str, session: str, content: str) -> str:
    code_blocks_for_session = []

    TAB_REGEX = r"^===!? \"(?P<title>[^\"]+)\"\n(?P<content>(?:(?P<indent>    )+[^\n]*\n)+)"
    def parse_tabs(block: BlockParser, m: re.Match, state: BlockState) -> str:
        state.append_token(
            {
                "raw": m.group(0),
                "type": "block_tab",
                "attrs": {
                    "title": m.group("title"),
                    "level": len(m.group("indent")) // 4,
                    "content": dedent(m.group("content")).strip(),
                },
            }
        )
        return m.end()

    def render_tabs(self, token: dict, state: BlockState):
        recursive_transformer = create_markdown(renderer=DocumentRenderer())
        recursive_transformer.block.register("block_tab", TAB_REGEX, parse_tabs, before='list')
        recursive_transformer.renderer.register("block_tab", render_tabs)
        return (
            f'=== "{token["attrs"]["title"]}"\n'
            f'{indent(recursive_transformer(token["attrs"]["content"]), "    " * token["attrs"]["level"])}\n'
        )

    class DocumentRenderer(MarkdownRenderer):
        def block_code(self, token: dict, state: BlockState):
            if token["style"] == "fenced":
                if token["attrs"]["info"]:
                    attributes = {}
                    block_language = token["attrs"]["info"].split()[0]
                    for match in re.finditer(r'(?P<key>\w+)=(?:(?P<value>(?:[\w]+))|"(?P<value_quoted>(?:[^"\s]+))")', token["attrs"]["info"]):
                        attributes[match.group("key")] = match.group("value") or match.group("value_quoted")
                    if block_language == language and "session" in attributes and attributes["session"] == session:
                        code_blocks_for_session.append(token["raw"].rstrip())
            return super().block_code(token, state)

    transformer: Markdown = create_markdown(renderer=DocumentRenderer())
    transformer.block.register("block_tab", TAB_REGEX, parse_tabs, before='list')
    transformer.renderer.register("block_tab", render_tabs)
    
    # Parses the page content, which causes the code blocks to be added to the code_blocks_for_session list.
    # There's probably some way to do this by using the renderer as a filter, but I would've had to NO-OP
    # all of the default behavior, and this was easier.
    transformer(content)

    return code_blocks_for_session


def get_hash_for_session(language: str, session: str, content: str) -> str:
    # include the preamble in the hash so we invalidate if it changes
    preamble_hash = _hash_string(load_preamble(language, session, "test"))

    code_blocks_for_session = [preamble_hash, *extract_code_blocks_for_session(language, session, content)]
    
    return _hash_string("\n".join(code_blocks_for_session))
