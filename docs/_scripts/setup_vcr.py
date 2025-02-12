# A list of patterns that, if found in a code block, will cause us to leave that block unchanged.
import hashlib
import os
from textwrap import dedent

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


def load_preamble(language: str, code: str, cassette_name: str) -> str:
    """Load the source code for the preamble for a given language."""
    _assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

    preamble_path = os.path.join(_assets_dir, preambles[language])
    with open(preamble_path, "r") as f:
        lines = f.readlines()
        hash_ = _hash_string(code)
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
