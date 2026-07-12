import subprocess
import sys
from textwrap import dedent


def test_checkpoint_savers_import_without_psycopg() -> None:
    """Keep alternative drivers usable when the optional psycopg extra is absent."""
    script = dedent(
        """
        import builtins

        original_import = builtins.__import__

        def block_psycopg(name, *args, **kwargs):
            if name.split('.', 1)[0] in {'psycopg', 'psycopg_pool'}:
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_psycopg

        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.checkpoint.postgres.driver import (
            AsyncPostgresDriverAdapter,
            SyncPostgresDriverAdapter,
        )

        assert PostgresSaver
        assert AsyncPostgresSaver
        assert SyncPostgresDriverAdapter
        assert AsyncPostgresDriverAdapter
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
