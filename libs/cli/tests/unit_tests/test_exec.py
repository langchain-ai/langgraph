import sys

from langgraph_cli.exec import subp_exec


async def test_verbose_prints_command_but_not_stdin(capsys):
    await subp_exec(
        sys.executable,
        "-c",
        "import sys; sys.stdin.read()",
        input="registry-push-token\n",
        verbose=True,
    )

    out = capsys.readouterr().out
    assert out.startswith(f"+ {sys.executable} -c"), out
    assert "registry-push-token" not in out, out
