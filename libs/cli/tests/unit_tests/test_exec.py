import asyncio

from langgraph_cli.exec import monitor_stream


async def test_monitor_stream_overrun_without_separator_passes_bytes() -> None:
    """A chunk longer than the limit with no newline must reach the handler.

    Regression test: the LimitOverrunError branch used
    `line = stream._buffer.clear()`, which returns None, so the oversized
    data was silently dropped and `handle(None, overrun=True)` raised
    TypeError in display mode (`sys.stdout.buffer.write(None)`).
    """
    limit = 16
    stream = asyncio.StreamReader(limit=limit)
    payload = b"x" * (limit * 4)  # no newline anywhere
    stream.feed_data(payload)
    stream.feed_eof()

    seen: list[tuple[bytes, bool]] = []

    def display_like_handler(line: bytes, overrun: bool) -> None:
        # mimics the display branch: crashes with TypeError if line is None
        assert isinstance(line, (bytes, bytearray)), (
            f"handler received {type(line).__name__}, expected bytes"
        )
        seen.append((bytes(line), overrun))

    # drive monitor_stream but intercept lines via a collecting on_line is not
    # possible for overrun chunks (they skip on_line), so patch display path:
    # display=True routes every chunk (incl. overrun) through
    # sys.stdout.buffer.write(line), which is where the TypeError fired.
    import sys

    class _FakeBuffer:
        def write(self, data: bytes) -> int:
            display_like_handler(data, True)
            return len(data)

    class _FakeStdout:
        buffer = _FakeBuffer()

        def __getattr__(self, name):
            return getattr(sys.__stdout__, name)

    real_stdout = sys.stdout
    sys.stdout = _FakeStdout()  # type: ignore[assignment]
    try:
        result = await monitor_stream(stream, collect=True, display=True)
    finally:
        sys.stdout = real_stdout

    # the oversized chunk must have been displayed as bytes, not dropped
    displayed = b"".join(chunk for chunk, _ in seen)
    assert payload in displayed, (
        f"oversized chunk was dropped; displayed only {displayed!r}"
    )
    # overrun chunks are collected too, so returned output is complete
    assert bytes(result) == payload


async def test_monitor_stream_overrun_with_separator_after_limit() -> None:
    """Separator found beyond the limit: full output is collected."""
    limit = 8
    stream = asyncio.StreamReader(limit=limit)
    long_line = b"y" * (limit * 3) + b"\n"
    stream.feed_data(long_line + b"tail\n")
    stream.feed_eof()

    collected = await monitor_stream(stream, collect=True, display=False)
    # the oversized line and the data after it are both collected
    assert collected is not None
    assert bytes(collected) == long_line + b"tail\n"


async def test_monitor_stream_collect_preserves_oversized_chunks() -> None:
    """Regression test: `collect=True` must not silently drop oversized lines.

    Previously `handle()` returned on overrun before `ba.extend(line)`, so
    `subp_exec(..., collect=True, verbose=False)` lost every line longer
    than the stream limit.
    """
    limit = 8
    stream = asyncio.StreamReader(limit=limit)
    stream.feed_data(b"x" * 24 + b"\n" + b"tail\n")
    stream.feed_eof()

    collected = await monitor_stream(stream, collect=True, display=False)
    assert collected is not None
    out = bytes(collected)
    assert out.count(b"x") == 24, f"lost oversized data: {out!r}"
    assert b"tail\n" in out


async def test_monitor_stream_plain_lines_collected() -> None:
    stream = asyncio.StreamReader(limit=64)
    stream.feed_data(b"hello\nworld\n")
    stream.feed_eof()

    collected = await monitor_stream(stream, collect=True, display=False)
    assert bytes(collected) == b"hello\nworld\n"
