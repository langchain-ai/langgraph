"""Adapted from httpx_sse to split lines on \n, \r, \r\n per the SSE spec."""

import io
from typing import AsyncIterator, Iterator

import httpx
import httpx_sse
import httpx_sse._decoders


class BytesLineDecoder:
    """
    Handles incrementally reading lines from text.

    Has the same behaviour as the stdllib bytes splitlines,
    but handling the input iteratively.
    """

    def __init__(self) -> None:
        self.buffer = io.BytesIO()
        self.trailing_cr: bool = False

    def decode(self, text: bytes) -> list[bytes]:
        # See https://docs.python.org/3/glossary.html#term-universal-newlines
        NEWLINE_CHARS = b"\n\r"

        # We always push a trailing `\r` into the next decode iteration.
        if self.trailing_cr:
            text = b"\r" + text
            self.trailing_cr = False
        if text.endswith(b"\r"):
            self.trailing_cr = True
            text = text[:-1]

        if not text:
            # NOTE: the edge case input of empty text doesn't occur in practice,
            # because other httpx internals filter out this value
            return []  # pragma: no cover

        trailing_newline = text[-1] in NEWLINE_CHARS
        lines = text.splitlines()

        if len(lines) == 1 and not trailing_newline:
            # No new lines, buffer the input and continue.
            self.buffer.write(lines[0])
            return []

        if self.buffer:
            # Include any existing buffer in the first portion of the
            # splitlines result.
            lines = [self.buffer.getvalue() + lines[0]] + lines[1:]
            self.buffer.truncate(0)

        if not trailing_newline:
            # If the last segment of splitlines is not newline terminated,
            # then drop it from our output and start a new buffer.
            self.buffer.write(lines.pop())

        return lines

    def flush(self) -> list[bytes]:
        if not self.buffer and not self.trailing_cr:
            return []

        lines = [self.buffer.getvalue()] if self.buffer else []
        self.buffer.truncate(0)
        self.trailing_cr = False
        return lines


async def aiter_lines_raw(response: httpx.Response) -> AsyncIterator[bytes]:
    decoder = BytesLineDecoder()
    async for chunk in response.aiter_bytes():
        for line in decoder.decode(chunk):
            yield line
    for line in decoder.flush():
        yield line


def iter_lines_raw(response: httpx.Response) -> Iterator[bytes]:
    decoder = BytesLineDecoder()
    for chunk in response.iter_bytes():
        for line in decoder.decode(chunk):
            yield line
    for line in decoder.flush():
        yield line


class EventSource(httpx_sse.EventSource):
    async def aiter_sse(self) -> AsyncIterator[httpx_sse.ServerSentEvent]:
        self._check_content_type()
        decoder = httpx_sse._decoders.SSEDecoder()
        async for line in aiter_lines_raw(self._response):
            line = line.rstrip(b"\n")
            sse = decoder.decode(line.decode())
            if sse is not None:
                yield sse

    def iter_sse(self) -> Iterator[httpx_sse.ServerSentEvent]:
        self._check_content_type()
        decoder = httpx_sse._decoders.SSEDecoder()
        for line in iter_lines_raw(self._response):
            line = line.rstrip(b"\n")
            sse = decoder.decode(line.decode())
            if sse is not None:
                yield sse
