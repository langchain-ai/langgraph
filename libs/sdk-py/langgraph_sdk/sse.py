"""Adapted from httpx_sse to split lines on \n, \r, \r\n per the SSE spec."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import cast

import httpx
import orjson

from langgraph_sdk.schema import StreamPart

BytesLike = bytes | bytearray | memoryview


class BytesLineDecoder:
    """
    Handles incrementally reading lines from text.

    Has the same behaviour as the stdllib bytes splitlines,
    but handling the input iteratively.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.trailing_cr: bool = False

    def decode(self, text: bytes) -> list[BytesLike]:
        # See https://docs.python.org/3/glossary.html#term-universal-newlines
        NEWLINE_CHARS = b"\n\r"

        if not text:
            return []
        if self.trailing_cr and text.startswith(b"\n"):
            text = text[1:]
        self.trailing_cr = text.endswith(b"\r")

        if not text:
            return []

        trailing_newline = text[-1] in NEWLINE_CHARS
        lines = cast(list[BytesLike], text.splitlines())

        if len(lines) == 1 and not trailing_newline:
            # No new lines, buffer the input and continue.
            self.buffer.extend(lines[0])
            return []

        if self.buffer:
            # Include any existing buffer in the first portion of the
            # splitlines result.
            self.buffer.extend(lines[0])
            lines = [self.buffer, *lines[1:]]
            self.buffer = bytearray()

        if not trailing_newline:
            # If the last segment of splitlines is not newline terminated,
            # then drop it from our output and start a new buffer.
            self.buffer.extend(lines.pop())

        return lines

    def flush(self) -> list[BytesLike]:
        if not self.buffer:
            return []

        lines: list[BytesLike] = [self.buffer]
        self.buffer = bytearray()
        self.trailing_cr = False
        return lines


class SSEDecoder:
    def __init__(self) -> None:
        self._event = ""
        self._data = bytearray()
        self._last_event_id: str | None = None
        self._retry: int | None = None
        self._at_stream_start = True
        self._has_data = False

    @property
    def last_event_id(self) -> str | None:
        """Return the last event identifier that was seen.

        `None` until the stream sends an `id` field, and `""` after an empty
        one, which resets the reconnect cursor.
        """

        return self._last_event_id

    def decode(self, line: bytes) -> StreamPart | None:
        # See: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation

        if self._at_stream_start:
            self._at_stream_start = False
            line = line.removeprefix(b"\xef\xbb\xbf")

        if not line:
            self._has_data = bool(self._data)
            if (
                not self._event
                and not self._data
                and self._last_event_id is None
                and self._retry is None
            ):
                return None

            body = self._data[:-1]
            data = None
            if body and not body.isspace():
                try:
                    data = orjson.loads(body)
                except orjson.JSONDecodeError:
                    data = orjson.loads(body.decode(errors="replace"))

            sse = StreamPart(
                event=self._event,
                data=data,  # ty: ignore[invalid-argument-type]
                id=self._last_event_id or None,
            )

            # NOTE: as per the SSE spec, do not reset last_event_id.
            self._event = ""
            self._data = bytearray()
            self._retry = None

            return sse

        if line.startswith(b":"):
            return None

        fieldname, _, value = line.partition(b":")

        if value.startswith(b" "):
            value = value[1:]

        if fieldname == b"event":
            self._event = value.decode(errors="replace")
        elif fieldname == b"data":
            self._data.extend(value)
            self._data.extend(b"\n")
        elif fieldname == b"id":
            if b"\0" in value:
                pass
            else:
                self._last_event_id = value.decode(errors="replace")
        elif fieldname == b"retry":
            if value and value.isdigit():
                with contextlib.suppress(ValueError):
                    self._retry = int(value)
        else:
            pass  # Field is ignored.

        return None


async def aiter_lines_raw(response: httpx.Response) -> AsyncIterator[BytesLike]:
    decoder = BytesLineDecoder()
    async for chunk in response.aiter_bytes():
        for line in decoder.decode(chunk):
            yield line
    for line in decoder.flush():
        yield line


def iter_lines_raw(response: httpx.Response) -> Iterator[BytesLike]:
    decoder = BytesLineDecoder()
    for chunk in response.iter_bytes():
        for line in decoder.decode(chunk):
            yield line
    for line in decoder.flush():
        yield line
