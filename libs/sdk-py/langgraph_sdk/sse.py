"""Adapted from httpx_sse to split lines on \n, \r, \r\n per the SSE spec."""

from typing import AsyncIterator, Iterator, Optional, Union

import httpx
import orjson

from langgraph_sdk.schema import StreamPart

BytesLike = Union[bytes, bytearray, memoryview]


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
            self.buffer.extend(lines[0])
            return []

        if self.buffer:
            # Include any existing buffer in the first portion of the
            # splitlines result.
            self.buffer.extend(lines[0])
            lines = [self.buffer] + lines[1:]
            self.buffer = bytearray()

        if not trailing_newline:
            # If the last segment of splitlines is not newline terminated,
            # then drop it from our output and start a new buffer.
            self.buffer.extend(lines.pop())

        return lines

    def flush(self) -> list[BytesLike]:
        if not self.buffer and not self.trailing_cr:
            return []

        lines = [self.buffer]
        self.buffer = bytearray()
        self.trailing_cr = False
        return lines


class SSEDecoder:
    def __init__(self) -> None:
        self._event = ""
        self._data = bytearray()
        self._last_event_id = ""
        self._retry: Optional[int] = None

    def decode(self, line: bytes) -> Optional[StreamPart]:
        # See: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation  # noqa: E501

        if not line:
            if (
                not self._event
                and not self._data
                and not self._last_event_id
                and self._retry is None
            ):
                return None

            sse = StreamPart(
                event=self._event,
                data=orjson.loads(self._data) if self._data else None,
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
            self._event = value.decode()
        elif fieldname == b"data":
            self._data.extend(value)
        elif fieldname == b"id":
            if b"\0" in value:
                pass
            else:
                self._last_event_id = value.decode()
        elif fieldname == b"retry":
            try:
                self._retry = int(value)
            except (TypeError, ValueError):
                pass
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
