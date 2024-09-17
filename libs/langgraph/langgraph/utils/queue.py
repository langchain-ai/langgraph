import asyncio
import queue
import sys
import threading
import types
from collections import deque
from time import monotonic

PY_310 = sys.version_info >= (3, 10)


class AsyncQueue(asyncio.Queue):
    """Async unbounded FIFO queue with a wait() method.

    Subclassed from asyncio.Queue, adding a wait() method."""

    async def wait(self):
        """If queue is empty, wait until an item is available.

        Copied from Queue.get(), removing the call to .get_nowait(),
        ie. this doesn't consume the item, just waits for it.
        """
        while self.empty():
            if PY_310:
                getter = self._get_loop().create_future()
            else:
                getter = self._loop.create_future()
            self._getters.append(getter)
            try:
                await getter
            except:
                getter.cancel()  # Just in case getter is not done yet.
                try:
                    # Clean self._getters from canceled getters.
                    self._getters.remove(getter)
                except ValueError:
                    # The getter could be removed from self._getters by a
                    # previous put_nowait call.
                    pass
                if not self.empty() and not getter.cancelled():
                    # We were woken up by put_nowait(), but can't take
                    # the call.  Wake up the next in line.
                    self._wakeup_next(self._getters)
                raise


class Semaphore(threading.Semaphore):
    """Semaphore subclass with a wait() method."""

    def wait(self, blocking: bool = True, timeout: float = None):
        """Block until the semaphore can be acquired, but don't acquire it."""
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        rc = False
        endtime = None
        with self._cond:
            while self._value == 0:
                if not blocking:
                    break
                if timeout is not None:
                    if endtime is None:
                        endtime = monotonic() + timeout
                    else:
                        timeout = endtime - monotonic()
                        if timeout <= 0:
                            break
                self._cond.wait(timeout)
            else:
                rc = True
        return rc


class SyncQueue:
    """Unbounded FIFO queue with a wait() method.
    Adapted from pure Python implementation of queue.SimpleQueue.
    """

    def __init__(self):
        self._queue = deque()
        self._count = Semaphore(0)

    def put(self, item, block=True, timeout=None):
        """Put the item on the queue.

        The optional 'block' and 'timeout' arguments are ignored, as this method
        never blocks.  They are provided for compatibility with the Queue class.
        """
        self._queue.append(item)
        self._count.release()

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until an item is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately
        available, else raise the Empty exception ('timeout' is ignored
        in that case).
        """
        if timeout is not None and timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        if not self._count.acquire(block, timeout):
            raise queue.Empty
        try:
            return self._queue.popleft()
        except IndexError:
            raise queue.Empty

    def wait(self, block=True, timeout=None):
        """If queue is empty, wait until an item maybe is available,
        but don't consume it.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        self._count.wait(block, timeout)

    def empty(self):
        """Return True if the queue is empty, False otherwise (not reliable!)."""
        return len(self._queue) == 0

    def qsize(self):
        """Return the approximate size of the queue (not reliable!)."""
        return len(self._queue)

    __class_getitem__ = classmethod(types.GenericAlias)
