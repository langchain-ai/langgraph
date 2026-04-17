import sys
import threading
import time
from collections.abc import Callable


class Progress:
    delay: float = 0.1

    @staticmethod
    def spinning_cursor():
        while True:
            yield from "|/-\\"

    def __init__(self, *, message="", elapsed: bool = False):
        self.message = message
        self._base_message = message
        self._show_elapsed = elapsed
        # use this to make sure we don't kill thread when we set msg to ""
        self._stop = threading.Event()
        # signalled when the spinner has no text on screen
        self._line_clear = threading.Event()
        self._line_clear.set()
        self.spinner_generator = self.spinning_cursor()

    def spinner_iteration(self):
        message = self.message
        sys.stdout.write(next(self.spinner_generator) + " " + message)
        sys.stdout.flush()
        time.sleep(self.delay)
        # clear the spinner and message
        sys.stdout.write(
            "\b" * (len(message) + 2)
            + " " * (len(message) + 2)
            + "\b" * (len(message) + 2)
        )
        sys.stdout.flush()

    def _format_elapsed(self, seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        if mins:
            return f"{self._base_message} ({mins}m {secs:02d}s)"
        return f"{self._base_message} ({secs}s)"

    def spinner_task(self):
        start = time.monotonic()
        while not self._stop.is_set():
            if not self.message:
                self._line_clear.set()
                time.sleep(self.delay)
                continue
            if self._show_elapsed:
                self.message = self._format_elapsed(time.monotonic() - start)
            message = self.message
            if not message:
                self._line_clear.set()
                continue
            self._line_clear.clear()
            sys.stdout.write(next(self.spinner_generator) + " " + message)
            sys.stdout.flush()
            time.sleep(self.delay)
            # clear the spinner and message
            sys.stdout.write(
                "\b" * (len(message) + 2)
                + " " * (len(message) + 2)
                + "\b" * (len(message) + 2)
            )
            sys.stdout.flush()
            self._line_clear.set()

    def __enter__(self) -> Callable[[str], None]:
        if sys.stdout.isatty():
            self.thread = threading.Thread(target=self.spinner_task)
            self.thread.start()

            def set_message(message):
                self.message = message
                self._base_message = message or self._base_message
                if not message:
                    self._line_clear.wait(timeout=0.5)

            return set_message
        else:

            def set_message(message):
                if message:
                    sys.stderr.write(message + "\n")
                    sys.stderr.flush()

            return set_message

    def __exit__(self, exception, value, tb):
        if sys.stdout.isatty():
            self.message = ""
            self._stop.set()
            try:
                self.thread.join()
            finally:
                del self.thread
            if exception is not None:
                return False
