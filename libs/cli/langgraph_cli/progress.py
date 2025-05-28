import sys
import threading
import time
from typing import Callable


class Progress:
    delay: float = 0.1

    @staticmethod
    def spinning_cursor():
        while True:
            yield from "|/-\\"

    def __init__(self, *, message=""):
        self.message = message
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

    def spinner_task(self):
        while self.message:
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

    def __enter__(self) -> Callable[[str], None]:
        if sys.stdout.isatty():
            self.thread = threading.Thread(target=self.spinner_task)
            self.thread.start()

            def set_message(message):
                self.message = message
                if not message:
                    self.thread.join()

            return set_message
        else:

            def set_message(message):
                sys.stderr.write(message + "\n")
                sys.stderr.flush()

            return set_message

    def __exit__(self, exception, value, tb):
        if sys.stdout.isatty():
            self.message = ""
            try:
                self.thread.join()
            finally:
                del self.thread
            if exception is not None:
                return False
