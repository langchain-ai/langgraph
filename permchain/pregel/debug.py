from pprint import pformat
from typing import Any

from langchain.schema.runnable import Runnable
from langchain.utils.input import get_bolded_text, get_colored_text

from permchain.pregel.constants import CHAINS_MAIN


def print_step_start(step: int, next_tasks: list[tuple[Runnable, Any, str]]) -> None:
    n_tasks = len(next_tasks)
    print(
        f"{get_colored_text('[pregel/step]', color='blue')} "
        + get_bolded_text(
            f"Starting step {step} with {n_tasks} task{'s' if n_tasks > 1 else ''}. Next tasks:\n"
        )
        + pformat(
            {name if name != CHAINS_MAIN else _main: val for _, val, name in next_tasks}
        )
    )


class MainChainStr:
    def __repr__(cls) -> str:
        return "main"


_main = MainChainStr()
