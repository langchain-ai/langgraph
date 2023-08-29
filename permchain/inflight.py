from langchain.schema.runnable.config import RunnableConfig

from permchain.topic import InflightMessage
from langchain.schema.runnable import Runnable, RunnableConfig


# TODO check for longer cycle patterns
# ie. currently this finds cycles only if the same topic is seen twice in a row
# we also want to catch cycles like ABC ABC ABC, up to a configurable limit
class CycleMonitor(Runnable[InflightMessage, None]):
    def __init__(self) -> None:
        self.cycle_count = 0
        self.last_topic_seen = None

    def invoke(
        self, input: InflightMessage, config: RunnableConfig | None = None
    ) -> None:
        config = config or {}
        if (
            self.last_topic_seen is not None
            and self.last_topic_seen == input["topic_name"]
        ):
            self.cycle_count += 1
            if self.cycle_count > config["recursion_limit"]:
                raise RecursionError(
                    f"Found a cycle for topic '{self.last_topic_seen}', exiting after {self.cycle_count} iterations"
                )
        else:
            self.cycle_count = 0
            self.last_topic_seen = input["topic_name"]
