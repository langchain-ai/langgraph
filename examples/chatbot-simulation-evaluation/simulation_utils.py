import functools
import logging
import os
from typing import Annotated, Any, Callable, Dict, List, Optional, Union

from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables import chain as as_runnable
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START

try:  # Optional dependency
    from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
except ImportError:  # pragma: no cover
    AzureAIOpenTelemetryTracer = None  # type: ignore


logger = logging.getLogger(__name__)

_TRACING_CALLBACKS: Optional[List[BaseCallbackHandler]] = None


def _get_tracing_callbacks() -> List[BaseCallbackHandler]:
    """Initialise Azure tracing callbacks once and reuse."""
    global _TRACING_CALLBACKS
    if _TRACING_CALLBACKS is not None:
        return _TRACING_CALLBACKS

    callbacks: List[BaseCallbackHandler] = []
    connection_string = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
    enable_content_recording = (
        os.getenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true").lower()
        == "true"
    )

    if connection_string:
        if AzureAIOpenTelemetryTracer:
            tracer_name = os.getenv("CHATBOT_SIM_TRACER_NAME", "chatbot-simulation")
            tracer_id = os.getenv("CHATBOT_SIM_TRACER_ID", tracer_name)
            callbacks.append(
                AzureAIOpenTelemetryTracer(
                    connection_string=connection_string,
                    enable_content_recording=enable_content_recording,
                    name=tracer_name,
                    id=tracer_id,
                )
            )
            logger.info(
                "AzureAIOpenTelemetryTracer initialised for chatbot simulation runs."
            )
        else:  # pragma: no cover - dependency missing at runtime
            logger.warning(
                "APPLICATION_INSIGHTS_CONNECTION_STRING is set but "
                "`langchain-azure-ai` is not installed; Azure tracing disabled."
            )
    else:
        logger.debug("Azure tracing disabled; APPLICATION_INSIGHTS_CONNECTION_STRING unset.")

    _TRACING_CALLBACKS = callbacks
    return _TRACING_CALLBACKS


def langchain_to_openai_messages(messages: List[BaseMessage]):
    """
    Convert a list of langchain base messages to a list of openai messages.

    Parameters:
        messages (List[BaseMessage]): A list of langchain base messages.

    Returns:
        List[dict]: A list of openai messages.
    """

    return [
        convert_message_to_dict(m) if isinstance(m, BaseMessage) else m
        for m in messages
    ]


def create_simulated_user(
    system_prompt: str, llm: Runnable | None = None
) -> Runnable[Dict, AIMessage]:
    """
    Creates a simulated user for chatbot simulation.

    Args:
        system_prompt (str): The system prompt to be used by the simulated user.
        llm (Runnable | None, optional): The language model to be used for the simulation.
            Defaults to gpt-3.5-turbo.

    Returns:
        Runnable[Dict, AIMessage]: The simulated user for chatbot simulation.
    """
    callbacks = _get_tracing_callbacks()
    llm = llm or ChatOpenAI(model="gpt-3.5-turbo")
    llm_config: Dict[str, Any] = {"run_name": "simulated_user"}
    if callbacks:
        llm_config.update(
            {
                "callbacks": callbacks,
                "tags": ["chatbot-simulation", "simulated-user"],
                "metadata": {
                    "component": "simulation_utils",
                    "role": "simulated_user",
                },
            }
        )
    llm = llm.with_config(llm_config)

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ) | llm.with_config(run_name="simulated_user")


Messages = Union[list[AnyMessage], AnyMessage]


def add_messages(left: Messages, right: Messages) -> Messages:
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


class SimulationState(TypedDict):
    """
    Represents the state of a simulation.

    Attributes:
        messages (List[AnyMessage]): A list of messages in the simulation.
        inputs (Optional[dict[str, Any]]): Optional inputs for the simulation.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    inputs: Optional[dict[str, Any]]


def create_chat_simulator(
    assistant: (
        Callable[[List[AnyMessage]], str | AIMessage]
        | Runnable[List[AnyMessage], str | AIMessage]
    ),
    simulated_user: Runnable[Dict, AIMessage],
    *,
    input_key: str,
    max_turns: int = 6,
    should_continue: Optional[Callable[[SimulationState], str]] = None,
):
    """Creates a chat simulator for evaluating a chatbot.

    Args:
        assistant: The chatbot assistant function or runnable object.
        simulated_user: The simulated user object.
        input_key: The key for the input to the chat simulation.
        max_turns: The maximum number of turns in the chat simulation. Default is 6.
        should_continue: Optional function to determine if the simulation should continue.
            If not provided, a default function will be used.

    Returns:
        The compiled chat simulation graph.

    """
    graph_builder = StateGraph(SimulationState)
    graph_builder.add_node(
        "user",
        _create_simulated_user_node(simulated_user),
    )
    graph_builder.add_node(
        "assistant", _fetch_messages | assistant | _coerce_to_message
    )
    graph_builder.add_edge("assistant", "user")
    graph_builder.add_conditional_edges(
        "user",
        should_continue or functools.partial(_should_continue, max_turns=max_turns),
    )
    # If your dataset has a 'leading question/input', then we route first to the assistant, otherwise, we let the user take the lead.
    graph_builder.add_edge(START, "assistant" if input_key is not None else "user")

    simulation_graph = graph_builder.compile()
    callbacks = _get_tracing_callbacks()
    runnable = RunnableLambda(_prepare_example).bind(input_key=input_key)
    pipeline: Runnable = runnable | simulation_graph
    if callbacks:
        pipeline = pipeline.with_config(
            {
                "callbacks": callbacks,
                "tags": ["chatbot-simulation"],
                "metadata": {"component": "simulation_utils", "role": "pipeline"},
            }
        )
    return pipeline


## Private methods


def _prepare_example(inputs: dict[str, Any], input_key: Optional[str] = None):
    if input_key is not None:
        if input_key not in inputs:
            raise ValueError(
                f"Dataset's example input must contain the provided input key: '{input_key}'.\nFound: {list(inputs.keys())}"
            )
        messages = [HumanMessage(content=inputs[input_key])]
        return {
            "inputs": {k: v for k, v in inputs.items() if k != input_key},
            "messages": messages,
        }
    return {"inputs": inputs, "messages": []}


def _invoke_simulated_user(state: SimulationState, simulated_user: Runnable):
    """Invoke the simulated user node."""
    runnable = (
        simulated_user
        if isinstance(simulated_user, Runnable)
        else RunnableLambda(simulated_user)
    )
    inputs = state.get("inputs", {})
    inputs["messages"] = state["messages"]
    return runnable.invoke(inputs)


def _swap_roles(state: SimulationState):
    new_messages = []
    for m in state["messages"]:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return {
        "inputs": state.get("inputs", {}),
        "messages": new_messages,
    }


@as_runnable
def _fetch_messages(state: SimulationState):
    """Invoke the simulated user node."""
    return state["messages"]


def _convert_to_human_message(message: BaseMessage):
    return {"messages": [HumanMessage(content=message.content)]}


def _create_simulated_user_node(simulated_user: Runnable):
    """Simulated user accepts a {"messages": [...]} argument and returns a single message."""
    return (
        _swap_roles
        | RunnableLambda(_invoke_simulated_user).bind(simulated_user=simulated_user)
        | _convert_to_human_message
    )


def _coerce_to_message(assistant_output: str | BaseMessage):
    if isinstance(assistant_output, str):
        return {"messages": [AIMessage(content=assistant_output)]}
    else:
        return {"messages": [assistant_output]}


def _should_continue(state: SimulationState, max_turns: int = 6):
    messages = state["messages"]
    # TODO support other stop criteria
    if len(messages) > max_turns:
        return END
    elif messages[-1].content.strip() == "FINISHED":
        return END
    else:
        return "assistant"
