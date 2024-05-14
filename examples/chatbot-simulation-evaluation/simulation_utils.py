import functools
from typing import Annotated, Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables import chain as as_runnable
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph
from langchain_community.adapters.openai import convert_message_to_dict


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
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ) | (llm or ChatOpenAI(model="gpt-3.5-turbo")).with_config(
        run_name="simulated_user"
    )


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
    graph_builder.set_entry_point("assistant" if input_key is not None else "user")

    return (
        RunnableLambda(_prepare_example).bind(input_key=input_key)
        | graph_builder.compile()
    )


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
