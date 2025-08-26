from collections.abc import Sequence
from inspect import signature
from typing import Callable, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from langgraph.agent.types import (
    AgentGoTo,
    AgentMiddleware,
    AgentState,
    AgentUpdate,
    GoTo,
    ModelRequest,
    ResponseFormat,
)
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.tool_node import ToolNode


def create_agent(
    *,
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable],
    system_prompt: str,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | None = None,
) -> StateGraph[AgentState, None, AgentUpdate]:
    # init chat model
    if isinstance(model, str):
        try:
            from langchain.chat_models import (  # type: ignore[import-not-found]
                init_chat_model,
            )
        except ImportError:
            raise ImportError(
                "Please install langchain (`pip install langchain`) to "
                "use '<provider>:<model>' string syntax for `model` parameter."
            )

        model = cast(BaseChatModel, init_chat_model(model))

    # init tool node
    tool_node = ToolNode(tools=tools)

    # validate middleware
    assert len({m.__class__.__name__ for m in middleware}) == len(middleware), (
        "Please remove duplicate middleware instances."
    )  # this is just to keep the node names simple, we can change if needed
    middleware_w_before = [
        m
        for m in middleware
        if m.__class__.before_model is not AgentMiddleware.before_model
    ]
    middleware_w_after = [
        m
        for m in middleware
        if m.__class__.after_model is not AgentMiddleware.after_model
    ]

    # create graph, add nodes
    graph = StateGraph(AgentState, input_schema=AgentUpdate, output_schema=AgentUpdate)
    graph.add_node(
        "model_request",
        _make_model_request_node(
            model=model,
            tools=list(tool_node.tools_by_name.values()),
            system_prompt=system_prompt,
            middleware=middleware,
            response_format=response_format,
        ),
    )
    graph.add_node("tools", tool_node)
    for m in middleware:
        if m.__class__.before_model is not AgentMiddleware.before_model:
            graph.add_node(f"{m.__class__.__name__}.before_model", m.before_model)
        if m.__class__.after_model is not AgentMiddleware.after_model:
            graph.add_node(f"{m.__class__.__name__}.after_model", m.after_model)

    # add start edge
    first_node = (
        f"{middleware_w_before[0].__class__.__name__}.before_model"
        if middleware_w_before
        else "model_request"
    )
    last_node = (
        f"{middleware_w_after[0].__class__.__name__}.after_model"
        if middleware_w_after
        else "model_request"
    )
    graph.add_edge(START, first_node)

    # add cond edges
    graph.add_conditional_edges(
        "tools",
        _make_tools_to_model_edge(tool_node, first_node),
        [first_node, END],
    )
    graph.add_conditional_edges(
        last_node, _make_model_to_tools_edge(first_node), ["tools", END]
    )

    # add before model edges
    if middleware_w_before:
        for m1, m2 in zip(middleware_w_before, middleware_w_before[1:]):
            _add_middleware_edge(
                graph,
                m1.before_model,
                f"{m1.__class__.__name__}.before_model",
                f"{m2.__class__.__name__}.before_model",
                first_node,
            )
        _add_middleware_edge(
            graph,
            middleware_w_before[-1].before_model,
            f"{middleware_w_before[-1].__class__.__name__}.before_model",
            "model_request",
            first_node,
        )

    # add after model edges
    if middleware_w_after:
        graph.add_edge(
            "model_request", f"{middleware_w_after[-1].__class__.__name__}.after_model"
        )
        for idx in range(len(middleware_w_after) - 1, 0, -1):
            m1 = middleware_w_after[idx]
            m2 = middleware_w_after[idx - 1]
            _add_middleware_edge(
                graph,
                m1.after_model,
                f"{m1.__class__.__name__}.after_model",
                f"{m2.__class__.__name__}.after_model",
                first_node,
            )

    return graph


def _make_model_request_node(
    *,
    system_prompt: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    middleware: Sequence[AgentMiddleware],
    response_format: ResponseFormat | None = None,
) -> Callable[[AgentState], AgentState]:
    def model_request(state: AgentState) -> AgentState:
        # create request
        request = ModelRequest(
            model=model,
            system_prompt=system_prompt,
            messages=state.messages,
            tool_choice=None,
            tools=tools,
            response_format=response_format,
        )
        # visit middleware in order
        for mw in middleware:
            request = mw.modify_model_request(request, state)
            # TODO assert request.tools in tools, or pass them to tool node
        # prepare messages
        if request.system_prompt:
            messages = [SystemMessage(request.system_prompt)] + request.messages
        else:
            messages = request.messages
        # prepare model
        if request.response_format:
            model_ = request.model.with_structured_output(request.response_format)
        else:
            model_ = request.model
        # call model
        output = model_.invoke(
            messages, tools=request.tools, tool_choice=request.tool_choice
        )
        return {"messages": output}

    return model_request


def _resolve_goto(goto: GoTo | None, first_node: str) -> str | None:
    if goto == "model":
        return first_node
    elif goto:
        return goto


def _make_model_to_tools_edge(first_node: str) -> Callable[[AgentState], str | None]:
    def model_to_tools(state: AgentState) -> str | None:
        if state.goto:
            return _resolve_goto(state.goto, first_node)
        message = state.messages[-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            return "tools"

        return END

    return model_to_tools


def _make_tools_to_model_edge(
    tool_node: ToolNode, next_node: str
) -> Callable[[AgentState], str | None]:
    def tools_to_model(state: AgentState) -> str | None:
        ai_message = [m for m in state.messages if isinstance(m, AIMessage)][-1]
        if all(
            tool_node.tools_by_name[c["name"]].return_direct
            for c in ai_message.tool_calls
            if c["name"] in tool_node.tools_by_name
        ):
            return END

        return next_node

    return tools_to_model


def _add_middleware_edge(
    graph: StateGraph,
    method: Callable[[AgentState], AgentUpdate | AgentGoTo | None],
    name: str,
    default_destination: str,
    model_destination: str,
) -> None:
    sig = signature(method)
    uses_goto = sig.return_annotation is AgentGoTo or AgentGoTo in getattr(
        sig.return_annotation, "__args__", ()
    )

    if uses_goto:

        def goto_edge(state: AgentState) -> str:
            return _resolve_goto(state.goto, model_destination) or default_destination

        destinations = [default_destination, END, "tools"]
        if name != model_destination:
            destinations.append(model_destination)

        graph.add_conditional_edges(name, goto_edge, destinations)
    else:
        graph.add_edge(name, default_destination)
