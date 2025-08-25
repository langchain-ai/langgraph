from collections.abc import Sequence
from typing import Callable, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from langgraph.agent.types import AgentInput, AgentMiddleware, AgentState, ModelRequest
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.tool_node import ToolNode


def create_agent(
    *,
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable],
    system_prompt: str,
    middleware: Sequence[AgentMiddleware] = (),
) -> StateGraph[AgentState, None, AgentInput]:
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
    graph = StateGraph(AgentState, input_schema=AgentInput)
    graph.add_node(
        "model_request",
        _make_model_request_node(
            model=model,
            tools=list(tool_node.tools_by_name.values()),
            system_prompt=system_prompt,
            middleware=middleware,
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
    graph.add_conditional_edges(last_node, _make_model_to_tools_edge(), ["tools", END])

    # add before model edges
    if middleware_w_before:
        for m1, m2 in zip(middleware_w_before, middleware_w_before[1:]):
            graph.add_edge(
                f"{m1.__class__.__name__}.before_model",
                f"{m2.__class__.__name__}.before_model",
            )
        graph.add_edge(
            f"{middleware_w_before[-1].__class__.__name__}.before_model",
            "model_request",
        )

    # add after model edges
    if middleware_w_after:
        graph.add_edge(
            "model_request", f"{middleware_w_after[-1].__class__.__name__}.after_model"
        )
        for idx in range(len(middleware_w_after) - 1, 0, -1):
            m1 = middleware_w_after[idx]
            m2 = middleware_w_after[idx - 1]
            graph.add_edge(
                f"{m1.__class__.__name__}.after_model",
                f"{m2.__class__.__name__}.after_model",
            )

    return graph


def _make_model_request_node(
    *,
    system_prompt: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    middleware: Sequence[AgentMiddleware],
) -> Callable[[AgentState], AgentState]:
    def model_request(state: AgentState) -> AgentState:
        # create request
        request = ModelRequest(
            model=model,
            system_prompt=system_prompt,
            messages=state.messages,
            tool_choice=None,
            tools=tools,
        )
        # visit middleware in order
        for mw in middleware:
            request = mw.modify_model_request(request)
        # prepare messages
        if request.system_prompt:
            messages = [SystemMessage(request.system_prompt)] + request.messages
        else:
            messages = request.messages
        # call model
        output = request.model.invoke(
            messages, tools=request.tools, tool_choice=request.tool_choice
        )
        return {"messages": output}

    return model_request


def _make_model_to_tools_edge() -> Callable[[AgentState], str | None]:
    def model_to_tools(state: AgentState) -> str | None:
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
