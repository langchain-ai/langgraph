from __future__ import annotations

from typing import Any


def run_server(
    graph: Any,
    *,
    host: str,
    port: int,
    config: dict[str, Any],
) -> None:
    """Wire *graph* + its `agent_card` into an A2A-compliant Starlette app.

    Only imported when the ``langgraph[a2a]`` extra is installed.
    """
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    card_dict = graph.agent_card.to_dict()

    async def agent_card_endpoint(request: Request) -> Response:
        return JSONResponse(card_dict)

    async def task_endpoint(request: Request) -> Response:
        body = await request.json()
        user_message = _extract_user_message(body)
        thread_id = body.get("id", "default")

        run_config: dict[str, Any] = {**config}
        if graph.checkpointer is not None:
            run_config.setdefault("configurable", {})["thread_id"] = thread_id

        result_parts: list[str] = []
        async for event in graph.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=run_config,
            stream_mode="values",
        ):
            msgs = event.get("messages", [])
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", "")
                if content:
                    result_parts.append(str(content))

        return JSONResponse(_build_a2a_response(body.get("id"), result_parts))

    app = Starlette(
        routes=[
            Route(
                "/.well-known/agent.json",
                agent_card_endpoint,
                methods=["GET"],
            ),
            Route("/", task_endpoint, methods=["POST"]),
        ]
    )

    uvicorn.run(app, host=host, port=port)


def _extract_user_message(body: dict[str, Any]) -> str:
    """Extract text from an A2A ``tasks/send`` request body."""
    try:
        return str(body["message"]["parts"]["text"])
    except (KeyError, IndexError, TypeError):
        return str(body)


def _build_a2a_response(task_id: str | None, parts: list[str]) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": {"state": "completed"},
        "artifacts": [{"parts": [{"type": "text", "text": p} for p in parts]}],
    }
