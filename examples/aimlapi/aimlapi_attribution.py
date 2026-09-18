"""Attribution headers for LangGraph applications that call aimlapi.com.

LangGraph itself never issues an HTTP request to a model provider: it has no
provider registry and no LLM client code.  Requests are made by whichever
LangChain chat model instance you hand to your graph.  For aimlapi.com that is
``ChatAimlapi`` from the ``langchain-aimlapi`` package, which already ships its
own attribution headers.

This module exists because ``ChatAimlapi(default_headers=...)`` **replaces**
those built-in headers rather than merging with them, so an application that
sets any header of its own silently drops all attribution.  Use
:func:`aimlapi_default_headers` instead of building the dict by hand.

Rules encoded here:

* merge, never assign - the caller's headers win on a key clash;
* never mutate a shared constant - a new dict is built on every call;
* scope by origin - the headers are only emitted for aimlapi.com itself, so
  they cannot ride a request to another provider or to a proxy that fronts us.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

# Partner id for rebate attribution, registered for LangGraph.  A malformed or
# invented id is worse than none -
# the API never rejects the request over it, so a wrong value fails silently and
# earns nothing.  When an id is registered it must match
# ``^part_[A-Za-z0-9]{1,64}$`` (alphanumerics only, no dashes or underscores).
AIMLAPI_PARTNER_ID = "part_Nw323K1Ij8QtPrTvXWm5cs6G"

# The origin these headers are allowed to reach.
AIMLAPI_ORIGIN = "api.aimlapi.com"

# Host-project identification.  ``HTTP-Referer`` and ``X-Title`` name the
# *calling application* - LangGraph - not aimlapi.com.
LANGGRAPH_ATTRIBUTION_HEADERS: Mapping[str, str] = {
    "HTTP-Referer": "https://github.com/langchain-ai/langgraph",
    "X-Title": "LangGraph",
    "X-AIMLAPI-Source": "agent/langgraph",
}


def _base_attribution() -> dict[str, str]:
    """Headers already shipped by ``langchain-aimlapi``, if it is installed."""
    try:
        from langchain_aimlapi import AIMLAPI_HEADERS
    except ImportError:  # pragma: no cover - package is optional
        return {}
    # copy: the package exposes a module-level dict that must not be mutated
    return dict(AIMLAPI_HEADERS)


def aimlapi_default_headers(
    user_headers: Mapping[str, str] | None = None,
    *,
    base_url: str = "https://api.aimlapi.com/v1/",
) -> dict[str, str]:
    """Build the ``default_headers`` dict for a ``ChatAimlapi`` instance.

    Returns a **new** dict on every call.  Precedence, lowest to highest:

    1. attribution already provided by ``langchain-aimlapi``;
    2. LangGraph host-project identification;
    3. ``AIMLAPI_PARTNER_ID``, only when it is non-empty;
    4. ``user_headers`` - the caller always wins.

    If ``base_url`` does not point at aimlapi.com, only ``user_headers`` is
    returned: attribution must never be attached to another provider's request.
    """
    if urlsplit(base_url).hostname != AIMLAPI_ORIGIN:
        return dict(user_headers or {})

    headers = _base_attribution()
    headers.update(LANGGRAPH_ATTRIBUTION_HEADERS)
    if AIMLAPI_PARTNER_ID:
        headers["X-AIMLAPI-Partner-ID"] = AIMLAPI_PARTNER_ID
    headers.update(user_headers or {})
    return headers
