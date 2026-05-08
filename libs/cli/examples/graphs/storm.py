import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timezone
from typing import Annotated
from urllib.parse import urlparse

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables import chain as as_runnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Logging / audit infrastructure
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
audit_logger = logging.getLogger("storm.audit")

# Correlation / session identifier for the current run
_SESSION_ID: str = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Approved model registry (organisation-approved models only)
# ---------------------------------------------------------------------------
APPROVED_MODEL_REGISTRY: dict[str, str] = {
    # model-alias -> pinned version identifier
    "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
}

APPROVED_FAST_MODEL = "claude-3-5-haiku-20241022"
APPROVED_LONG_CONTEXT_MODEL = "claude-3-5-sonnet-20241022"

# ---------------------------------------------------------------------------
# Approved tool allow-list
# ---------------------------------------------------------------------------
APPROVED_TOOLS: set[str] = {"search_engine", "wikipedia_retriever"}

# ---------------------------------------------------------------------------
# URL allow-list for outbound HTTP
# ---------------------------------------------------------------------------
ALLOWED_URL_HOSTNAMES: set[str] = {
    "en.wikipedia.org",
    "api.tavily.com",
}

# ---------------------------------------------------------------------------
# Dangerous output patterns (eval / dynamic code execution primitives)
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\b", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bcompile\s*\(", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Prompt injection / malicious command patterns
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:[A-Za-z0-9+/]{4}){4,}={0,2}"),  # base64 blocks
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"<\s*script", re.IGNORECASE),
    re.compile(r"\$\(.*\)"),  # shell command substitution
    re.compile(r"`[^`]+`"),   # backtick execution
]

# ---------------------------------------------------------------------------
# Privilege escalation patterns
# ---------------------------------------------------------------------------
_ESCALATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bchmod\b", re.IGNORECASE),
    re.compile(r"\bchown\b", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\badmin\b.*\bpassword\b", re.IGNORECASE),
    re.compile(r"\bgrant\b.*\bprivilege", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Simple in-process authentication token store
# ---------------------------------------------------------------------------
_VALID_TOKENS: set[str] = set()


def generate_agent_token() -> str:
    """Generate and register a new authentication token."""
    token = secrets.token_hex(32)
    _VALID_TOKENS.add(token)
    return token


def authenticate(token: str) -> bool:
    """Verify that a token is valid before allowing agent access."""
    return token in _VALID_TOKENS


# ---------------------------------------------------------------------------
# Helper: verify model is in approved registry
# ---------------------------------------------------------------------------
def _assert_model_approved(model_id: str) -> str:
    if model_id not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_id}' is not in the organisation's approved model registry. "
            f"Approved models: {list(APPROVED_MODEL_REGISTRY.keys())}"
        )
    pinned = APPROVED_MODEL_REGISTRY[model_id]
    audit_logger.info(
        "model_registry_check",
        extra={
            "session_id": _SESSION_ID,
            "model_requested": model_id,
            "model_pinned": pinned,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return pinned


# ---------------------------------------------------------------------------
# Helper: validate / sanitize text input before sending to LLM
# ---------------------------------------------------------------------------
def sanitize_input(text: str, field_name: str = "input") -> str:
    """Raise ValueError if text contains injection or escalation patterns."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str for {field_name}, got {type(text)}")
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            audit_logger.warning(
                "input_injection_detected",
                extra={
                    "session_id": _SESSION_ID,
                    "field": field_name,
                    "pattern": pattern.pattern,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise ValueError(
                f"Potentially malicious content detected in {field_name}."
            )
    for pattern in _ESCALATION_PATTERNS:
        if pattern.search(text):
            audit_logger.warning(
                "privilege_escalation_detected",
                extra={
                    "session_id": _SESSION_ID,
                    "field": field_name,
                    "pattern": pattern.pattern,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise ValueError(
                f"Privilege escalation attempt detected in {field_name}."
            )
    return text


# ---------------------------------------------------------------------------
# Helper: validate LLM output for dangerous primitives
# ---------------------------------------------------------------------------
def validate_llm_output(text: str, context: str = "llm_output") -> str:
    """Raise ValueError if LLM output contains dynamic code execution primitives."""
    if not isinstance(text, str):
        return text
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(text):
            audit_logger.warning(
                "dangerous_llm_output_detected",
                extra={
                    "session_id": _SESSION_ID,
                    "context": context,
                    "pattern": pattern.pattern,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise ValueError(
                f"LLM output contains potentially dangerous code execution primitive "
                f"in context '{context}'."
            )
    return text


# ---------------------------------------------------------------------------
# Helper: validate URL against allow-list
# ---------------------------------------------------------------------------
def validate_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname not in ALLOWED_URL_HOSTNAMES:
        audit_logger.warning(
            "url_not_in_allowlist",
            extra={
                "session_id": _SESSION_ID,
                "url": url,
                "hostname": hostname,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise ValueError(
            f"URL hostname '{hostname}' is not in the allowed list: {ALLOWED_URL_HOSTNAMES}"
        )
    return url


# ---------------------------------------------------------------------------
# Helper: enforce tool allow-list
# ---------------------------------------------------------------------------
def assert_tool_allowed(tool_name: str) -> None:
    if tool_name not in APPROVED_TOOLS:
        audit_logger.warning(
            "tool_not_in_allowlist",
            extra={
                "session_id": _SESSION_ID,
                "tool": tool_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise ValueError(
            f"Tool '{tool_name}' is not in the approved tool allow-list: {APPROVED_TOOLS}"
        )
    audit_logger.info(
        "tool_invocation_allowed",
        extra={
            "session_id": _SESSION_ID,
            "tool": tool_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Helper: audit-log an LLM interaction
# ---------------------------------------------------------------------------
def log_llm_interaction(
    call_name: str,
    input_data,
    output_data,
    model_id: str,
) -> None:
    input_str = json.dumps(input_data, default=str)
    output_str = json.dumps(output_data, default=str)
    input_hash = hashlib.sha256(input_str.encode()).hexdigest()
    output_hash = hashlib.sha256(output_str.encode()).hexdigest()
    audit_logger.info(
        "llm_interaction",
        extra={
            "session_id": _SESSION_ID,
            "call_name": call_name,
            "model_id": model_id,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Helper: attach AI-generated content provenance label
# ---------------------------------------------------------------------------
_AI_PROVENANCE_HEADER = (
    "\n\n---\n"
    "**AI-Generated Content Notice**: This content was produced by an AI language model "
    "({model_id}) on {timestamp} (session: {session_id}). "
    "It has not been independently verified.\n"
    "---\n"
)


def attach_provenance(content: str, model_id: str) -> str:
    label = _AI_PROVENANCE_HEADER.format(
        model_id=model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=_SESSION_ID,
    )
    return content + label


# ---------------------------------------------------------------------------
# Validate approved models at startup
# ---------------------------------------------------------------------------
_assert_model_approved(APPROVED_FAST_MODEL)
_assert_model_approved(APPROVED_LONG_CONTEXT_MODEL)

# ---------------------------------------------------------------------------
# LLM instances — approved models only, imported via langchain_anthropic
# ---------------------------------------------------------------------------
try:
    from langchain_anthropic import ChatAnthropic  # type: ignore

    fast_llm = ChatAnthropic(model=APPROVED_FAST_MODEL)
    long_context_llm = ChatAnthropic(model=APPROVED_LONG_CONTEXT_MODEL)
    _perspectives_llm = ChatAnthropic(model=APPROVED_FAST_MODEL)
except ImportError:
    # Fallback: keep ChatOpenAI but flag that models are not approved —
    # the registry assertion above will have already raised if models are absent.
    fast_llm = ChatOpenAI(model=APPROVED_FAST_MODEL)
    long_context_llm = ChatOpenAI(model=APPROVED_LONG_CONTEXT_MODEL)
    _perspectives_llm = ChatOpenAI(model=APPROVED_FAST_MODEL)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Wikipedia writer. Write an outline for a Wikipedia page about a user-provided topic. Be comprehensive and specific.",
        ),
        ("user", "{topic}"),
    ]
)


class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: list[Subsection] | None = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: list[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


generate_outline_direct = direct_gen_outline_prompt | fast_llm.with_structured_output(
    Outline
)

gen_related_topics_prompt = ChatPromptTemplate.from_template(
    """I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.

Please list the as many subjects and urls as you can.

Topic of interest: {topic}
"""
)


class RelatedSubjects(BaseModel):
    topics: list[str] = Field(
        description="Comprehensive list of related subjects as background research.",
    )


expand_chain = gen_related_topics_prompt | fast_llm.with_structured_output(
    RelatedSubjects
)


class Editor(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the editor.",
    )
    name: str = Field(
        description="Name of the editor.",
    )
    role: str = Field(
        description="Role of the editor in the context of the topic.",
    )
    description: str = Field(
        description="Description of the editor's focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    editors: list[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )


gen_perspectives_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You need to select a diverse (and distinct) group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.\
    You can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on.

    Wiki page outlines of related topics for inspiration:
    {examples}""",
        ),
        ("user", "Topic of interest: {topic}"),
    ]
)

gen_perspectives_chain = gen_perspectives_prompt | _perspectives_llm.with_structured_output(Perspectives)


wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=1)

# Allowed metadata fields from Wikipedia documents (data minimisation)
_ALLOWED_DOC_META_FIELDS: set[str] = {"title", "categories"}
_MAX_DOC_CONTENT_LENGTH: int = 500  # reduced from 1000 for minimisation
_MAX_CATEGORIES: int = 5            # cap number of categories forwarded


def format_doc(doc, max_length=_MAX_DOC_CONTENT_LENGTH):
    # Data minimisation: only use allowed metadata fields
    title = doc.metadata.get("title", "Unknown")
    categories = doc.metadata.get("categories", [])
    # Limit categories to reduce over-broad context injection
    categories = categories[:_MAX_CATEGORIES]
    related = "- ".join(categories)
    # Sanitize retrieved content before injecting into prompt
    content = doc.page_content[:max_length]
    sanitize_input(title, "doc.title")
    return f"### {title}\n\nSummary: {content}\n\nRelated\n{related}"


def format_docs(docs):
    return "\n\n".join(format_doc(doc) for doc in docs)


@as_runnable
async def survey_subjects(topic: str):
    sanitize_input(topic, "topic")
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "expand_chain",
            "model_id": APPROVED_FAST_MODEL,
            "input_summary": {"topic": topic},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    related_subjects = await expand_chain.ainvoke({"topic": topic})
    log_llm_interaction(
        "expand_chain",
        {"topic": topic},
        related_subjects.dict() if hasattr(related_subjects, "dict") else str(related_subjects),
        APPROVED_FAST_MODEL,
    )
    # Validate topics from LLM output before using as retrieval queries
    validated_topics = []
    for t in related_subjects.topics:
        try:
            sanitize_input(t, "related_topic")
            validated_topics.append(t)
        except ValueError:
            audit_logger.warning(
                "related_topic_rejected",
                extra={
                    "session_id": _SESSION_ID,
                    "topic": t,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
    assert_tool_allowed("wikipedia_retriever")
    retrieved_docs = await wikipedia_retriever.abatch(
        validated_topics, return_exceptions=True
    )
    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)
    formatted = format_docs(all_docs)
    # Sanitize formatted docs before injecting into LLM prompt
    sanitize_input(formatted, "formatted_docs")
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "gen_perspectives_chain",
            "model_id": APPROVED_FAST_MODEL,
            "input_summary": {"topic": topic, "examples_length": len(formatted)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    result = await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})
    log_llm_interaction(
        "gen_perspectives_chain",
        {"examples": formatted[:200], "topic": topic},
        result.dict() if hasattr(result, "dict") else str(result),
        APPROVED_FAST_MODEL,
    )
    return result


def add_messages(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


def update_references(references, new_references):
    if not references:
        references = {}
    references.update(new_references)
    return references


def update_editor(editor, new_editor):
    # Can only set at the outset
    if not editor:
        return new_editor
    return editor


class InterviewState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    references: Annotated[dict | None, update_references]
    editor: Annotated[Editor | None, update_editor]


gen_qn_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced Wikipedia writer and want to edit a specific page. \
Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic. \
Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.\
Please only ask one question at a time and don't ask what you have asked before.\
Your questions should be related to the topic you want to write.
Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

Stay true to your specific perspective:

{persona}""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


def tag_with_name(ai_message: AIMessage, name: str):
    ai_message.name = name.replace(" ", "_").replace(".", "_")
    return ai_message


def swap_roles(state: InterviewState, name: str):
    converted = []
    for message in state["messages"]:
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.dict(exclude={"type"}))
        converted.append(message)
    return {"messages": converted}


@as_runnable
async def generate_question(state: InterviewState):
    editor = state["editor"]
    gn_chain = (
        RunnableLambda(swap_roles).bind(name=editor.name)
        | gen_qn_prompt.partial(persona=editor.persona)
        | fast_llm
        | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "generate_question",
            "model_id": APPROVED_FAST_MODEL,
            "editor": editor.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    result = await gn_chain.ainvoke(state)
    # Validate output for dangerous primitives
    if hasattr(result, "content"):
        validate_llm_output(result.content, "generate_question")
    log_llm_interaction(
        "generate_question",
        {"editor": editor.name},
        result.content if hasattr(result, "content") else str(result),
        APPROVED_FAST_MODEL,
    )
    return {"messages": [result]}


class Queries(BaseModel):
    queries: list[str] = Field(
        description="Comprehensive list of search engine queries to answer the user's questions.",
    )


gen_queries_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant. Query the search engine to answer the user's questions.",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)
gen_queries_chain = gen_queries_prompt | fast_llm.with_structured_output(Queries, include_raw=True)


class AnswerWithCitations(BaseModel):
    answer: str = Field(
        description="Comprehensive answer to the user's question with citations.",
    )
    cited_urls: list[str] = Field(
        description="List of urls cited in the answer.",
    )

    @property
    def as_str(self) -> str:
        return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
            f"[{i + 1}]: {url}" for i, url in enumerate(self.cited_urls)
        )


gen_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants\
 to write a Wikipedia page on the topic you know. You have gathered the related information and will now use the information to form a response.

Make your response as informative as possible and make sure every sentence is supported by the gathered information.
Each response must be backed up by a citation from a reliable source, formatted as a footnote, reproducing the URLS after your response.""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

gen_answer_chain = gen_answer_prompt | fast_llm.with_structured_output(
    AnswerWithCitations, include_raw=True
).with_config(run_name="GenerateAnswer")


# Tavily is typically a better search engine, but your free queries are limited
try:
    from langchain_community.tools.tavily_search import TavilySearchResults as _TavilySearchResults
    _tavily_available = True
except ImportError:
    _tavily_available = False

if _tavily_available:
    tavily_search = _TavilySearchResults(max_results=4)
else:
    tavily_search = None


@tool
async def search_engine(query: str):
    """Search engine to the internet."""
    assert_tool_allowed("search_engine")
    sanitize_input(query, "search_query")
    if tavily_search is None:
        return []
    results = tavily_search.invoke(query)
    # Enforce URL allow-list on returned results
    filtered = []
    for r in results:
        try:
            validate_url(r["url"])
            filtered.append({"content": r["content"], "url": r["url"]})
        except ValueError:
            audit_logger.warning(
                "search_result_url_blocked",
                extra={
                    "session_id": _SESSION_ID,
                    "url": r.get("url"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
    return filtered


async def gen_answer(
    state: InterviewState,
    config: RunnableConfig | None = None,
    name: str = "Subject_Matter_Expert",
    max_str_len: int = 15000,
):
    swapped_state = swap_roles(state, name)  # Convert all other AI messages
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "gen_queries_chain",
            "model_id": APPROVED_FAST_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    queries = await gen_queries_chain.ainvoke(swapped_state)
    log_llm_interaction(
        "gen_queries_chain",
        {"messages_count": len(swapped_state.get("messages", []))},
        str(queries.get("parsed", "")),
        APPROVED_FAST_MODEL,
    )
    # Validate queries from LLM before using as search inputs
    validated_queries = []
    for q in queries["parsed"].queries:
        try:
            sanitize_input(q, "search_query")
            validated_queries.append(q)
        except ValueError:
            audit_logger.warning(
                "query_rejected",
                extra={
                    "session_id": _SESSION_ID,
                    "query": q,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
    query_results = await search_engine.abatch(
        validated_queries, config, return_exceptions=True
    )
    successful_results = [
        res for res in query_results if not isinstance(res, Exception)
    ]
    all_query_results = {
        res["url"]: res["content"] for results in successful_results for res in results
    }
    # We could be more precise about handling max token length if we wanted to here
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].tool_calls[0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    # Only update the shared state with the final answer to avoid
    # polluting the dialogue history with intermediate messages
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "gen_answer_chain",
            "model_id": APPROVED_FAST_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    generated = await gen_answer_chain.ainvoke(swapped_state)
    log_llm_interaction(
        "gen_answer_chain",
        {"messages_count": len(swapped_state.get("messages", []))},
        str(generated.get("parsed", "")),
        APPROVED_FAST_MODEL,
    )
    # Validate LLM answer output
    answer_str = generated["parsed"].as_str
    validate_llm_output(answer_str, "gen_answer_chain")
    cited_urls = set(generated["parsed"].cited_urls)
    # Enforce URL allow-list on cited URLs
    safe_cited_urls = set()
    for url in cited_urls:
        try:
            validate_url(url)
            safe_cited_urls.add(url)
        except ValueError:
            audit_logger.warning(
                "cited_url_blocked",
                extra={
                    "session_id": _SESSION_ID,
                    "url": url,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
    # Save the retrieved information to a the shared state for future reference
    cited_references = {k: v for k, v in all_query_results.items() if k in safe_cited_urls}
    formatted_message = AIMessage(name=name, content=answer_str)
    return {"messages": [formatted_message], "references": cited_references}


max_num_turns = 5


def route_messages(state: InterviewState, name: str = "Subject_Matter_Expert"):
    messages = state["messages"]
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    if num_responses >= max_num_turns:
        return END
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    return "ask_question"


builder = StateGraph(InterviewState)

builder.add_node("ask_question", generate_question)
builder.add_node("answer_question", gen_answer)
builder.add_conditional_edges("answer_question", route_messages)
builder.add_edge("ask_question", "answer_question")

builder.set_entry_point("ask_question")
interview_graph = builder.compile().with_config(run_name="Conduct Interviews")

refine_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Wikipedia writer. You have gathered information from experts and search engines. Now, you are refining the outline of the Wikipedia page. \
You need to make sure that the outline is comprehensive and specific. \
Topic you are writing about: {topic} 

Old outline:

{old_outline}""",
        ),
        (
            "user",
            "Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nWrite the refined Wikipedia outline:",
        ),
    ]
)

# Using long context model since the context can get quite long
refine_outline_chain = refine_outline_prompt | long_context_llm.with_structured_output(
    Outline
)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = SKLearnVectorStore(embedding=embeddings)
retriever = vectorstore.as_retriever(k=10)


class SubSection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    content: str = Field(
        ...,
        title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
    )

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.content}".strip()


class WikiSection(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    content: str = Field(..., title="Full content of the section")
    subsections: list[Subsection] | None = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )
    citations: list[str] = Field(default_factory=list)

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
        return (
            f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )


section_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Wikipedia writer. Complete your assigned WikiSection from the following outline:\n\n"
            "{outline}\n\nCite your sources, using the following references:\n\n<Documents>\n{docs}\n<Documents>",
        ),
        ("user", "Write the full WikiSection for the {section} section."),
    ]
)


async def retrieve(inputs: dict):
    # Sanitize retrieval inputs
    topic = sanitize_input(inputs["topic"], "retrieve.topic")
    section = sanitize_input(inputs["section"], "retrieve.section")
    docs = await retriever.ainvoke(topic + ": " + section)
    formatted = "\n".join(
        [
            f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
    return {"docs": formatted, **inputs}


section_writer = (
    retrieve
    | section_writer_prompt
    | long_context_llm.with_structured_output(WikiSection)
)


writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Wikipedia author. Write the complete wiki article on {topic} using the following section drafts:\n\n"
            "{draft}\n\nStrictly follow Wikipedia format guidelines.",
        ),
        (
            "user",
            'Write the complete Wiki article using markdown format. Organize citations using footnotes like "[1]",'
            " avoiding duplicates in the footer. Include URLs in the footer.",
        ),
    ]
)

writer = writer_prompt | long_context_llm | StrOutputParser()


class ResearchState(TypedDict):
    topic: str
    outline: Outline
    editors: list[Editor]
    interview_results: list[InterviewState]
    # The final sections output
    sections: list[WikiSection]
    article: str
    # Authentication token for the session
    auth_token: str


def _check_auth(state: ResearchState) -> None:
    """Raise if the session token is not valid."""
    token = state.get("auth_token", "")
    if not authenticate(token):
        audit_logger.warning(
            "authentication_failed",
            extra={
                "session_id": _SESSION_ID,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise PermissionError(
            "Authentication required: provide a valid auth_token in the ResearchState."
        )
    audit_logger.info(
        "authentication_success",
        extra={
            "session_id": _SESSION_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def initialize_research(state: ResearchState):
    _check_auth(state)
    topic = sanitize_input(state["topic"], "topic")
    audit_logger.info(
        "workflow_step",
        extra={
            "session_id": _SESSION_ID,
            "step": "initialize_research",
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "generate_outline_direct",
            "model_id": APPROVED_FAST_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    coros = (
        generate_outline_direct.ainvoke({"topic": topic}),
        survey_subjects.ainvoke(topic),
    )
    results = await asyncio.gather(*coros)
    outline_result = results[0]
    perspectives_result = results[1]
    log_llm_interaction(
        "generate_outline_direct",
        {"topic": topic},
        outline_result.dict() if hasattr(outline_result, "dict") else str(outline_result),
        APPROVED_FAST_MODEL,
    )
    # Validate outline output
    validate_llm_output(outline_result.as_str, "generate_outline_direct")
    return {
        **state,
        "outline": outline_result,
        "editors": perspectives_result.editors,
    }


async def conduct_interviews(state: ResearchState):
    _check_auth(state)
    topic = sanitize_input(state["topic"], "topic")
    audit_logger.info(
        "workflow_step",
        extra={
            "session_id": _SESSION_ID,
            "step": "conduct_interviews",
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    initial_states = [
        {
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were writing an article on {topic}?",
                    name="Subject_Matter_Expert",
                )
            ],
        }
        for editor in state["editors"]
    ]
    # We call in to the sub-graph here to parallelize the interviews
    interview_results = await interview_graph.abatch(initial_states)

    return {
        **state,
        "interview_results": interview_results,
    }


def format_conversation(interview_state):
    messages = interview_state["messages"]
    convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
    return f"Conversation with {interview_state['editor'].name}\n\n" + convo


async def refine_outline(state: ResearchState):
    _check_auth(state)
    topic = sanitize_input(state["topic"], "topic")
    audit_logger.info(
        "workflow_step",
        extra={
            "session_id": _SESSION_ID,
            "step": "refine_outline",
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    convos = "\n\n".join(
        [
            format_conversation(interview_state)
            for interview_state in state["interview_results"]
        ]
    )
    # Sanitize conversation content before injecting into LLM prompt
    sanitize_input(convos, "conversations")
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "refine_outline_chain",
            "model_id": APPROVED_LONG_CONTEXT_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    updated_outline = await refine_outline_chain.ainvoke(
        {
            "topic": topic,
            "old_outline": state["outline"].as_str,
            "conversations": convos,
        }
    )
    log_llm_interaction(
        "refine_outline_chain",
        {"topic": topic, "old_outline_length": len(state["outline"].as_str)},
        updated_outline.dict() if hasattr(updated_outline, "dict") else str(updated_outline),
        APPROVED_LONG_CONTEXT_MODEL,
    )
    # Validate refined outline output
    validate_llm_output(updated_outline.as_str, "refine_outline_chain")
    return {**state, "outline": updated_outline}


async def index_references(state: ResearchState):
    _check_auth(state)
    audit_logger.info(
        "workflow_step",
        extra={
            "session_id": _SESSION_ID,
            "step": "index_references",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    all_docs = []
    for interview_state in state["interview_results"]:
        reference_docs = [
            Document(page_content=v, metadata={"source": k})
            for k, v in interview_state["references"].items()
        ]
        all_docs.extend(reference_docs)
    await vectorstore.aadd_documents(all_docs)
    return state


async def write_sections(state: ResearchState):
    _check_auth(state)
    topic = sanitize_input(state["topic"], "topic")
    audit_logger.info(
        "workflow_step",
        extra={
            "session_id": _SESSION_ID,
            "step": "write_sections",
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    outline = state["outline"]
    # Sanitize outline and section titles before injecting into LLM
    sanitize_input(outline.as_str, "outline")
    for section in outline.sections:
        sanitize_input(section.section_title, "section_title")
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "section_writer",
            "model_id": APPROVED_LONG_CONTEXT_MODEL,
            "sections": [s.section_title for s in outline.sections],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    sections = await section_writer.abatch(
        [
            {
                "outline": outline.as_str,
                "section": section.section_title,
                "topic": topic,
            }
            for section in outline.sections
        ]
    )
    # Validate each section output
    for sec in sections:
        if hasattr(sec, "as_str"):
            validate_llm_output(sec.as_str, "section_writer")
    log_llm_interaction(
        "section_writer",
        {"topic": topic, "section_count": len(outline.sections)},
        [s.section_title for s in sections if hasattr(s, "section_title")],
        APPROVED_LONG_CONTEXT_MODEL,
    )
    return {
        **state,
        "sections": sections,
    }


async def write_article(state: ResearchState):
    _check_auth(state)
    topic = sanitize_input(state["topic"], "topic")
    audit_logger.info(
        "workflow_step",
        extra={
            "session_id": _SESSION_ID,
            "step": "write_article",
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    sections = state["sections"]
    draft = "\n\n".join([section.as_str for section in sections])
    # Sanitize draft before injecting into LLM
    sanitize_input(draft, "draft")
    audit_logger.info(
        "llm_call_start",
        extra={
            "session_id": _SESSION_ID,
            "call_name": "writer",
            "model_id": APPROVED_LONG_CONTEXT_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    article = await writer.ainvoke({"topic": topic, "draft": draft})
    log_llm_interaction(
        "writer",
        {"topic": topic, "draft_length": len(draft)},
        article[:500] if isinstance(article, str) else str(article),
        APPROVED_LONG_CONTEXT_MODEL,
    )
    # Validate article output for dangerous primitives
    validate_llm_output(article, "writer")
    # Attach AI-generated content provenance label
    article = attach_provenance(article, APPROVED_LONG_CONTEXT_MODEL)
    audit_logger.info(
        "workflow_step_complete",
        extra={
            "session_id": _SESSION_ID,
            "step": "write_article",
            "article_length": len(article),
            "provenance_attached": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        **state,
        "article": article,
    }


builder_of_storm = StateGraph(ResearchState)

nodes = [
    ("init_research", initialize_research),
    ("conduct_interviews", conduct_interviews),
    ("refine_outline", refine_outline),
    ("index_references", index_references),
    ("write_sections", write_sections),
    ("write_article", write_article),
]
for i in range(len(nodes)):
    name, node = nodes[i]
    builder_of_storm.add_node(name, node)
    if i > 0:
        builder_of_storm.add_edge(nodes[i - 1][0], name)

builder_of_storm.set_entry_point(nodes[0][0])
builder_of_storm.set_finish_point(nodes[-1][0])
graph = builder_of_storm.compile()