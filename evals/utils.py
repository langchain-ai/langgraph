import math
import bisect
from typing import Dict, List, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableLambda


_DEFAULT_OPENAI_LOGPROBS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user input.{class_descriptions} MAKE SURE your output is one of the classes and NOTHING else.",  # noqa: E501
        ),
        ("human", "{input}"),
    ]
)


def _parse_logprobs(
    result: LLMResult, classes: List[str], top_k: int
) -> Union[Dict, List]:
    original_classes = classes.copy()
    classes = [c.lower() for c in classes]
    top_classes = [
        c for c in classes if result.generations[0][0].text.lower().startswith(c)
    ]
    generation = result.generations[0][0].copy()

    logprobs = generation.generation_info["logprobs"]["content"]
    all_logprobs = [lp for token in logprobs for lp in token["top_logprobs"]]
    present_token_classes = [
        lp for lp in all_logprobs if lp["token"].strip().lower() in classes
    ]
    if not top_classes and not present_token_classes:
        res = {"classification": None, "confidence": None}
        return res if top_k == 1 else [res]

    # If any individual token matches a class.
    cumulative = {}
    for lp in present_token_classes:
        normalized = lp["token"].strip().lower()
        if normalized in cumulative:
            cumulative[normalized] += math.exp(lp["logprob"])
        else:
            cumulative[normalized] = math.exp(lp["logprob"])

    # If there are present classes that span more than a token.
    present_multi_token_classes = set(top_classes).difference(cumulative)
    spans = [len(logprobs[0]["token"])]
    for lp in logprobs[1:]:
        spans.append(len(lp["token"]))
    for top_class in present_multi_token_classes:
        start = generation.text.find(top_class)
        start_token_idx = bisect.bisect(spans, start)
        end = start + len(top_class)
        end_token_idx = bisect.bisect_left(spans, end)
        cumulative[top_class] = math.exp(
            sum(lp["logprob"] for lp in logprobs[start_token_idx : end_token_idx + 1])
        )
    res = sorted(
        [
            {"classification": original_classes[classes.index(k)], "confidence": v}
            for k, v in cumulative.items()
        ],
        key=(lambda x: x["confidence"]),
        reverse=True,
    )
    # Softmax
    total = sum(r["confidence"] for r in res)
    for r in res:
        r["confidence"] /= total
    return res[0] if top_k == 1 else res[:top_k]


def create_openai_logprobs_classification_chain(
    llm: BaseChatModel,
    classes: Union[Sequence[str], Dict[str, str]],
    /,
    *,
    prompt: Optional[BasePromptTemplate] = None,
    top_k: Optional[int] = None,
) -> Runnable[Dict, Dict]:
    """"""
    prompt = prompt or _DEFAULT_OPENAI_LOGPROBS_PROMPT
    if isinstance(classes, Dict):
        descriptions = "\n".join(f"{k}: {v}" for k, v in classes.items())
        class_descriptions = f"\n\nThe classes are:\n\n{descriptions}\n\n"
    else:
        names = ", ".join(classes)
        class_descriptions = f"The classes are: {names}."
    prompt = prompt.partial(class_descriptions=class_descriptions)
    if top_k is None:
        top_k = min(len(classes), 20)  # OpenAI max
    generate = RunnableLambda(llm.generate_prompt, afunc=llm.agenerate_prompt).bind(
        logprobs=True, top_logprobs=top_k
    )
    parse = RunnableLambda(_parse_logprobs).bind(classes=list(classes), top_k=top_k)
    return prompt | (lambda x: [x]) | generate | parse
