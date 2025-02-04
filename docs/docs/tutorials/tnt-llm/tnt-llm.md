# TNT-LLM: Text Mining at Scale

[TNT-LLM](https://arxiv.org/abs/2403.12173) by Wan, et. al describes a taxonomy generation and classification system developed by Microsoft for their Bing Copilot application.

It generates a rich, interpretable taxonomy of user intents (or other categories) from raw conversation logs. This taxonomy can then be used downstream by LLMs to label logs, which in turn can be used as training data to adapt a cheap classifier (such as logistic regression classifier on embeddings) that can be deployed in your app.

TNT-LLM has three main phases:

1. Generate Taxonomy
2. Label Training Data
3. Finetune classifier + deploy

When applying LangGraph in this notebook, we will focus on the first phase: taxonomy generation (blue in the diagram below). We then show how to label and fit the classifier in subsequent steps below.

<img src="../img/tnt_llm.png" src="../img/tnt_llm.png">

To generate the taxonomy, TNT-LLM proposes 5 steps:

1. **Summarize** chat logs using a lower-cost LLM (batched over all logs in the sample)
2. **Batch** the logs into random minibatches
3. **Generate** an initial taxonomy from the first minibatch
4. **Update** the taxonomy on each subsequent minibatch via a ritique and revise prompt
5. **Review** the final taxonomy, scoring its quality and generating a final value using a final sample.

## Setup

First, let's install our required packages and set our API keys



```
%%capture --no-stderr
%pip install -U langgraph langchain_anthropic langsmith langchain-community
%pip install -U sklearn langchain_openai
```


```python
import getpass
import os


def _set_env(var: str):
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var + ":")


_set_env("ANTHROPIC_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>   

## Define the graph

### Graph State

Since each node of a StateGraph accepts the state (and returns an updated state), we'll define that at the outset.

Our flow takes in a list of documents, batches them, and then generates and refines candidate taxonomies as interpretable "clusters".



```python
import logging
import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("tnt-llm")


class Doc(TypedDict):
    id: str
    content: str
    summary: Optional[str]
    explanation: Optional[str]
    category: Optional[str]


class TaxonomyGenerationState(TypedDict):
    # The raw docs; we inject summaries within them in the first step
    documents: List[Doc]
    # Indices to be concise
    minibatches: List[List[int]]
    # Candidate Taxonomies (full trajectory)
    clusters: Annotated[List[List[dict]], operator.add]
```

### Define nodes

#### 1. Summarize Docs

Chat logs can get quite long. Our taxonomy generation step needs to see large, diverse minibatches to be able to adequately capture the distribution of categories. To ensure they can all fit efficiently into the context window, we first summarize each chat log. Downstream steps will use these summaries instead of the raw doc content.



```python
import re

from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough

summary_prompt = hub.pull("wfh/tnt-llm-summary-generation").partial(
    summary_length=20, explanation_length=30
)


def parse_summary(xml_string: str) -> dict:
    summary_pattern = r"<summary>(.*?)</summary>"
    explanation_pattern = r"<explanation>(.*?)</explanation>"

    summary_match = re.search(summary_pattern, xml_string, re.DOTALL)
    explanation_match = re.search(explanation_pattern, xml_string, re.DOTALL)

    summary = summary_match.group(1).strip() if summary_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return {"summary": summary, "explanation": explanation}


summary_llm_chain = (
    summary_prompt | ChatAnthropic(model="claude-3-haiku-20240307") | StrOutputParser()
    # Customize the tracing name for easier organization
).with_config(run_name="GenerateSummary")
summary_chain = summary_llm_chain | parse_summary


# Now combine as a "map" operation in a map-reduce chain
# Input: state
# Output: state U summaries
# Processes docs in parallel
def get_content(state: TaxonomyGenerationState):
    docs = state["documents"]
    return [{"content": doc["content"]} for doc in docs]


map_step = RunnablePassthrough.assign(
    summaries=get_content
    # This effectively creates a "map" operation
    # Note you can make this more robust by handling individual errors
    | RunnableLambda(func=summary_chain.batch, afunc=summary_chain.abatch)
)


def reduce_summaries(combined: dict) -> TaxonomyGenerationState:
    summaries = combined["summaries"]
    documents = combined["documents"]
    return {
        "documents": [
            {
                "id": doc["id"],
                "content": doc["content"],
                "summary": summ_info["summary"],
                "explanation": summ_info["explanation"],
            }
            for doc, summ_info in zip(documents, summaries)
        ]
    }


# This is actually the node itself!
map_reduce_chain = map_step | reduce_summaries
```

#### 2. Split into Minibatches

Each minibatch contains a random sample of docs. This lets the flow identify inadequacies in the current taxonomy using new data.



```python
import random


def get_minibatches(state: TaxonomyGenerationState, config: RunnableConfig):
    batch_size = config["configurable"].get("batch_size", 200)
    original = state["documents"]
    indices = list(range(len(original)))
    random.shuffle(indices)
    if len(indices) < batch_size:
        # Don't pad needlessly if we can't fill a single batch
        return [indices]

    num_full_batches = len(indices) // batch_size

    batches = [
        indices[i * batch_size : (i + 1) * batch_size] for i in range(num_full_batches)
    ]

    leftovers = len(indices) % batch_size
    if leftovers:
        last_batch = indices[num_full_batches * batch_size :]
        elements_to_add = batch_size - leftovers
        last_batch += random.sample(indices, elements_to_add)
        batches.append(last_batch)

    return {
        "minibatches": batches,
    }
```

#### 3.a Taxonomy Generation Utilities

This section of the graph is a generate -> update 🔄 -> review cycle. Each node shares a LOT of logic, which we have factored out into the shared functions below.



```python
from typing import Dict

from langchain_core.runnables import Runnable


def parse_taxa(output_text: str) -> Dict:
    """Extract the taxonomy from the generated output."""
    cluster_matches = re.findall(
        r"\s*<id>(.*?)</id>\s*<name>(.*?)</name>\s*<description>(.*?)</description>\s*",
        output_text,
        re.DOTALL,
    )
    clusters = [
        {"id": id.strip(), "name": name.strip(), "description": description.strip()}
        for id, name, description in cluster_matches
    ]
    # We don't parse the explanation since it isn't used downstream
    return {"clusters": clusters}


def format_docs(docs: List[Doc]) -> str:
    xml_table = "<conversations>\n"
    for doc in docs:
        xml_table += f'<conv_summ id={doc["id"]}>{doc["summary"]}</conv_summ>\n'
    xml_table += "</conversations>"
    return xml_table


def format_taxonomy(clusters):
    xml = "<cluster_table>\n"
    for label in clusters:
        xml += "  <cluster>\n"
        xml += f'    <id>{label["id"]}</id>\n'
        xml += f'    <name>{label["name"]}</name>\n'
        xml += f'    <description>{label["description"]}</description>\n'
        xml += "  </cluster>\n"
    xml += "</cluster_table>"
    return xml


def invoke_taxonomy_chain(
    chain: Runnable,
    state: TaxonomyGenerationState,
    config: RunnableConfig,
    mb_indices: List[int],
) -> TaxonomyGenerationState:
    configurable = config["configurable"]
    docs = state["documents"]
    minibatch = [docs[idx] for idx in mb_indices]
    data_table_xml = format_docs(minibatch)

    previous_taxonomy = state["clusters"][-1] if state["clusters"] else []
    cluster_table_xml = format_taxonomy(previous_taxonomy)

    updated_taxonomy = chain.invoke(
        {
            "data_xml": data_table_xml,
            "use_case": configurable["use_case"],
            "cluster_table_xml": cluster_table_xml,
            "suggestion_length": configurable.get("suggestion_length", 30),
            "cluster_name_length": configurable.get("cluster_name_length", 10),
            "cluster_description_length": configurable.get(
                "cluster_description_length", 30
            ),
            "explanation_length": configurable.get("explanation_length", 20),
            "max_num_clusters": configurable.get("max_num_clusters", 25),
        }
    )

    return {
        "clusters": [updated_taxonomy["clusters"]],
    }
```

#### 3. Generate initial taxonomy



```python
# We will share an LLM for each step of the generate -> update -> review cycle
# You may want to consider using Opus or another more powerful model for this
taxonomy_generation_llm = ChatAnthropic(
    model="claude-3-haiku-20240307", max_tokens_to_sample=2000
)


## Initial generation
taxonomy_generation_prompt = hub.pull("wfh/tnt-llm-taxonomy-generation").partial(
    use_case="Generate the taxonomy that can be used to label the user intent in the conversation.",
)

taxa_gen_llm_chain = (
    taxonomy_generation_prompt | taxonomy_generation_llm | StrOutputParser()
).with_config(run_name="GenerateTaxonomy")


generate_taxonomy_chain = taxa_gen_llm_chain | parse_taxa


def generate_taxonomy(
    state: TaxonomyGenerationState, config: RunnableConfig
) -> TaxonomyGenerationState:
    return invoke_taxonomy_chain(
        generate_taxonomy_chain, state, config, state["minibatches"][0]
    )
```

#### 4. Update Taxonomy

This is a "critique -> revise" step that is repeated N times.



```python
taxonomy_update_prompt = hub.pull("wfh/tnt-llm-taxonomy-update")

taxa_update_llm_chain = (
    taxonomy_update_prompt | taxonomy_generation_llm | StrOutputParser()
).with_config(run_name="UpdateTaxonomy")


update_taxonomy_chain = taxa_update_llm_chain | parse_taxa


def update_taxonomy(
    state: TaxonomyGenerationState, config: RunnableConfig
) -> TaxonomyGenerationState:
    which_mb = len(state["clusters"]) % len(state["minibatches"])
    return invoke_taxonomy_chain(
        update_taxonomy_chain, state, config, state["minibatches"][which_mb]
    )
```

#### 5. Review Taxonomy

This runs once we've processed all the minibatches.



```python
taxonomy_review_prompt = hub.pull("wfh/tnt-llm-taxonomy-review")

taxa_review_llm_chain = (
    taxonomy_review_prompt | taxonomy_generation_llm | StrOutputParser()
).with_config(run_name="ReviewTaxonomy")


review_taxonomy_chain = taxa_review_llm_chain | parse_taxa


def review_taxonomy(
    state: TaxonomyGenerationState, config: RunnableConfig
) -> TaxonomyGenerationState:
    batch_size = config["configurable"].get("batch_size", 200)
    original = state["documents"]
    indices = list(range(len(original)))
    random.shuffle(indices)
    return invoke_taxonomy_chain(
        review_taxonomy_chain, state, config, indices[:batch_size]
    )
```

### Compile the Graph

With all the functionality defined, we can build the graph!



```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(TaxonomyGenerationState)
graph.add_node("summarize", map_reduce_chain)
graph.add_node("get_minibatches", get_minibatches)
graph.add_node("generate_taxonomy", generate_taxonomy)
graph.add_node("update_taxonomy", update_taxonomy)
graph.add_node("review_taxonomy", review_taxonomy)

graph.add_edge("summarize", "get_minibatches")
graph.add_edge("get_minibatches", "generate_taxonomy")
graph.add_edge("generate_taxonomy", "update_taxonomy")


def should_review(state: TaxonomyGenerationState) -> str:
    num_minibatches = len(state["minibatches"])
    num_revisions = len(state["clusters"])
    if num_revisions < num_minibatches:
        return "update_taxonomy"
    return "review_taxonomy"


graph.add_conditional_edges(
    "update_taxonomy",
    should_review,
    # Optional (but required for the diagram to be drawn correctly below)
    {"update_taxonomy": "update_taxonomy", "review_taxonomy": "review_taxonomy"},
)
graph.add_edge("review_taxonomy", END)

graph.add_edge(START, "summarize")
app = graph.compile()
```


```python
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![](data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAJ2ATQDASIAAhEBAxEB/8QAHQABAAMBAQEBAQEAAAAAAAAAAAUGBwQIAwECCf/EAFsQAAEDAwICAwgLCgoHBwUAAAEAAgMEBQYREgchEzGUFBUXIkFW0tMIFjI3UVVhdZKz0SMmMzZUcXSVstQlQkRSU2Jjc4GTJDQ1kaGxtAlDcoKj8PEYg6TBwv/EABsBAQADAQEBAQAAAAAAAAAAAAABAgQDBQcG/8QANhEBAAECAgcFBwMFAQEAAAAAAAEDEQISFCExUVKR0QRhcaHSEzIzQWKSsSKBwQUVI0LwwuH/2gAMAwEAAhEDEQA/AP8AVNERAREQEREBERAREQEREBERAREQEREBERARFDXu8VENTFbbZGya6zsLwZQTDTRg6GWXTQ6a8msBBeeQIAe9lsOGcc2gS0krIWF8j2sY3rc46AKOdlNmaSHXegBHWDUs+1R8eBWueQT3Zjr/AFepPTXLSRrdeWjI9NjBpy8VoPwkkkrvGK2RoAFnoAByAFKzl/wXa1GNszP/AH/bk6j21WT44oO1M+1PbVZPjig7Uz7V++1ay/FFB2Zn2J7VrL8UUHZmfYn+Hv8AJOp+e2qyfHFB2pn2p7arJ8cUHamfav32rWX4ooOzM+xPatZfiig7Mz7E/wAPf5Gp+e2qyfHFB2pn2p7arJ8cUHamfav32rWX4ooOzM+xPatZfiig7Mz7E/w9/kan3pL1b69+ylr6apf/ADYZmvP/AAK7VCVmD47cI9lTYbbM0dW+kjJH5jpyPl1C4X2qtxFpqLXJV3G2sGslqleZnsb5XQPcd2v9m5xBA0bt8rJTxasE6+/r/wB4otC0ovhRVsFxpIaqmkbNTzND2SN6nA9S+64TExNpQIiKAREQEREBERAREQEREBERAREQEREBERAREQEREBERAVYwTS4Ulfe36OnuVXK4O58oI3ujhb8niNDtBy3Pd166mzqscOR3Pi0VC7US0E89G8Eac2SuAP5i3a4fI4LRh1UsUxvjlr/mIT8lnRQuU5tjuDUcVXkl+tmP0s0nRRz3WsjpmPfoTtDnuAJ0BOg+Aqs//UJws018JeH6fD3+pfWLOhM8SeIdq4V4dXZLeRUSUVK6KPoaOLpJppJJGxxxsbqNXOe9oGpA58yBzWX8TfZDX3GMYxG6WvBMggqLrktNZ6m3XKlgZUtjc4bmsHdAYXyA7WODizUO3Fuinc14j4RxNxC7Y/j8+McVa6pibvxakvtJvqog9u8g7iGlg8cE6eM1vNvWMwpuFfErwU0oktlTU1lizWlyCx4zdLxHU1kdthcz/RHVZcWF+vSlu57gBtBcfIGx5hxkkw222yrmwLMbmaukNZPBbKCGd9A0AFzZyJgzeNfcxueTodNVGXX2SmOU1ZidJabVfMpqMptL7xaY7NSxv6eFvR6gmSRgjdpID4+gG0gkO0BonEvCMu4g5tab3eeHDsnsMtjNNDjNdeKdkFpuPTPLqiobvLJQ6MxgPjEjmbXaN56r84EcIsvw++cKZb3ZhRxY7h9fY66UVUMjWVBqqcxbdriS17InPBA5AgO2u5ILZg3HC/5Pxzy/DarDbtTWq2MoehrdlOBSmWGSRzqkioJIeWtazo2u6vGA61tKxVluyPh1x5zHJXWNlww7JKS3vqb0LhBTstPcrJWSunZK5rnM2uD9zNdNDqrW32QfC17g1vErEHOJ0AF+pdT/AOogv6KhU/H3hhVzxwQcR8SmmlcGMjjvlK5z3E6AACTUknyK+oKxjOlsyPILMzRsEborjAwa+Iycv3D/ADYpXf8AnVnVZtDe6s+yGraD0cFLSUGpboOkb0srtD5fFnjVmWiv79+6PxC07RERZ1RERAREQEREBERAREQEREBERAREQEREBERAREQEREBVy4wS45dqi80sD6iiqg3vjTwsc+Xc0BrZ42jUuIaA1zQNXNa3bzbtfY0V8GPJPdKYclNUUN7o46iCSCupX82SMIkYfJyPUv6720n5LB/lj7FE3DCLXW1clZE2e21shLpKm3VD6d0h001eGEB5005vB6h8AXN7R5vJlF+aPg6eI/8AONdctKdmK3jHT/4aliio6eB+6OCON3Vq1gBX2VW9pE/nTfv8+L1Se0ifzpv3+fF6pPZ0+PylNo3rSiyriPbrpi1ts89DlN5MlXerfQSdNNER0U1SyOTT7mPG2uOny6cirZ7SJ/Om/f58Xqk9nT4/KS0b1oc0PaWuALSNCD5Vz97aM/yWD/LH2Kv+0ifzpv3+fF6pPaRP5037/Pi9Uns6fH5SWjesIt1ICCKWEEeURhRl5yNtLUG224R117e3VlLu5RA9UkxHuGfKeZ00aCVxHBGzANqr9faqPTQsNcYt35zEGH/ipq0WSgsNKae30sdLETucIxze7+c49bj8p1KWpYNd808o6/8AbUaofOwWVlhtzaZshnlc98087hoZZXuLnvPwaknQeQaAcgFJIi44sU4pnFO2UCIiqCIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiIM941ECy41qSB7Z7P1fpsXyj/38K0JZ7xq17y41pp+M9n90B+WxfD/APPwLQkBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERBnnGwa2TGtXBv30Wfm4a/y2LktDWecbNO8mNa6j76LP1DX+WxLQ0BERAREQEREBERAREQEREBERAREQEREBERAREQEREBE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VH69O+1/8zbn2qj9ens/qj7sPUsm1F2X3yqr5oj+ueviLrfydPadch8pqqPT65TOMWOrgrqq73JkcNdUxsgZTRSF7YImlxALtBq8l5LiBoNGga7dzq47U8GK8xri2qYn8GxZERF5SoiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiDP+M41s2N8tfvmtHk1/lkfyH/AN+Uda0BZ7xqbusuNDQn757OeQ1/lsS0JAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQZ5xsIFkxrU6D20Wfya/y2JaGs+407jZcb2l4PtntGuwanTuyLX/AA+H5FoKAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIvlVVUFDTyVFTNHTwRjc+WVwa1o+Ek8gFMRfVA+qKrHilhwP40Wj84rYyD/xTwqYd50Wntkf2rvo9bgnlK2Wdy0oqt4VMO86LT2yP7U8KmHedFp7ZH9qaNW4J5SZZ3LSiq3hUw7zotPbI/tTwqYd50Wntkf2po1bgnlJlnctKKreFTDvOi09sj+1PCph3nRae2R/amjVuCeUmWdyg8fOJuGWdtktNxyyx0N0pcjtE89DU3KGOeFgqopC97HPDmtDdHanlpz6lq9hyC15TaoLpZblR3e2T7uiraCdk8Mm1xa7a9pIOjmkHQ8iCPIvAv8A2hHCWzcV71i+WYbdLZXXuWaO03KKCpYT0Tj9zndofcs5hzj1At+BetOEtdw+4ScN8exC2ZRaXUlppWwdJ3WwdI/m6SQ8+tz3Od/5k0atwTykyzuasiq3hUw7zotPbI/tTwqYd50Wntkf2po1bgnlJlnctKKreFTDvOi09sj+1PCph3nRae2R/amjVuCeUmWdy0oqt4VMO86LT2yP7U8KmHedFp7ZH9qaNW4J5SZZ3LSiq3hUw7zotPbI/tXdac4x6/VTaa3Xy311S4EthgqWPe4AanRoOp08qiaFXDF5wTbwlFpTaIi4IEREBERAREQEREBERAREQEREBERAREQFS8pcLhmdqt04ElJFRy1vQu5tdKJI2scR1HaC7QHXm7XrAKuipF898mh+aZvro1r7L8SZ7p/C0JRERaFRERAREQEREBERAREQEREBERAUffrbDdLVUQzN5hpfHIOT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QEREBfOop4quCSGeJk0MgLXxyNDmuB6wQesL6ImwVd3C7DXnU4nZCfm+L0V+eCzDPNKyfq+L0VaUWjSK3HPOVs071W8FmGeaVk/V8Xop4LMM80rJ+r4vRVpRNIrcc85M071W8FmGeaVk/V8Xop4LMM80rJ+r4vRVpRNIrcc85M071W8FmGeaVk/V8Xop4LMM80rJ+r4vRVpRNIrcc85M072O8XOHeLW60Y++kx21Ub5MitUMjoaOJhfG6rja9hOg1a4Egjyg6aHqV58FmGeaVk/V8XoqH40EizY3tO0+2e0Dy9XdkXwf+/hWgppFbjnnJmneq3gswzzSsn6vi9FPBZhnmlZP1fF6KtKJpFbjnnJmneq3gswzzSsn6vi9FPBZhnmlZP1fF6KtKJpFbjnnJmneq3gswzzSsn6vi9FPBZhnmlZP1fF6KtKJpFbjnnJmneq3gswzzSsn6vi9Fd9owrHrBUCe2WK22+cagS0tJHG4a9fNoB5qaRRNerii045mPGUXneIiLggREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREGe8agXWXGtGdJ989nOnPl/psXPl8HWtCWe8a2l9lxoBrn/fRZzo39Ni5rQkBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERBnvGkB1lxrUA/fPZz42v5ZF8H/wAfCtCXiH2fXH7iRwZyTGaW3WqxV2K1dTTXGiqamlndOKymlbIYXubMAQSGHk0Ha7TXUar1fwivGT5Dw0x66ZlSUVBktbStqKylt8b44YS8lzWBr3OcCGFodqT42v5kFvREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQF/MkjYY3yPcGsaC5zj5AF/S4b5/sW4fo8n7JVsMXmIFNp6y+ZVSxXKO9VFjpalglp6Wjghc5kZALTI6WN+ryOZAAA1056bj/Xee++el47PQ/uy/vDvxRsf6DB9W1TC9fFMYMU4YwxaO6Oi0zaUJ3nvvnpeOz0P7snee++el47PQ/uym0Vc/0x9sdC6E7z33z0vHZ6H92TvPffPS8dnof3ZTaJn+mPtjoXQnee++el47PQ/uyd57756Xjs9D+7KbRM/0x9sdC6E7z33z0vHZ6H92TvPffPS8dnof3ZTaJn+mPtjoXZ7nvBul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FoAIbpqAApFEsIGy4BjGN1lbV2jHLRaquu1FVPRUMUMlRqdT0jmtBdz+HVdEWI2KCitVHHZbdHSWl7ZbfTspYxHRva0ta6FumkZDXOaC3TQOI8qlkQQUeB4zDkr8ijx20x5A8aOuzaGIVThppoZdu7q5da+suHWCegr6GWx22Sir6juurpn0kZjqZ9Wu6WRumj36sYdx1OrQdeQUwiCCyHGnV0dfW2aSjs2T1FKKOO+PoWVEscYcXNaQS0vaC55DS7QFxPwrk4a4BRcM8Po7DRTSVYifLPPVztaJKmeWR0ksrg0AAue9x0A0A0A5BWhEsCiaP3zrf8z1f11MpZRVE0niZQEaaNtFUDz6tZqfT/kf9y6YdmLwn8LQvCIi8lUREQEREBERAREQEREBERAREQEREFMfh14tf3Gx3Ojit4/BUtfSvlMA/mte2RviDyAgkdWugAX8d4Mw+NLH2Cb1yuyLXHaqnztP7QtdSe8GYfGlj7BN65O8GYfGlj7BN65XZFOlVN0coLqT3gzD40sfYJvXJ3gzD40sfYJvXK7ImlVN0coLqT3gzD40sfYJvXJ3gzD40sfYJvXK7ImlVN0coLqT3gzD40sfYJvXJ3gzD40sfYJvXK7ImlVN0coLspzSty7D6K21Dqqy1fdlzo7aGtopm7DPM2Lf+FOu3drp5dPIrB3gzD40sfYJvXLj41vDLLjRI11yizjyeWti+ELQk0qpujlBdSe8GYfGlj7BN65O8GYfGlj7BN65XZE0qpujlBdSe8GYfGlj7BN65O8GYfGlj7BN65XZE0qpujlBdSe8GYfGlj7BN65O8GYfGlj7BN65XZE0qpujlBdSe8GYfGlj7BN65O8GYfGlj7BN65XZE0qpujlBdSRYMv1Gt0smnl0oJvXKax7Gn2maatrasXC6TtbHJO2Poo2MGpDI2au2t1JJ1c4knmSA0NnEVMfaMeOMs2t3REIuIiLMgREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQZ7xqcW2XGtH7Pvns411I1/02Lly+HqWhLPeNbyyy40QNdcos46yOutiC0JAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQZ5xr07yY1rt09tFn91r+WxfAtDXnD2TXslOHeAXSz4xfshktd8o7za7lNT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)

## Use the graph

The docs can contain **any** content, but we've found it works really well on chat bot logs, such as those captured by [LangSmith](https://smith.langchain.com).

We will use that as an example below. Update the `project_name` to your own LangSmith project.

You will likely have to customize the `run_to_doc` function below, since your expected keys may differ from those of this notebook's author.



```python
from datetime import datetime, timedelta

from langsmith import Client

project_name = "YOUR PROJECT NAME"  # Update to your own project
client = Client()

past_week = datetime.now() - timedelta(days=7)
runs = list(
    client.list_runs(
        project_name=project_name,
        filter="eq(is_root, true)",
        start_time=past_week,
        # We only need to return the inputs + outputs
        select=["inputs", "outputs"],
    )
)


# Convert the langsmith traces to our graph's Doc object.
def run_to_doc(run) -> Doc:
    turns = []
    idx = 0
    for turn in run.inputs.get("chat_history") or []:
        key, value = next(iter(turn.items()))
        turns.append(f"<{key} idx={idx}>\n{value}\n</{key}>")
        idx += 1
    turns.append(
        f"""
<human idx={idx}>
{run.inputs['question']}
</human>"""
    )
    if run.outputs and run.outputs["output"]:
        turns.append(
            f"""<ai idx={idx+1}>
{run.outputs['output']}
</ai>"""
        )
    return {
        "id": str(run.id),
        "content": ("\n".join(turns)),
    }
```

#### Invoke

Now convert the runs to docs and kick off your graph flow. This will take some time! The summary step takes the longest. If you want to speed things up, you could try splitting the load across model providers.



```python
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Optional. If you are running into errors or rate limits and want to avoid repeated computation,
# you can set this while debugging

set_llm_cache(InMemoryCache())
```


```python
# We will randomly sample down to 1K docs to speed things up
docs = [run_to_doc(run) for run in runs if run.inputs]
docs = random.sample(docs, min(len(docs), 1000))
use_case = (
    "Generate the taxonomy that can be used both to label the user intent"
    " as well as to identify any required documentation (references, how-tos, etc.)"
    " that would benefit the user."
)

stream = app.stream(
    {"documents": docs},
    {
        "configurable": {
            "use_case": use_case,
            # Optional:
            "batch_size": 400,
            "suggestion_length": 30,
            "cluster_name_length": 10,
            "cluster_description_length": 30,
            "explanation_length": 20,
            "max_num_clusters": 25,
        },
        # We batch summarize the docs. To avoid getting errors, we will limit the
        # degree of parallelism to permit.
        "max_concurrency": 2,
    },
)

for step in stream:
    node, state = next(iter(step.items()))
    print(node, str(state)[:20] + " ...")
```

## Final Result

Below, render the final result as markdown:



```python
from IPython.display import Markdown


def format_taxonomy_md(clusters):
    md = "## Final Taxonomy\n\n"
    md += "| ID | Name | Description |\n"
    md += "|----|------|-------------|\n"

    # Fill the table with cluster data
    for label in clusters:
        id = label["id"]
        name = label["name"].replace(
            "|", "\\|"
        )  # Escape any pipe characters within the content
        description = label["description"].replace(
            "|", "\\|"
        )  # Escape any pipe characters
        md += f"| {id} | {name} | {description} |\n"

    return md


Markdown(format_taxonomy_md(step["__end__"]["clusters"][-1]))
```




## Final Taxonomy

| ID | Name | Description |
|----|------|-------------|
| 1 | Troubleshooting Network Connectivity Issues | Resolving problems with DNS, network connections, and GitHub extension activation. |
| 2 | Extracting and Analyzing Data | Retrieving and processing data from various sources like text files, databases, and APIs. |
| 3 | Providing Healthcare Insights | Generating medical diagnosis, symptom checking, drug information, and skin condition analysis. |
| 4 | Configuring and Optimizing Models | Adjusting model parameters and hyperparameters to improve performance for a given task. |
| 5 | Generating Creative Poetry | Creating poems using language models and AI-powered tools. |
| 6 | Interacting with Databases | Querying databases, extracting data, and managing errors during data processing. |
| 7 | Querying Vector Databases | Interacting with vector databases like Milvus to store and retrieve high-dimensional data. |
| 8 | Generating Synthetic Data | Creating synthetic data using language models and machine learning techniques. |
| 9 | Integrating Tools and Workflows | Incorporating various tools and libraries into a cohesive workflow for different tasks. |
| 10 | Improving Information Retrieval | Storing and querying multiple vectors per document for better semantic understanding. |
| 11 | Processing Documents and Extracting Text | Parsing and extracting text from various document formats like PDF, DOCX, and HTML. |
| 12 | Building Local Knowledge Bases | Creating knowledge bases from text files, handling text splitting, embeddings, and storage. |
| 13 | Optimizing Conversational Retrieval | Troubleshooting and improving the performance of the ConversationalRetrievalChain in LangChain. |
| 14 | Connecting Databases and Using Agents | Connecting to databases, using agents, and understanding the differences between agent types. |
| 15 | Introspecting LangChain Tools | Accessing and retrieving details about the functions and source code of LangChain tools. |
| 16 | Generating Styled Answers with Retrieval Augmentation | Creating a QA system that generates well-cited answers in a specific style. |
| 17 | Using ZERO_SHOT_REACT_DESCRIPTION Agents | Applying the ZERO_SHOT_REACT_DESCRIPTION agent type in LangChain for chat models. |
| 18 | Automating Microlearning Course Creation | Generating microlearning courses based on input parameters like topic, volume, and learning style. |
| 19 | Integrating with Chroma Vector Store | Storing and retrieving data in the Chroma vector database, including handling document embeddings. |
| 20 | Managing LangChain Callback Tokens | Understanding and utilizing the callback token feature in the LCEL chain. |
| 21 | Troubleshooting FastAPI Deployments | Resolving issues with deploying a React app with a FastAPI backend. |
| 22 | Analyzing Data with LangChain Agents | Using LangChain agents to interact with Pandas and Spark DataFrames for data exploration. |
| 23 | Implementing the OpenAI Chat API | Implementing the OpenAI chat completion API and understanding the required inputs and outputs. |
| 24 | Comparing LangChain and LLMIndex | Evaluating the differences between LangChain and LLMIndex, including their UI support for Markdown. |
| 25 | Suppressing Tools in AgentExecutor | Temporarily disabling tools in an AgentExecutor for a fixed number of invocations. |




## Phase 2: Labeling

Now that we have our taxonomy, it's time to label a subset of our data to train a classifier.

Input classification can be useful for anything from in-line prompt optimization (tailor the prompt for each classified intent), to system improvements (identifying categories for which the system doesn't produce good responses) to product analytics (understand which intent categories could be improved to drive profits).

The problem is that LLM-based tagging can be expensive.

Embeddings can be ~100x cheaper to compute, and a simple logistic regression classifier on top of that would add negligible cost.

Let's tag and train a classifier!

#### Label Training Data

Use an LLM to label the data in a fully-automated fashion. For better accuracy, you can sample a portion of the results to label by hand as well to verify the quality.



```python
labeling_prompt = hub.pull("wfh/tnt-llm-classify")

labeling_llm = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens_to_sample=2000)
labeling_llm_chain = (labeling_prompt | labeling_llm | StrOutputParser()).with_config(
    run_name="ClassifyDocs"
)


def parse_labels(output_text: str) -> Dict:
    """Parse the generated labels from the predictions."""
    category_matches = re.findall(
        r"\s*<category>(.*?)</category>.*",
        output_text,
        re.DOTALL,
    )
    categories = [{"category": category.strip()} for category in category_matches]
    if len(categories) > 1:
        logger.warning(f"Multiple selected categories: {categories}")
    label = categories[0]
    stripped = re.sub(r"^\d+\.\s*", "", label["category"]).strip()
    return {"category": stripped}


labeling_chain = labeling_llm_chain | parse_labels
```


```python
final_taxonomy = step["__end__"]["clusters"][-1]
xml_taxonomy = format_taxonomy(final_taxonomy)
results = labeling_chain.batch(
    [
        {
            "content": doc["content"],
            "taxonomy": xml_taxonomy,
        }
        for doc in docs
    ],
    {"max_concurrency": 5},
    return_exceptions=True,
)
# Update the docs to include the categories
updated_docs = [{**doc, **category} for doc, category in zip(docs, results)]
```


```python
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OPENAI_API_KEY: ")
```


```python
from langchain_openai import OpenAIEmbeddings

# Consider using other embedding models here too!
encoder = OpenAIEmbeddings(model="text-embedding-3-large")
vectors = encoder.embed_documents([doc["content"] for doc in docs])
embedded_docs = [{**doc, "embedding": v} for doc, v in zip(updated_docs, vectors)]
```

#### Train Classifier

Now that we've extracted the features from the text, we can generate the classifier on them.



```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Create a dictionary mapping category names to their indices in the taxonomy
category_to_index = {d["name"]: i for i, d in enumerate(final_taxonomy)}
category_to_index["Other"] = len(category_to_index)
# Convert category strings to numeric labels
labels = [
    category_to_index.get(d["category"], category_to_index["Other"])
    for d in embedded_docs
]

label_vectors = [d["embedding"] for d in embedded_docs]

X_train, X_test, y_train, y_test = train_test_split(
    label_vectors, labels, test_size=0.2, random_state=42
)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Weight the classes to partially handle imbalanced data
model = LogisticRegression(class_weight=class_weight_dict)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
train_f1 = f1_score(y_train, train_preds, average="weighted")
test_f1 = f1_score(y_test, test_preds, average="weighted")

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Train F1 Score: {train_f1:.3f}")
print(f"Test F1 Score: {test_f1:.3f}")
```

## Phase 3: Deploy

Now that you have your classifier, you can easily deploy it and apply to future runs! All you need is to embed the input and apply your LogisticRegression classifier. Let's try it. We will use python's [joblib](https://joblib.readthedocs.io/en/stable/) library to serialize our sklearn classifier. Below is an example:



```python
from joblib import dump as jl_dump

categories = list(category_to_index)

# Save the model and categories to a file
with open("model.joblib", "wb") as file:
    jl_dump((model, categories), file)
```

#### To deploy

When deploying, you can load the classifier and initialize your embeddings encoder. They fit together easily using LCEL:



```python
from joblib import load as jl_load
from langchain_openai import OpenAIEmbeddings

loaded_model, loaded_categories = jl_load("model.joblib")
encoder = OpenAIEmbeddings(model="text-embedding-3-large")


def get_category_name(predictions):
    return [loaded_categories[pred] for pred in predictions]


classifier = (
    RunnableLambda(encoder.embed_documents, encoder.aembed_documents)
    | loaded_model.predict
    | get_category_name
)
```

#### Example:

Assuming you've had some more data come in, you can fetch it and apply it below



```python
client = Client()

past_5_min = datetime.now() - timedelta(minutes=5)
runs = list(
    client.list_runs(
        project_name=project_name,
        filter="eq(is_root, true)",
        start_time=past_5_min,
        # We only need to return the inputs + outputs
        select=["inputs", "outputs"],
        limit=100,
    )
)
docs = [run_to_doc(r) for r in runs]
```


```python
classes = classifier.invoke([doc["content"] for doc in docs])
print(classes[:2])
```

## Conclusion

Congrats on implementing TNT-LLM! While most folks use clustering-based approaches like LDA, k-means, etc. it can often be hard to really interpret what each cluster represents. TNT-LLM generates human-interpretable labels you can use downstream to monitor and improve your application.

The technique also lends itself to hierarchical sub-categorizing: once you have the above taxonomy, use it to label your data, then on each sub-category, generate a new taxonomy using a similar technique to the one described above!

