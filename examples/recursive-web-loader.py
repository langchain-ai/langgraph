from typing import Callable, FrozenSet, Optional, TypedDict

import httpx
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.utils.html import extract_sub_links

from permchain import Pregel, channels
from permchain.pregel import PregelRead


class LoadUrlInput(TypedDict):
    url: str
    visited: FrozenSet[str]
    client: httpx.Client


class LoadUrlInputAsync(TypedDict):
    url: str
    visited: FrozenSet[str]
    client: httpx.AsyncClient


def load_url(input: LoadUrlInput) -> str:
    response = input["client"].get(input["url"])
    return response.text


async def load_url_async(input: LoadUrlInputAsync) -> str:
    response = await input["client"].get(input["url"])
    return response.text


def _metadata_extractor(raw_html: str, url: str) -> dict:
    """Extract metadata from raw html using BeautifulSoup."""
    metadata = {"source": url}

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return metadata
    soup = BeautifulSoup(raw_html, "html.parser")
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", None)
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", None)
    return metadata


def recursive_web_loader(
    *,
    max_depth: int = 2,
    extractor: Optional[Callable[[str], str]] = None,
    metadata_extractor: Optional[Callable[[str, str], dict]] = None,
) -> Pregel:
    extractor = extractor or (lambda x: x)
    metadata_extractor = metadata_extractor or _metadata_extractor
    chain = (
        Pregel.subscribe_to_each("next_urls")
        | {
            "url": RunnablePassthrough(),
            "visited": PregelRead("visited"),
            "client": PregelRead("client"),
            "base_url": PregelRead("url"),
        }
        | RunnablePassthrough.assign(body=RunnableLambda(load_url, load_url_async))
        | Pregel.send_to(
            visited=lambda x: x["url"],
            documents=lambda x: Document(
                page_content=extractor(x["body"]),
                metadata=metadata_extractor(x["body"], x["url"]),
            ),
            next_urls=lambda x: [
                url
                for url in extract_sub_links(
                    x["body"], x["url"], base_url=x["base_url"]
                )
                if url not in x["visited"] and url != x["url"]
            ],
            _max_steps=max_depth,
        )
    )
    return Pregel(
        Pregel.subscribe_to("url") | Pregel.send_to("next_urls"),
        chain,
        channels={
            "url": channels.LastValue(str),
            "next_urls": channels.UniqueInbox(str),
            "documents": channels.Stream(Document),
            "visited": channels.Set(str),
            "client": channels.ContextManager(
                httpx.Client | httpx.AsyncClient, httpx.Client, httpx.AsyncClient
            ),
        },
        input="url",
        output="visited",
    )


loader = recursive_web_loader(max_depth=3)

documents = loader.invoke("https://docs.python.org/3.9/")

print(documents)
