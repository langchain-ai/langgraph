from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Callable, FrozenSet, Generator, Optional, TypedDict

import httpx
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.utils.html import extract_sub_links

from permchain import Pregel, channels

# Load url with sync httpx client


@contextmanager
def httpx_client() -> Generator[httpx.Client, None, None]:
    with httpx.HTTPTransport(retries=3) as transport, httpx.Client(
        transport=transport
    ) as client:
        yield client


class LoadUrlInput(TypedDict):
    url: str
    visited: FrozenSet[str]
    client: httpx.Client


def load_url(input: LoadUrlInput) -> str:
    response = input["client"].get(input["url"])
    return response.text


# Same as above but with async httpx client


@asynccontextmanager
async def httpx_aclient() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncHTTPTransport(retries=3) as transport, httpx.AsyncClient(
        transport=transport
    ) as client:
        yield client


class LoadUrlInputAsync(TypedDict):
    url: str
    visited: FrozenSet[str]
    client: httpx.AsyncClient


async def load_url_async(input: LoadUrlInputAsync) -> str:
    response = await input["client"].get(input["url"])
    return response.text


# default metadata extractor copied from langchain.document_loaders


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
    # assign default extractors
    extractor = extractor or (lambda x: x)
    metadata_extractor = metadata_extractor or _metadata_extractor
    # the main chain that gets executed recursively
    visitor = (
        # while there are urls in next_urls
        # run the chain below for each url in next_urls
        # adding the current values of visited set, base_url and httpx client
        Pregel.subscribe_to_each("next_urls", key="url").join(
            ["visited", "client", "base_url"]
        )
        # load the url (with sync and async implementations)
        | RunnablePassthrough.assign(body=RunnableLambda(load_url, load_url_async))
        | Pregel.write_to(
            # send this url to the visited set
            visited=lambda x: x["url"],
            # send a new document to the documents stream
            documents=lambda x: Document(
                page_content=extractor(x["body"]),
                metadata=metadata_extractor(x["body"], x["url"]),
            ),
            # send the next urls to the next_urls set
            # only if not visited already
            next_urls=lambda x: [
                url
                for url in extract_sub_links(
                    x["body"], x["url"], base_url=x["base_url"]
                )
                if url not in x["visited"] and url != x["url"]
            ],
        )
    )
    return Pregel(
        chains={
            # use the base_url as the first url to visit
            "input": Pregel.subscribe_to("base_url") | Pregel.write_to("next_urls"),
            # add the main chain
            "visitor": visitor,
        },
        # define the channels
        channels={
            "base_url": channels.LastValue(str),
            "next_urls": channels.UniqueInbox(str),
            "documents": channels.Stream(Document),
            "visited": channels.Set(str),
            "client": channels.ContextManager(httpx_client, httpx_aclient),
        },
        # this will accept a string as input
        input="base_url",
        # and return a dict with documents and visited set
        output=["documents", "visited"],
        # debug logging
        debug=True,
    ).with_config({"recursion_limit": max_depth + 1})


loader = recursive_web_loader(max_depth=3)

documents = loader.invoke("https://docs.python.org/3.9/")

print(len(documents["documents"]))
