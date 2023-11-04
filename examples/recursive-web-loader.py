from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Callable, FrozenSet, Generator, Optional, TypedDict

import httpx
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.utils.html import extract_sub_links

from permchain import Channel, Pregel
from permchain.channels.context import Context
from permchain.channels.topic import Topic

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
    # define the channels
    channels = {
        "next_urls": Topic(str, unique=True),
        "documents": Topic(Document, accumulate=True),
        "client": Context(httpx_client, httpx_aclient),
    }
    # the main chain that gets executed recursively
    # while there are urls in next_urls
    visitor = (
        # run the chain below for each url in next_urls
        # adding the current values of base_url and httpx client
        Channel.subscribe_to_each("next_urls", key="url").join(["client", "base_url"])
        # load the url (with sync and async implementations)
        | RunnablePassthrough.assign(body=RunnableLambda(load_url, load_url_async))
        | Channel.write_to(
            # send a new document to the documents stream
            documents=lambda x: Document(
                page_content=extractor(x["body"]),
                metadata=metadata_extractor(x["body"], x["url"]),
            ),
            # send the next urls to the next_urls topic
            next_urls=lambda x: extract_sub_links(
                x["body"], x["url"], base_url=x["base_url"]
            ),
        )
    )
    return Pregel(
        channels=channels,
        chains={
            # use the base_url as the first url to visit
            "input": Channel.subscribe_to("base_url") | Channel.write_to("next_urls"),
            # add the main chain
            "visitor": visitor,
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
