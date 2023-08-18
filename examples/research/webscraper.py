# main.py

from duckduckgo_search import DDGS
from fastapi import FastAPI
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

ddgs = DDGS()

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/query")
def read_item(query: str):
    query = query.strip().strip('"')
    search_results = ddgs.text(query)
    urls_to_look = []
    for res in search_results:
        if res.get("href", None):
            urls_to_look.append(res["href"])
        if len(urls_to_look) >= 4:
            break

    # Relevant urls
    # Load, split, and add new urls to vectorstore
    if urls_to_look:
        loader = AsyncHtmlLoader(urls_to_look)
        html2text = Html2TextTransformer()
        docs = loader.load()
        docs = list(html2text.transform_documents(docs))
    else:
        docs = []
    return docs
