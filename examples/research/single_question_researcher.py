from operator import itemgetter
from typing import List

import requests
from fastapi import FastAPI
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.runnables.openai_functions import OpenAIFunctionsRouter
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel

from permchain.connection_inmemory import InMemoryPubSubConnection
from permchain.pubsub import PubSub
from permchain.topic import Topic

prompt = ChatPromptTemplate.from_template(
    "Answer the user's question given the search results\n\n<question>{question}</question><search_results>{search_results}</search_results>"
)

summarizer_chain = (
    prompt
    | ChatOpenAI(max_retries=0).with_fallbacks(
        [ChatOpenAI(model="gpt-3.5-turbo-16k"), ChatAnthropic(model="claude-2")]
    )
    | StrOutputParser()
)


def retrieve_documents(query):
    response = requests.get("http://127.0.0.1:8080/query", params={"query": query})
    return response.json()


summarizer_inbox = Topic("summarizer")

search_actor = (
    Topic.IN.subscribe()
    | {
        "search_results": retrieve_documents,
        "question": Topic.IN.current(),
    }
    | summarizer_inbox.publish()
)

summ_actor = (
    summarizer_inbox.subscribe() | {"answer": summarizer_chain} | Topic.OUT.publish()
)

web_researcher = PubSub(
    processes=(search_actor, summ_actor),
    connection=InMemoryPubSubConnection(),
)

app = FastAPI()


class Data(BaseModel):
    questions: List[str]


@app.get("/invoke")
def read_item(question: str):
    return web_researcher.invoke(question)


@app.post("/batch")
def batch(data: Data):
    return web_researcher.batch(data.questions)
