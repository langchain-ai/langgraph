from operator import itemgetter

import requests
from fastapi import FastAPI
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.runnables.openai_functions import OpenAIFunctionsRouter
from langchain.schema.output_parser import StrOutputParser

from permchain.connection_inmemory import InMemoryPubSubConnection
from permchain.pubsub import PubSub
from permchain.topic import Topic

template = """Write between 2 and 5 sub questions that serve as google search queries to search online that form an objective opinion from the following: {question}"""
functions = [
    {
        "name": "sub_questions",
        "description": "List of sub questions",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "List of sub questions to ask.",
                    "items": {"type": "string"},
                },
            },
        },
    },
]
prompt = ChatPromptTemplate.from_template(template)
question_chain = (
    prompt
    | ChatOpenAI(temperature=0).bind(
        functions=functions, function_call={"name": "sub_questions"}
    )
    | JsonKeyOutputFunctionsParser(key_name="questions")
)

template = """You are tasked with writing a research report to answer the following question:

<question>
{question}
</question>

In order to do that, you first came up with several sub questions and researched those. please find those below:

<research>
{research}
</research>

Now, write your final report answering the original question!"""
prompt = ChatPromptTemplate.from_template(template)
report_chain = prompt | ChatOpenAI() | StrOutputParser()

research_inbox = Topic("research")
writer_inbox = Topic("writer_inbox")


def web_researcher(questions):
    response = requests.post(
        "http://127.0.0.1:8081/batch", json={"questions": questions}
    )
    return response.json()


subquestion_actor = (
    # Listed in inputs
    Topic.IN.subscribe()
    | question_chain
    # The draft always goes to the editors inbox
    | research_inbox.publish()
)
research_actor = (
    research_inbox.subscribe()
    | {
        "research": lambda x: web_researcher(x),
        # "research": (lambda x: [web_researcher(i) for i in x]),
        "question": Topic.IN.current() | itemgetter("question"),
    }
    | writer_inbox.publish()
)
write_actor = (
    writer_inbox.subscribe() | {"response": report_chain} | Topic.OUT.publish()
)

longer_researcher = PubSub(
    processes=(subquestion_actor, research_actor, write_actor),
    connection=InMemoryPubSubConnection(),
)

app = FastAPI()


@app.get("/report")
def read_item(question: str):
    return longer_researcher.invoke({"question": question})
