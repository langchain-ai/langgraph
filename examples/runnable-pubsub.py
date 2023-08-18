from operator import itemgetter
from pprint import pprint

from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.runnables.openai_functions import OpenAIFunctionsRouter
from langchain.schema.output_parser import StrOutputParser

from permchain.connection_inmemory import InMemoryPubSubConnection
from permchain.pubsub import PubSub
from permchain.topic import Topic

drafter_prompt = (
    SystemMessagePromptTemplate.from_template(
        "You are an expert on turtles, who likes to write in pirate-speak. You have been tasked by your editor with drafting a 100-word article answering the following question."
    )
    + "Question:\n\n{question}"
)

reviser_prompt = (
    SystemMessagePromptTemplate.from_template(
        "You are an expert on turtles. You have been tasked by your editor with revising the following draft, which was written by a non-expert. You may follow the editor's notes or not, as you see fit."
    )
    + "Draft:\n\n{draft}"
    + "Editor's notes:\n\n{notes}"
)

editor_prompt = (
    SystemMessagePromptTemplate.from_template(
        "You are an editor. You have been tasked with editing the following draft, which was written by a non-expert. Please accept the draft if it is good enough to publish, or send it for revision, along with your notes to guide the revision."
    )
    + "Draft:\n\n{draft}"
)


drafter_llm = ChatOpenAI(model="gpt-3.5-turbo")
editor_llm = ChatOpenAI(model="gpt-4")

# create topics
editor_inbox = Topic("editor_inbox")
reviser_inbox = Topic("reviser_inbox")


# write a first draft
drafter = (
    Topic.IN.subscribe()
    | {"draft": drafter_prompt | drafter_llm | StrOutputParser()}
    | editor_inbox.publish()
)

# edit every draft, produce revision notes or accept
editor = (
    editor_inbox.subscribe()
    | editor_prompt
    | editor_llm.bind(
        functions=[
            {
                "name": "revise",
                "description": "Sends the draft for revision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "description": "The editor's notes to guide the revision.",
                        },
                    },
                },
            },
            {
                "name": "accept",
                "description": "Accepts the draft",
                "parameters": {
                    "type": "object",
                    "properties": {"ready": {"const": True}},
                },
            },
        ]
    )
    | OpenAIFunctionsRouter(
        {
            "revise": (
                {
                    "notes": itemgetter("notes"),
                    "draft": editor_inbox.current() | itemgetter("draft"),
                    "question": Topic.IN.current() | itemgetter("question"),
                }
                | reviser_inbox.publish()
            ),
            "accept": editor_inbox.current() | Topic.OUT.publish(),
        },
    )
)

# every time revision notes are posted, revise latest draft
reviser = (
    reviser_inbox.subscribe()
    | {"draft": reviser_prompt | drafter_llm | StrOutputParser()}
    | editor_inbox.publish()
)

web_researcher = PubSub(
    processes=(drafter, editor, reviser),
    connection=InMemoryPubSubConnection(),
)

# for output in web_researcher.stream({"question": "What food do turtles eat?"}):
#     print("got output", output)

# print("---done with stream()---")

pprint(
    web_researcher.batch(
        [{"question": "What food do turtles eat?"}, {"question": "What is art?"}]
    )
)


# agent = PubSub(
#     Channel.IN | Channel("planner"),
#     Channel("executor") | executor | Channel("planner"),
#     Channel("planner")
#     | planner
#     | {"action": Channel("executor"), "finish": Channel.OUT},
# )

# graph = (
#     drafter
#     | editor
#     | RouterRunnable(
#         {
#             "send_for_revision": reviser,
#             "accept_draft": lambda x: x["draft"],
#         }
#     )
# )
