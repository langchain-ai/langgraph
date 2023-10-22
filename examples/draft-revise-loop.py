from __future__ import annotations

from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from permchain import Pregel, channels
from permchain.pregel import PregelIO

# prompts

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

editor_functions = [
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

# llms

gpt3 = ChatOpenAI(model="gpt-3.5-turbo")
gpt4 = ChatOpenAI(model="gpt-4")

# chains

drafter_chain = (
    {"question": RunnablePassthrough(input_type=str)}
    | drafter_prompt
    | gpt3
    | StrOutputParser()
)

editor_chain = (
    editor_prompt
    | gpt4.bind(functions=editor_functions)
    | JsonOutputFunctionsParser(args_only=False)
)

reviser_chain = reviser_prompt | gpt3 | StrOutputParser()

# application

draft_revise_loop = Pregel(
    drafter_chain | Pregel.send_to("draft"),
    chains={
        "editor": (
            # subscribe to draft channel as a dict with a single key, "draft"
            Pregel.subscribe_to(["draft"])
            | editor_chain
            | Pregel.send_to(
                # send to "notes" channel if the editor does not accept the draft
                notes=lambda x: x["arguments"]["notes"]
                if x["name"] == "revise"
                else None
            )
        ),
        "reviser": (
            # subscribe to new values of "notes" channel,
            # and join them with the input value (question) and "draft"
            Pregel.subscribe_to(["notes"]).join([PregelIO.IN, "draft"])
            | reviser_chain
            | Pregel.send_to("draft")
        ),
    },
    channels={
        "draft": channels.LastValue(str),
        "notes": channels.LastValue(str),
    },
    # output will be a dict with keys "draft" and "notes"
    output=["draft", "notes"],
    # debug logging
    debug=True,
)

# run

for draft in draft_revise_loop.stream("What food do turtles eat?"):
    print(draft)
    print("---")


async def main():
    async for draft in draft_revise_loop.astream("What food do turtles eat?"):
        print(draft)
        print("---")


# import asyncio

# asyncio.run(main())
