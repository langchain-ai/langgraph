from __future__ import annotations

from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser

from permchain import Pregel, channels

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

drafter_chain = drafter_prompt | gpt3 | StrOutputParser()

reviser_chain = reviser_prompt | gpt3 | StrOutputParser()

editor_chain = (
    editor_prompt
    | gpt4.bind(functions=editor_functions)
    | JsonOutputFunctionsParser(args_only=False)
)

# state

question = channels.LastValue[str]("question")

draft = channels.LastValue[str]("draft")

notes = channels.LastValue[str]("notes")

# application

drafter_node = (
    Pregel.subscribe_to(question=question) | drafter_chain | Pregel.send_to(draft)
)

editor_node = (
    Pregel.subscribe_to(draft=draft)
    | editor_chain
    | Pregel.send_to(
        {notes: lambda x: x["arguments"]["notes"] if x["name"] == "revise" else None}
    )
)

reviser_node = (
    Pregel.subscribe_to(notes=notes).join(question=question, draft=draft)
    | reviser_chain
    | Pregel.send_to(draft)
)

draft_revise_loop = Pregel(
    (drafter_node, reviser_node, editor_node),
    input=question,
    output=draft,
)

# run

for draft in draft_revise_loop.stream("What food do turtles eat?"):
    print('Draft: "' + draft + '"')
    print("---")
