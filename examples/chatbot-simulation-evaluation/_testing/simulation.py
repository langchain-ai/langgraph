from typing import List

import openai
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from simulation_utils import (
    create_chat_simulator,
    create_simulated_user,
    langchain_to_openai_messages,
)

openai_client = openai.Client()


def my_chat_bot(messages: list) -> str:
    oai_messages = langchain_to_openai_messages(messages)
    system_message = {
        "role": "system",
        "content": "You are a customer support agent for an airline.",
    }
    messages = [system_message] + oai_messages
    completion = openai_client.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo"
    )
    return completion.choices[0].message.content


my_chat_bot([{"role": "user", "content": "hi!"}])


system_prompt_template = """You are a customer of an airline company. \
You are interacting with a user who is a customer support person. \

{instructions}

Your task is to get a big discount on your next flight. \

When you are finished with the conversation, respond with a single word 'FINISHED'"""

simulated_user = create_simulated_user(
    system_prompt_template, llm=ChatOpenAI(model="gpt-3.5-turbo")
)

# my chat bot accepts a list of LangChain mesages
# Simulated user accepts a list of LangChain messages
# TODO: Pass additional arguments to the simulated user
simulator = create_chat_simulator(my_chat_bot, simulated_user)
simulator.invoke(
    {
        "instructions": "You are extremely disgruntled and will cusss and swear to get your way. Try to get a discount by any means necessary."
    }
)
