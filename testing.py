# import asyncio
# from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent
# from langchain_core.tools import tool
# from langgraph.graph import END, START, StateGraph
# from langgraph.graph import MessagesState


# @tool
# def get_weather(city: str) -> str:
#     """
#     Get the weather of a city
#     """
#     return f"The weather of {city} is sunny."


# agent = create_react_agent(
#     model=ChatOpenAI(
#         model="gpt-4.1-mini",
#         temperature=0,
#     ),
#     prompt="""
#             You are a helpful travel assistant that can help user to get travel information.
#             When providing travel information, please also include:
#             1. Top tourist attractions and landmarks
#             2. Any travel recommendations based on the city weather
#             """,
#     tools=[get_weather],
# )


# async def node(state: MessagesState) -> MessagesState:
#     print("BEGIN")
#     msg_content = ""

#     async for ns, msg in agent.astream(
#         {
#             "messages": [
#                 ("user", state["messages"][-1].content),
#             ]
#         },
#         stream_mode="messages",
#         # subgraphs=True,

#     ):
#         msg_content += msg[0].content
#     print("END")

#     return {"messages": [("assistant", msg_content)]}


# graph = StateGraph(state_schema=MessagesState)
# graph.add_node("node", node)
# graph.add_edge(START, "node")
# graph.add_edge("node", END)
# workflow = graph.compile()

# async def main():
#     result = await workflow.ainvoke({"messages": [("user", "What is the weather in Tokyo?")]})
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(main())

from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o-mini", message_version="v1")