from typing import Annotated

class MainAgentState(TypedDict):
    input: str
    output: list[str]

advanced_flow = AdvancedStateGraph(MainAgentState)
advanced_flow.add_async_channel("inbox", str) # default to infinite buffer like Rust channel

prompt = "based on current state, decide to use tool, or kick off subagent, or complete"
main_llm = new_mock_llm(prompt, [email_tool, slack_tool])

async def llm_node(state: MainAgentState):
    decisions = await main_llm.ainvoke(state)
    sends = []
    for decision in decisions:
        if decision.type == "end":
            return Command(goto=Send("end", decision.complete)) # NOTE: we can simplify to just Complete(decision.complete)
        elif decision.type == "sub_agent":
            sends.append(Send("sub_agent", decision.sub_agent)) # NOTE: we can simplify to just sends.apppend(subagent, decision.sub_agent)
        elif decision.type == "tool":
            sends.append(Send("tool", decision.tool))
    sends.append(Send("wait_node"))
    return Command(goto=sends)        

async def wait_node(state: MainAgentState):
    # wait for at least one message on the inbox channel
    # this is a "lightweight" interrupt, that does not block the entire graph
    msgs = wait_for("inbox") 

    output = state.output
    if msg.type == "tool":
        output.append("tool: " + msg.payload)
    elif msg.type == "sub_agent":
        output.append("sub_agent: " + msg.payload)
    elif msg.type == "user_input":
        output.append("user_input: " + msg.payload)
    
    # loop back to llm node with new output
    return Command(goto=Send("llm_node", output))


async def tool_node(tool_input: str):
    await asyncio.sleep(5)
    output = "tool completed for: " + tool_input
    publish_to_channel("inbox", {"type": "tool", "payload": output})
    # just complete without going to next node

async def order_food_node(state: str):
    return "order_food_node completed for: " + state

# sub agent uses regular/simple state graph 
sub_agent = StateGraph(str)
async def research_node (state: str):
    await asyncio.sleep(10)
    return "research sub agent completed for: " + state
sub_agent.add_node("research_node", research_node)
sub_agent.add_edge(START, "research_node")
sub_agent.add_edge("research_node", END)

async def sub_agent_node(sub_agent_input: str):
    sub_agent_output = sub_agent.invoke({"input": sub_agent_input})
    publish_to_channel("inbox", {"type": "sub_agent", "payload": sub_agent_output})
    # just complete without going to next node

advanced_flow.add_node("llm_node", llm_node) 
advanced_flow.add_node("wait_node", wait_node) 
advanced_flow.add_node("tool_node", tool_node)
advanced_flow.add_node("sub_agent_node", sub_agent_node)
advanced_flow.add_node("order_food_node", order_food_node)
advanced_flow.set_entry_point("llm_node")
advanced_flow.set_finish_point("order_food_node")

## NOTE: above can be simplified to:
# advanced_flow.add_entry_node(llm_node)
# advanced_flow.node(wait_node)
# advanced_flow.node(tool_node)
# advanced_flow.node(sub_agent_node)
# advanced_flow.add_finish_node(order_food_node)

main_agent = advanced_flow.compile()

async def test_async_sub_graph():
    main_llm.mock_response = [
        # first llm invoke
        [
            {
                "type": "sub_agent",
                "sub_agent": "research_node"
            },
                        {
                "type": "tool",
                "tool": "slack_tool"
            }
        ],
        # 2nd llm invoke, after additional user input
        [
            {
                "type": "sub_agent",
            }
        ],
        # 3rd llm invoke, after tool node completes
        [], 
        # 4th llm invoke, after 1st sub agent node completes
        [],
        # 5th llm invoke, after 2nd sub agent node completes
        [
            {
                "type": "end"
            }
        ],
    ]

    started = main_agent.ainvoke({"input": "help me get something for lunch"})
    # provide more info after 2 seconds
    await asyncio.sleep(2)
    main_agent.apublish_to_channel("inbox", {"type": "user_input", "payload": "No spicy food please"})
    result = await started
    assert result == {
        "input": "help me get something for lunch",
        "output":[

        ]
    }