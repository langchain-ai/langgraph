"""Tutorial 05: Multi-Agent Collaboration with Shared Memory

This tutorial demonstrates multiple AI agents working in parallel on a complex
problem, sharing their findings and building on each other's work through
Kusto checkpoints.

Scenario: Research Team
- Multiple specialist agents investigate different aspects of a topic
- Each agent can see what others have discovered
- Agents build on each other's findings
- All research is persisted in Kusto for future reference

Requirements:
    - Azure OpenAI or OpenAI API key
    - Kusto cluster with langgraph database
    - Tables created via provision.kql
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import TypedDict, Annotated
from operator import add

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph.checkpoint.kusto import AsyncKustoSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# Configure environment
CLUSTER_URI = os.getenv(
    "KUSTO_CLUSTER_URI",
    "https://your-cluster.region.kusto.windows.net"
)
DATABASE = os.getenv("KUSTO_DATABASE", "langgraph")


def merge_findings(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Merge findings from multiple agents.
    
    Args:
        left: Existing findings
        right: New findings to merge
        
    Returns:
        Merged findings dictionary
    """
    if not left:
        return right
    if not right:
        return left
    
    # Merge dictionaries, right values take precedence
    merged = left.copy()
    merged.update(right)
    return merged


def merge_research_history(
    left: dict[str, list[str]], 
    right: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Merge research history from multiple agents running in parallel.
    
    Each agent updates its own list of findings. This reducer ensures
    parallel updates are combined correctly.
    
    Args:
        left: Existing research history
        right: New research history to merge
        
    Returns:
        Merged history with all agent histories combined
    """
    if not left:
        return right
    if not right:
        return left
    
    # Merge dictionaries, combining lists for each agent
    merged = left.copy()
    for agent_name, findings_list in right.items():
        if agent_name in merged:
            # Append new findings to existing list
            merged[agent_name] = merged[agent_name] + findings_list
        else:
            merged[agent_name] = findings_list
    return merged


class ResearchState(TypedDict):
    """Shared state for research agents."""
    messages: Annotated[list, add_messages]
    topic: str
    research_phase: str
    findings: Annotated[dict[str, str], merge_findings]  # Now can handle parallel updates
    research_history: Annotated[dict[str, list[str]], merge_research_history]  # Persistent across sessions


def create_specialist_agent(
    name: str,
    specialty: str,
    llm: ChatOpenAI | AzureChatOpenAI
):
    """Create a specialist research agent.
    
    Args:
        name: Agent name (e.g., "technical_analyst")
        specialty: Agent's area of expertise
        llm: Language model to use
        
    Returns:
        Agent function that performs research
    """
    
    async def agent_function(state: ResearchState) -> ResearchState:
        """Research agent that investigates topic from its specialty."""
        
        # Get current findings from other agents (this execution)
        other_findings = "\n".join([
            f"- {agent}: {findings}"
            for agent, findings in state.get("findings", {}).items()
            if agent != name
        ])
        
        # Get historical findings from Kusto (previous executions)
        research_history = state.get("research_history", {})
        agent_history = research_history.get(name, [])
        other_agent_history = {
            agent: history
            for agent, history in research_history.items()
            if agent != name and history
        }
        
        # Build context-aware prompt with both current and historical context
        context_parts = []
        
        if other_findings:
            context_parts.append(f"Current session - Other team members found:\n{other_findings}")
        
        if agent_history:
            context_parts.append(
                f"\nYour previous findings from Kusto:\n" + 
                "\n".join([f"  - {finding[:100]}..." for finding in agent_history[-3:]])  # Last 3
            )
        
        if other_agent_history:
            history_summary = "\n".join([
                f"- {agent}: {len(history)} previous finding(s)"
                for agent, history in other_agent_history.items()
            ])
            context_parts.append(f"\nOther agents' research history in Kusto:\n{history_summary}")
        
        context = ""
        if context_parts:
            context = "\n\n" + "\n".join(context_parts) + "\n\nBuild on these insights with new perspectives."
        
        system_prompt = (
            f"You are {name.replace('_', ' ').title()}, a specialist in {specialty}. "
            f"Provide concise, insightful analysis from your expertise. "
            f"Be specific and cite examples. Reference previous findings when relevant.{context}"
        )
        
        # Get the research topic
        topic = state.get("topic", "")
        phase = state.get("research_phase", "initial")
        
        # Create research query based on phase
        if phase == "initial":
            query = f"Analyze {topic} from a {specialty} perspective. What are the key aspects?"
        elif phase == "deep_dive":
            query = f"Given what we know about {topic}, dive deeper into {specialty} implications. What insights can you add?"
        else:
            query = f"Synthesize your {specialty} findings about {topic}. What's most important?"
        
        # Query the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = await llm.ainvoke(messages)
        
        # Update state with this agent's findings
        # The merge_findings reducer will combine with other agents
        agent_findings = {name: response.content}
        
        # Update research history - append to this agent's history
        updated_history = research_history.copy()
        if name not in updated_history:
            updated_history[name] = []
        updated_history[name] = updated_history[name] + [response.content]
        
        # Add to message history
        new_messages = [
            HumanMessage(content=f"[{name}] {query}"),
            AIMessage(content=f"[{name}] {response.content}")
        ]
        
        return {
            "messages": new_messages,
            "findings": agent_findings,
            "research_history": updated_history,
        }
    
    return agent_function


async def synthesizer_agent(
    state: ResearchState,
    llm: ChatOpenAI | AzureChatOpenAI
) -> ResearchState:
    """Synthesize all agent findings into a comprehensive analysis."""
    
    findings = state.get("findings", {})
    if not findings:
        return state
    
    # Compile all findings
    findings_text = "\n\n".join([
        f"**{agent.replace('_', ' ').title()}:**\n{content}"
        for agent, content in findings.items()
    ])
    
    system_prompt = (
        "You are a research coordinator. Synthesize the team's findings into "
        "a comprehensive, coherent analysis. Identify connections between insights "
        "and provide an integrated perspective."
    )
    
    query = (
        f"Here are the specialist findings:\n\n{findings_text}\n\n"
        f"Provide a comprehensive synthesis that integrates all perspectives."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    response = await llm.ainvoke(messages)
    
    new_messages = [
        HumanMessage(content="[Synthesizer] Integrating team findings..."),
        AIMessage(content=f"[Synthesizer] {response.content}")
    ]
    
    return {"messages": new_messages}


def create_research_graph(llm: ChatOpenAI | AzureChatOpenAI) -> StateGraph:
    """Create a multi-agent research workflow.
    
    Args:
        llm: Language model to use
        
    Returns:
        Compiled research workflow graph
    """
    
    # Create specialist agents
    technical_agent = create_specialist_agent(
        "technical_analyst",
        "technical architecture, implementation, and engineering",
        llm
    )
    
    business_agent = create_specialist_agent(
        "business_analyst",
        "business value, market fit, and ROI",
        llm
    )
    
    ux_agent = create_specialist_agent(
        "ux_researcher",
        "user experience, usability, and adoption",
        llm
    )
    
    # Create synthesizer wrapper that properly handles async
    async def synthesizer_wrapper(state: ResearchState) -> ResearchState:
        return await synthesizer_agent(state, llm)
    
    # Create workflow graph
    workflow = StateGraph(ResearchState)
    
    # Add agents as parallel nodes
    workflow.add_node("technical_analyst", technical_agent)
    workflow.add_node("business_analyst", business_agent)
    workflow.add_node("ux_researcher", ux_agent)
    workflow.add_node("synthesizer", synthesizer_wrapper)
    
    # All agents work in parallel
    workflow.add_edge(START, "technical_analyst")
    workflow.add_edge(START, "business_analyst")
    workflow.add_edge(START, "ux_researcher")
    
    # All converge to synthesizer
    workflow.add_edge("technical_analyst", "synthesizer")
    workflow.add_edge("business_analyst", "synthesizer")
    workflow.add_edge("ux_researcher", "synthesizer")
    
    # End after synthesis
    workflow.add_edge("synthesizer", END)
    
    return workflow


async def main():
    """Run multi-agent research collaboration demo."""
    
    print("\n" + "=" * 70)
    print("🤖 Multi-Agent Research Team Demo")
    print("=" * 70 + "\n")
    
    # Check which OpenAI service to use
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Setup LLM
    if azure_endpoint and azure_key:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        print(f"🤖 Using Azure OpenAI")
        print(f"   Endpoint: {azure_endpoint}")
        print(f"   Deployment: {deployment}\n")
        
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment,
            api_version="2024-02-15-preview",
            temperature=0.7,
        )
    elif openai_key:
        print(f"🤖 Using OpenAI")
        print(f"   Model: gpt-4o-mini\n")
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )
    else:
        print("❌ No API key configured!")
        print("\nSet either:")
        print("  Azure: $env:AZURE_OPENAI_ENDPOINT, $env:AZURE_OPENAI_API_KEY")
        print("  OpenAI: $env:OPENAI_API_KEY")
        sys.exit(1)
    
    print(f"🗄️  Connecting to Kusto...")
    print(f"   Cluster: {CLUSTER_URI}")
    print(f"   Database: {DATABASE}\n")
    
    try:
        # Create checkpointer with shared memory
        async with AsyncKustoSaver.from_connection_string(
            cluster_uri=CLUSTER_URI,
            database=DATABASE,
        ) as checkpointer:
            await checkpointer.setup()
            print("✓ Connected to Kusto\n")
            
            # Create research workflow
            workflow = create_research_graph(llm)
            app = workflow.compile(checkpointer=checkpointer)
            
            print("✓ Multi-agent workflow compiled\n")
            
            # Research session configuration
            session_id = f"research-team-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            config = {
                "configurable": {
                    "thread_id": session_id,
                }
            }
            
            # Try to load previous research history from Kusto
            checkpoint = await checkpointer.aget_tuple(config)
            previous_history = {}
            if checkpoint:
                # Load research history from previous session
                channel_values = checkpoint.checkpoint.get("channel_values", {})
                previous_history = channel_values.get("research_history", {})
                if previous_history:
                    print("=" * 70)
                    print("📚 Loaded Previous Research from Kusto")
                    print("=" * 70)
                    for agent, findings in previous_history.items():
                        print(f"   {agent.replace('_', ' ').title()}: {len(findings)} previous finding(s)")
                    print()
            
            print("=" * 70)
            print("🎯 Research Topic: LangGraph Multi-Agent Systems")
            print("=" * 70 + "\n")
            
            # Phase 1: Initial investigation
            print("📋 PHASE 1: Initial Investigation")
            print("-" * 70)
            print("Deploying 3 specialist agents in parallel...")
            if previous_history:
                print("(Agents have access to previous research from Kusto)")
            print()
            
            initial_state = {
                "topic": "LangGraph for building multi-agent AI systems",
                "research_phase": "initial",
                "findings": {},
                "research_history": previous_history,  # Load from Kusto
                "messages": []
            }
            
            result = await app.ainvoke(initial_state, config=config)
            await checkpointer.flush()
            
            # Display findings
            print("\n📊 Initial Findings:")
            print("-" * 70)
            for agent, findings in result["findings"].items():
                print(f"\n🔍 {agent.replace('_', ' ').title()}:")
                print(f"   {findings[:200]}...")
            
            print("\n" + "=" * 70)
            print("📋 PHASE 2: Deep Dive (Building on Phase 1)")
            print("=" * 70 + "\n")
            
            # Get updated research history from result (includes Phase 1 findings)
            current_history = result.get("research_history", previous_history)
            
            # Phase 2: Deep dive with context from Phase 1 AND Kusto history
            deep_dive_state = {
                "topic": "LangGraph for building multi-agent AI systems",
                "research_phase": "deep_dive",
                "findings": {},  # Reset for new phase
                "research_history": current_history,  # Includes Phase 1 + Kusto history
                "messages": []
            }
            
            result = await app.ainvoke(deep_dive_state, config=config)
            await checkpointer.flush()
            
            # Display deep dive findings
            print("\n📊 Deep Dive Findings:")
            print("-" * 70)
            for agent, findings in result["findings"].items():
                print(f"\n🔍 {agent.replace('_', ' ').title()}:")
                print(f"   {findings[:200]}...")
            
            # Get final synthesis
            print("\n" + "=" * 70)
            print("📋 FINAL SYNTHESIS")
            print("=" * 70)
            
            final_messages = result.get("messages", [])
            synthesis_messages = [
                msg for msg in final_messages
                if "[Synthesizer]" in (msg.content if hasattr(msg, 'content') else "")
            ]
            
            if synthesis_messages:
                final_synthesis = synthesis_messages[-1]
                print(f"\n{final_synthesis.content}\n")
            
            # Show complete history
            print("=" * 70)
            print("📜 Complete Research History (from Kusto)")
            print("=" * 70)
            
            checkpoint = await checkpointer.aget_tuple(config)
            if checkpoint:
                channel_values = checkpoint.checkpoint.get("channel_values", {})
                all_messages = channel_values.get("messages", [])
                final_history = channel_values.get("research_history", {})
                
                print(f"\nTotal interactions: {len(all_messages)}")
                print(f"Session ID: {session_id}")
                
                # Show accumulated research by each agent
                if final_history:
                    print(f"\n📚 Accumulated Research in Kusto:")
                    for agent, findings_list in final_history.items():
                        print(f"   {agent.replace('_', ' ').title()}: {len(findings_list)} total finding(s)")
                
                print(f"\n💬 Recent Interactions:")
                for idx, msg in enumerate(all_messages[-10:], 1):  # Show last 10
                    if hasattr(msg, 'content'):
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        msg_type = "👤" if isinstance(msg, HumanMessage) else "🤖"
                        print(f"   {msg_type} {idx}. {content}")
            
            print("\n" + "=" * 70)
            print("✨ Multi-Agent Research Complete!")
            print("=" * 70)
            print(f"\n💡 All research is saved in Kusto with session ID: {session_id}")
            print("   ✅ Each agent's findings are stored in research_history")
            print("   ✅ Agents can access previous findings from Kusto")
            print("   ✅ Run again to see agents build on historical research!")
            print(f"\n🔄 To continue this research:")
            print(f'   Change session_id to "{session_id}" and run again\n')
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
