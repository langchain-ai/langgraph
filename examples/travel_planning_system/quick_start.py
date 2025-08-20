#!/usr/bin/env python3
"""
Quick Start for Enhanced Travel Planning System
Run this to quickly test the system with sample data
"""

import os
import sys
from pathlib import Path

def quick_start():
    """Quick start with a sample scenario."""
    print("🌍 Enhanced Travel Planning System - Quick Start")
    print("="*60)
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        from enhanced_multi_agent_system import (
            create_travel_planning_graph, 
            SAMPLE_INPUTS,
            TravelPlanState
        )
        from langchain_core.messages import HumanMessage
        from uuid import uuid4
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Try running the setup first:")
        print("   python setup.py")
        return
    
    # Check if we have LLM configuration
    from config import config
    validation = config.validate_required_config()
    
    if not validation["llm_configured"]:
        print("⚠️  No LLM configuration found!")
        print("This quick start will show the system structure")
        print("but cannot run the full agent workflow.\n")
        print("To run the full system:")
        print("1. Copy .env.template to .env")
        print("2. Add your Azure OpenAI or OpenAI API key")
        print("3. Run: python enhanced_multi_agent_system.py\n")
        
        # Show what we can do without LLM
        print("🎯 Available Features (without LLM):")
        print("✅ Data source integration with fallback data")
        print("✅ Tool definitions and workflow structure")
        print("✅ Configuration management")
        print("✅ LangGraph workflow creation")
        
        # Demonstrate non-LLM features
        print("\n🔧 Creating workflow graph...")
        try:
            graph = create_travel_planning_graph()
            print("✅ LangGraph workflow created successfully!")
        except Exception as e:
            print(f"❌ Graph creation failed: {e}")
        
        # Show sample inputs
        print(f"\n📝 Available sample scenarios ({len(SAMPLE_INPUTS)}):")
        for i, (key, sample) in enumerate(SAMPLE_INPUTS.items(), 1):
            print(f"{i}. {key.replace('_', ' ').title()}")
        
        return
    
    # Full system available
    print("✅ LLM configured! Running sample scenario...")
    
    # Select a sample scenario
    sample_key = "business_trip"
    sample = SAMPLE_INPUTS[sample_key]
    user_request = sample["request"]
    
    print(f"\n🎯 Running scenario: {sample_key.replace('_', ' ').title()}")
    print(f"📝 Request: {user_request}")
    print(f"💡 Description: {sample['description']}")
    
    # Create graph
    print("\n🔧 Creating travel planning graph...")
    app = create_travel_planning_graph()
    
    # Initialize state
    session_id = str(uuid4())
    initial_state = TravelPlanState(
        messages=[HumanMessage(content=user_request)],
        user_request=user_request,
        destination=None,
        origin=None,
        budget=None,
        travelers=2,
        departure_date=None,
        return_date=None,
        trip_type=None,
        preferences={},
        flight_results=[],
        hotel_results=[],
        activity_results=[],
        weather_results=[],
        budget_analysis=None,
        current_step="start",
        completed_steps=[],
        final_plan=None,
        session_id=session_id,
        agent_handoffs=[]
    )
    
    # Configure execution
    config_dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20
    }
    
    print("\n🚀 Starting abbreviated travel planning process...")
    print("(This will show first few steps, run full system for complete planning)")
    
    try:
        # Run just a few steps to demonstrate
        step_count = 0
        max_steps = 3  # Limit for quick start
        
        for step in app.stream(initial_state, config_dict):
            step_count += 1
            if step_count > max_steps:
                break
                
            agent_name = list(step.keys())[0]
            agent_state = step[agent_name]
            
            print(f"\n🤖 {agent_name.replace('_', ' ').title()} Agent")
            print("-" * 40)
            
            if agent_state.get("messages"):
                last_message = agent_state["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    # Show abbreviated output
                    content = last_message.content
                    if len(content) > 300:
                        content = content[:300] + "... [truncated for quick start]"
                    print(content)
            
            completed = agent_state.get('completed_steps', [])
            if completed:
                print(f"\n✅ Completed: {', '.join(completed)}")
        
        print(f"\n🎊 Quick start completed! ({step_count} steps shown)")
        print("\n🚀 For full travel planning, run:")
        print("   python enhanced_multi_agent_system.py")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("This might be due to missing dependencies or configuration.")
        print("\n🔧 Try running setup first:")
        print("   python setup.py")

def main():
    """Main entry point."""
    try:
        quick_start()
    except KeyboardInterrupt:
        print("\n\n👋 Quick start cancelled. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n🔧 For help, try:")
        print("   python setup.py")
        print("   python demo.py")

if __name__ == "__main__":
    main()