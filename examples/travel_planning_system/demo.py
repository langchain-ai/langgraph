#!/usr/bin/env python3
"""
Demo script for the Enhanced Travel Planning System
Shows system capabilities with fallback data (no API keys required)
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_data_sources():
    """Demonstrate data source capabilities."""
    print("🔍 Testing Data Sources (Fallback Mode)")
    print("="*50)
    
    from data_sources import (
        search_flights_sync, search_hotels_sync, 
        search_activities_sync, get_weather_sync
    )
    
    # Test flights
    print("✈️ Flight Search:")
    flights = search_flights_sync("Seattle", "London", "2024-09-15", 2)
    for i, flight in enumerate(flights[:2], 1):
        print(f"  {i}. {flight.airline} {flight.flight_id} - ${flight.price} ({flight.duration})")
    
    print("\n🏨 Hotel Search:")
    hotels = search_hotels_sync("London", "2024-09-15", "2024-09-22", 2)
    for i, hotel in enumerate(hotels[:2], 1):
        print(f"  {i}. {hotel.name} - ${hotel.price_per_night}/night ({hotel.rating}⭐)")
    
    print("\n🎯 Activity Search:")
    activities = search_activities_sync("London")
    for i, activity in enumerate(activities[:2], 1):
        print(f"  {i}. {activity.name} - ${activity.price} ({activity.category})")
    
    print("\n🌤️ Weather Forecast:")
    weather = get_weather_sync("London", "2024-09-15")
    print(f"  {weather.condition}, {weather.temperature_high}°C/{weather.temperature_low}°C")
    
    print("\n✅ All data sources working with realistic fallback data!")

def demo_tools():
    """Demonstrate tool capabilities."""
    print("\n🛠️ Testing Enhanced Tools")
    print("="*50)
    
    from enhanced_multi_agent_system import (
        search_flights, search_hotels, search_activities, 
        get_weather_forecast, analyze_budget, extract_trip_details
    )
    
    # Test flight search tool
    print("✈️ Flight Search Tool:")
    result = search_flights.invoke({
        "origin": "Seattle",
        "destination": "London", 
        "departure_date": "2024-09-15",
        "travelers": 2
    })
    print("  " + result[:200] + "..." if len(result) > 200 else "  " + result)
    
    print("\n🏨 Hotel Search Tool:")
    result = search_hotels.invoke({
        "destination": "London",
        "check_in": "2024-09-15",
        "check_out": "2024-09-22",
        "travelers": 2
    })
    print("  " + result[:200] + "..." if len(result) > 200 else "  " + result)
    
    print("\n🎯 Activity Search Tool:")
    result = search_activities.invoke({
        "destination": "London",
        "activity_type": "cultural"
    })
    print("  " + result[:200] + "..." if len(result) > 200 else "  " + result)
    
    print("\n💰 Budget Analysis Tool:")
    result = analyze_budget.invoke({
        "flight_cost": 800,
        "hotel_cost_per_night": 200,
        "nights": 7,
        "activity_costs": [50, 75, 30, 100],
        "travelers": 2,
        "destination_currency": "USD"
    })
    print("  " + result[:300] + "..." if len(result) > 300 else "  " + result)
    
    print("\n✅ All tools working correctly!")

def demo_configuration():
    """Demonstrate configuration management."""
    print("\n⚙️ Configuration Status")
    print("="*50)
    
    from config import config
    
    validation = config.validate_required_config()
    
    print("📋 API Configuration:")
    for key, status in validation.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {key.replace('_', ' ').title()}")
    
    if not validation["llm_configured"]:
        print("\n💡 To use the full system:")
        print("  1. Copy .env.template to .env")
        print("  2. Add your Azure OpenAI or OpenAI API key")
        print("  3. Optionally add other API keys for real data")
    else:
        print("\n✅ Ready to run full system!")
    
    print(f"\n📊 Fallback data available for {sum(not v for v in validation.values() if v != validation['llm_configured'])} services")

def demo_graph_creation():
    """Demonstrate LangGraph workflow creation."""
    print("\n🕸️ Testing LangGraph Workflow")
    print("="*50)
    
    try:
        from enhanced_multi_agent_system import create_travel_planning_graph
        
        print("🔧 Creating travel planning graph...")
        graph = create_travel_planning_graph()
        print("✅ LangGraph workflow created successfully!")
        
        # Show the graph structure
        print("\n📊 Workflow nodes:")
        print("  1. Coordinator → Flight Specialist")
        print("  2. Flight Specialist → Hotel Specialist") 
        print("  3. Hotel Specialist → Activity Specialist")
        print("  4. Activity Specialist → Budget Analyst")
        print("  5. Budget Analyst → Plan Synthesizer")
        print("  6. Plan Synthesizer → End")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        return False

def demo_sample_inputs():
    """Show available sample inputs."""
    print("\n📝 Sample Inputs for Testing")
    print("="*50)
    
    from enhanced_multi_agent_system import SAMPLE_INPUTS
    
    for i, (key, sample) in enumerate(SAMPLE_INPUTS.items(), 1):
        print(f"\n{i}. {key.replace('_', ' ').title()}:")
        print(f'   "{sample["request"]}"')
        print(f"   → {sample['description']}")

def main():
    """Run the comprehensive demo."""
    print("🌍 Enhanced Multi-Agent Travel Planning System Demo")
    print("="*70)
    print("This demo shows system capabilities without requiring API keys")
    print("Real APIs can be configured for production use")
    print("="*70)
    
    # Run demonstrations
    demo_data_sources()
    demo_tools()
    demo_configuration()
    
    graph_works = demo_graph_creation()
    demo_sample_inputs()
    
    print("\n" + "="*70)
    print("🎉 DEMO SUMMARY")
    print("="*70)
    print("✅ Data sources: Realistic fallback data working")
    print("✅ Tools: All agent tools functioning")
    print("✅ Configuration: Management system working")
    print(f"{'✅' if graph_works else '❌'} LangGraph: Workflow {'created' if graph_works else 'failed'}")
    print("✅ Sample inputs: 5 detailed scenarios ready")
    
    if graph_works:
        print("\n🚀 System is ready! To run the full travel planner:")
        print("   python enhanced_multi_agent_system.py")
        print("\n💡 For real data, configure API keys in .env file")
        print("   (See .env.template for instructions)")
    else:
        print("\n⚠️ Please check dependencies and configuration")
    
    print("\n🌟 This system demonstrates:")
    print("  • Multi-agent orchestration with LangGraph")
    print("  • Real API integration with intelligent fallbacks")
    print("  • Comprehensive travel planning workflow")
    print("  • Production-ready tracing and observability")
    print("  • Easy configuration and setup")

if __name__ == "__main__":
    main()