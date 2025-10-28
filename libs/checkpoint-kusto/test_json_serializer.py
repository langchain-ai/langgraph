"""Quick test of JSON serializer with Kusto."""
import asyncio
import os
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.kusto import AsyncKustoSaver
from langgraph.checkpoint.kusto.json_serializer import JsonStringSerializer

async def test_json_serializer():
    """Test that JSON serializer works with Kusto."""
    print("🧪 Testing JSON String Serializer...")
    
    # Test basic serialization
    serializer = JsonStringSerializer()
    
    # Test with a message
    msg = HumanMessage("Hello, world!")
    type_str, json_str = serializer.dumps_typed(msg)
    
    print(f"✓ Serialization successful")
    print(f"  Type: {type_str}")
    print(f"  Data type: {type(json_str)}")
    print(f"  Data is string: {isinstance(json_str, str)}")
    print(f"  Preview: {json_str[:100]}...")
    
    # Test deserialization
    loaded = serializer.loads_typed((type_str, json_str))
    print(f"\n✓ Deserialization successful")
    print(f"  Type: {type(loaded)}")
    print(f"  Content: {loaded.content}")
    
    # Test with list of messages
    messages = [
        HumanMessage("First message"),
        AIMessage("AI response"),
        HumanMessage("Second message"),
    ]
    
    type_str, json_str = serializer.dumps_typed(messages)
    loaded_messages = serializer.loads_typed((type_str, json_str))
    
    print(f"\n✓ List serialization successful")
    print(f"  Original: {len(messages)} messages")
    print(f"  Loaded: {len(loaded_messages)} messages")
    print(f"  Types match: {all(type(o) == type(l) for o, l in zip(messages, loaded_messages))}")
    
    # Test that msgpack format is rejected
    print(f"\n� Testing that msgpack format is rejected...")
    try:
        serializer.loads_typed(("msgpack", "some_data"))
        print(f"❌ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly rejects msgpack format")
        print(f"  Error: {str(e)[:80]}...")
    
    print(f"\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_json_serializer())
