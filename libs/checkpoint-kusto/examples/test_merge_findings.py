"""Quick test to verify the merge_findings reducer works correctly."""

from tutorial_05_multi_agent import merge_findings, ResearchState
from typing import get_type_hints

def test_merge_findings():
    """Test the merge_findings reducer function."""
    
    # Test 1: Empty left
    result = merge_findings({}, {"agent1": "finding1"})
    assert result == {"agent1": "finding1"}, "Failed: empty left"
    print("✓ Test 1: Empty left works")
    
    # Test 2: Empty right
    result = merge_findings({"agent1": "finding1"}, {})
    assert result == {"agent1": "finding1"}, "Failed: empty right"
    print("✓ Test 2: Empty right works")
    
    # Test 3: Merge two agents
    result = merge_findings(
        {"agent1": "finding1"},
        {"agent2": "finding2"}
    )
    assert result == {"agent1": "finding1", "agent2": "finding2"}, "Failed: merge two"
    print("✓ Test 3: Merge two agents works")
    
    # Test 4: Update existing agent
    result = merge_findings(
        {"agent1": "old_finding"},
        {"agent1": "new_finding"}
    )
    assert result == {"agent1": "new_finding"}, "Failed: update existing"
    print("✓ Test 4: Update existing agent works")
    
    # Test 5: Multiple agents merging
    left = {"agent1": "finding1", "agent2": "finding2"}
    right = {"agent3": "finding3"}
    result = merge_findings(left, right)
    assert result == {
        "agent1": "finding1",
        "agent2": "finding2",
        "agent3": "finding3"
    }, "Failed: multiple agents"
    print("✓ Test 5: Multiple agents merge works")
    
    print("\n✅ All merge_findings tests passed!")
    
    # Verify the annotation is correct
    hints = get_type_hints(ResearchState, include_extras=True)
    findings_type = hints.get("findings")
    print(f"\n📋 ResearchState.findings type: {findings_type}")
    print("   (Should show Annotated with merge_findings)")

if __name__ == "__main__":
    test_merge_findings()
