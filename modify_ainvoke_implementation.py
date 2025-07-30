#!/usr/bin/env python3
"""Script to update ainvoke method implementation to pass durability explicitly."""

import re

def modify_ainvoke_implementation():
    file_path = "/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the ainvoke method astream call and add explicit durability parameter
    # Pattern to match the astream call in ainvoke method
    astream_call_pattern = r'(        async for chunk in self\.astream\(\s*\n.*?interrupt_after=interrupt_after,)\s*\n(            \*\*kwargs,)'
    
    # Replacement with durability parameter added
    astream_call_replacement = r'\1\n            durability=durability,\n\2'
    
    # Apply the replacement
    content = re.sub(astream_call_pattern, astream_call_replacement, content, flags=re.DOTALL)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Successfully updated ainvoke method implementation to pass durability explicitly")

if __name__ == "__main__":
    modify_ainvoke_implementation()
