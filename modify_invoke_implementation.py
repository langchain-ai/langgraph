#!/usr/bin/env python3
"""Script to update invoke method implementation to pass durability explicitly."""

import re

def modify_invoke_implementation():
    file_path = "/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the invoke method stream call and add explicit durability parameter
    # Pattern to match the stream call in invoke method
    stream_call_pattern = r'(        for chunk in self\.stream\(\s*\n.*?interrupt_after=interrupt_after,)\s*\n(            \*\*kwargs,)'
    
    # Replacement with durability parameter added
    stream_call_replacement = r'\1\n            durability=durability,\n\2'
    
    # Apply the replacement
    content = re.sub(stream_call_pattern, stream_call_replacement, content, flags=re.DOTALL)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Successfully updated invoke method implementation to pass durability explicitly")

if __name__ == "__main__":
    modify_invoke_implementation()
