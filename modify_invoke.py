#!/usr/bin/env python3
"""Script to add durability parameter to invoke method signature."""

import re

def modify_invoke_method():
    file_path = "/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the invoke method and add durability parameter
    # Pattern to match the invoke method signature
    invoke_pattern = r'(    def invoke\(\s*\n.*?interrupt_after: All \| Sequence\[str\] \| None = None,)\s*\n(        \*\*kwargs: Any,)'
    
    # Replacement with durability parameter added
    invoke_replacement = r'\1\n        durability: Durability | None = None,\n\2'
    
    # Apply the replacement
    content = re.sub(invoke_pattern, invoke_replacement, content, flags=re.DOTALL)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Successfully added durability parameter to invoke method signature")

if __name__ == "__main__":
    modify_invoke_method()
