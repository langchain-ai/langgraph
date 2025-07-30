#!/usr/bin/env python3
"""Script to add durability parameter to ainvoke method signature and update docstring."""

import re

def modify_ainvoke_method():
    file_path = "/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the ainvoke method and add durability parameter
    # Pattern to match the ainvoke method signature
    ainvoke_pattern = r'(    async def ainvoke\(\s*\n.*?interrupt_after: All \| Sequence\[str\] \| None = None,)\s*\n(        \*\*kwargs: Any,)'
    
    # Replacement with durability parameter added
    ainvoke_replacement = r'\1\n        durability: Durability | None = None,\n\2'
    
    # Apply the replacement
    content = re.sub(ainvoke_pattern, ainvoke_replacement, content, flags=re.DOTALL)
    
    # Update the ainvoke method docstring to include durability parameter
    docstring_pattern = r'(            interrupt_before: Optional\. The nodes to interrupt before\. Default is None\.\s*\n\s*interrupt_after: Optional\. The nodes to interrupt after\. Default is None\.\s*\n)(            \*\*kwargs: Additional keyword arguments\.)'
    
    # Replacement with durability parameter documentation added
    docstring_replacement = r'\1            durability: The durability mode for the graph execution, defaults to "async". Options are:\n                - `"sync"`: Changes are persisted synchronously before the next step starts.\n                - `"async"`: Changes are persisted asynchronously while the next step executes.\n                - `"exit"`: Changes are persisted only when the graph exits.\n\2'
    
    # Apply the replacement
    content = re.sub(docstring_pattern, docstring_replacement, content, flags=re.DOTALL)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Successfully added durability parameter to ainvoke method signature and updated docstring")

if __name__ == "__main__":
    modify_ainvoke_method()
