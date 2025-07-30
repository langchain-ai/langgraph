#!/usr/bin/env python3
"""Script to add durability parameter to invoke method signature and update docstring."""

import re

def modify_invoke_method():
    file_path = "/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the invoke method docstring to include durability parameter
    docstring_pattern = r'(            interrupt_before: Optional\. The nodes to interrupt the graph run before\.\s*\n\s*interrupt_after: Optional\. The nodes to interrupt the graph run after\.\s*\n)(            \*\*kwargs: Additional keyword arguments to pass to the graph run\.)'
    
    # Replacement with durability parameter documentation added
    docstring_replacement = r'\1            durability: The durability mode for the graph execution, defaults to "async". Options are:\n                - `"sync"`: Changes are persisted synchronously before the next step starts.\n                - `"async"`: Changes are persisted asynchronously while the next step executes.\n                - `"exit"`: Changes are persisted only when the graph exits.\n\2'
    
    # Apply the replacement
    content = re.sub(docstring_pattern, docstring_replacement, content, flags=re.DOTALL)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Successfully updated invoke method docstring with durability parameter documentation")

if __name__ == "__main__":
    modify_invoke_method()

