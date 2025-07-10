#!/usr/bin/env python3

# Read the file
with open('libs/sdk-py/langgraph_sdk/client.py', 'r') as f:
    content = f.read()

# Split into lines for line-specific changes
lines = content.split('\n')

# Update line 4922 (index 4921) - change return type
if len(lines) > 4921 and '-> dict:' in lines[4921]:
    lines[4921] = lines[4921].replace('-> dict:', '-> dict[str, Any]:')
    print('Updated return type annotation')

# Update line 4931 (index 4930) - change docstring Returns section
if len(lines) > 4930 and 'None' in lines[4930]:
    lines[4930] = lines[4930].replace('None', 'dict[str, Any]: The final state values of the thread')
    print('Updated docstring Returns section')

# Write back to file
with open('libs/sdk-py/langgraph_sdk/client.py', 'w') as f:
    f.write('\n'.join(lines))

print('Changes applied successfully')


