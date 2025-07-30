#!/usr/bin/env python3
"""Script to add the _coerce_context function to pregel/main.py"""

# Read the original file
with open('/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py', 'r') as f:
    content = f.read()

# Find the location to insert the function
insert_pos = content.find('_WriteValue = Union[Callable[[Input], Output], Any]\n\n\nclass NodeBuilder:')
if insert_pos == -1:
    print('Could not find insertion point')
    exit(1)

# Insert the _coerce_context function
function_code = '''

def _coerce_context(
    context_schema: type[ContextT] | None, context: Any
) -> ContextT | None:
    """Coerce context input to the appropriate schema type.
    
    Args:
        context_schema: The schema type to coerce to (BaseModel, dataclass, or TypedDict)
        context: The context value to coerce
        
    Returns:
        The coerced context value or None if context is None
    """
    if context is None or context_schema is None:
        return context
    
    # If context is a dict and we have a schema, try to coerce
    if isinstance(context, dict):
        # Handle TypedDict first - pass through as-is since TypedDicts are just dicts at runtime
        if is_typeddict(context_schema):
            return context
        # Handle Pydantic BaseModel
        elif isclass(context_schema) and issubclass(context_schema, BaseModel):
            return context_schema(**context)
        # Handle dataclass
        elif is_dataclass(context_schema):
            return context_schema(**context)
    
    # If context is already the correct type, return as-is
    # This check is done after dict check to avoid TypedDict isinstance issues
    try:
        if isinstance(context, context_schema):
            return context
    except TypeError:
        # TypedDict raises TypeError on isinstance check
        pass
    
    # If we can't coerce, return the context as-is
    return context
'''

# Insert after _WriteValue definition
insert_at = insert_pos + len('_WriteValue = Union[Callable[[Input], Output], Any]\n\n')
new_content = content[:insert_at] + function_code + '\n' + content[insert_at:]

# Write the modified content back
with open('/home/daytona/langgraph/libs/langgraph/langgraph/pregel/main.py', 'w') as f:
    f.write(new_content)

print('Successfully added _coerce_context function')
