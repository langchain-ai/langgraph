# Use Pydantic field aliases in state schemas

This guide demonstrates how to use Pydantic field aliases in LangGraph state schemas. Field aliases allow you to use different field names in your input data than what is defined in your Pydantic model, which can be useful for:

- Making field names more user-friendly (e.g., using camelCase in APIs but snake_case in Python)
- Avoiding name collisions with reserved keywords
- Integrating with external systems that use different field naming conventions

## Setup

First, install the required packages:

```bash
pip install langgraph pydantic
```

## Basic usage with Pydantic field aliases

You can define a Pydantic model with field aliases and use it as a state schema in LangGraph:

```python
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class State(BaseModel):
    user_id: str = Field(alias='id')
    user_name: str = Field(alias='name')
    email: str

def process_user(state: State) -> dict:
    """Process the user information."""
    return {
        "id": state.user_id + "_processed", 
        "name": state.user_name.upper(),
        "email": state.email
    }

# Create the graph
graph = StateGraph(State)
graph.add_node("process", process_user)
graph.add_edge(START, "process")
graph.add_edge("process", END)

# Compile the graph
compiled = graph.compile()

# Use the graph with aliased field names
result = compiled.invoke({
    "id": "user123",
    "name": "John Doe",
    "email": "john.doe@example.com"
})

print(result)
# Output: {'id': 'user123_processed', 'name': 'JOHN DOE', 'email': 'john.doe@example.com'}
```

## Using aliases with nested models

You can also use field aliases with nested Pydantic models:

```python
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class Address(BaseModel):
    street: str = Field(alias='streetAddress')
    city: str
    postal_code: str = Field(alias='postalCode')

class User(BaseModel):
    user_id: str = Field(alias='id')
    user_name: str = Field(alias='name')
    address: Address

def process_user(state: User) -> dict:
    return {
        "id": state.user_id + "_processed",
        "name": state.user_name.upper(),
        "address": {
            "streetAddress": state.address.street + " Ave",
            "city": state.address.city,
            "postalCode": state.address.postal_code
        }
    }

# Create the graph
graph = StateGraph(User)
graph.add_node("process", process_user)
graph.add_edge(START, "process")
graph.add_edge("process", END)
compiled = graph.compile()

# Use the graph with aliased field names in nested structures
result = compiled.invoke({
    "id": "user123",
    "name": "John Doe",
    "address": {
        "streetAddress": "123 Main",
        "city": "San Francisco",
        "postalCode": "94105"
    }
})

print(result)
```

## Mixing aliased and non-aliased fields

You can mix aliased and non-aliased fields in the same model:

```python
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class State(BaseModel):
    aliased_field: str = Field(alias='myAlias')
    normal_field: str

def process(state: State) -> dict:
    return {
        "myAlias": state.aliased_field + "_processed",
        "normal_field": state.normal_field + "_processed"
    }

graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
compiled = graph.compile()

result = compiled.invoke({
    "myAlias": "alias_value",
    "normal_field": "normal_value"
})

print(result)
# Output: {'myAlias': 'alias_value_processed', 'normal_field': 'normal_value_processed'}
```
