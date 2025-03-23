# Testing Version Admonitions

This page demonstrates how to use version admonitions in LangGraph documentation.

## Interrupt Class

!!! version-added "Added in v0.2.24"
    The `Interrupt` class was introduced in LangGraph v0.2.24 (September 2024).
    
    ```python
    class Interrupt:
        value: Any
        when: Literal["during"] = "during"
    ```
    
    This class allows you to interrupt the execution of a graph and surface a value to the client.

## How To Use Version Admonitions

You can add version information to documentation by using these custom admonitions:

### In Markdown Files

```markdown
!!! version-added "Added in v0.X.Y"
    Feature description here.

!!! version-changed "Changed in v0.X.Y"
    Change description here.
```

### In Python Docstrings

```python
class SomeClass:
    """Class description.
    
    !!! version-added "Added in v0.X.Y"
        When this class was added.
    
    !!! version-changed "Changed in v0.X.Y"
        When this class was changed.
    """
``` 