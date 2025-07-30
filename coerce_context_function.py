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
