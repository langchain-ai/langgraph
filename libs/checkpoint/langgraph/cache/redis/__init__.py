        async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        # For backward compatibility (as requested), delegate to sync clear
        self.clear(namespaces)
