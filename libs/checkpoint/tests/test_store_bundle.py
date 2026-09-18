from langgraph.store.memory import InMemoryStore


def test_bundle():
    store = InMemoryStore()
    assert hasattr(store, "export_bundle")
    assert hasattr(store, "import_bundle")
