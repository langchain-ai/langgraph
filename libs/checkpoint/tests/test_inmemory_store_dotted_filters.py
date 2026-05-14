from langgraph.store.memory import InMemoryStore


def test_inmemory_store_search_supports_dotted_filter_keys() -> None:
    store = InMemoryStore()
    store.put(("docs",), "hyphen", {"user": {"access-level": "nested"}})

    results = store.search(("docs",), filter={"user.access-level": "nested"})

    assert [result.key for result in results] == ["hyphen"]
