import pytest
from datetime import datetime, timedelta, timezone
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import _ensure_ttl

class MockStore(InMemoryStore):
    supports_ttl = True
    
    def __init__(self):
        super().__init__()
        self.captured_ops = []

    def batch(self, ops):
        # Convert iterator to list to capture and pass on
        ops_list = list(ops)
        self.captured_ops.extend(ops_list)
        return super().batch(ops_list)

    async def abatch(self, ops):
        # Convert iterator to list to capture and pass on
        ops_list = list(ops)
        self.captured_ops.extend(ops_list)
        return await super().abatch(ops_list)

def test_ensure_ttl_timedelta():
    td = timedelta(minutes=10)
    result = _ensure_ttl(None, td)
    assert result == 10.0

def test_ensure_ttl_datetime_utc():
    now = datetime.now(timezone.utc)
    future = now + timedelta(minutes=30)
    result = _ensure_ttl(None, future)
    # allow small delta for execution time
    assert 29.9 <= result <= 30.0

def test_ensure_ttl_datetime_naive():
    now = datetime.now()
    future = now + timedelta(minutes=30)
    # This uses local time in _ensure_ttl
    result = _ensure_ttl(None, future)
    assert 29.9 <= result <= 30.0

def test_store_put_timedelta():
    store = MockStore()
    store.put(("test",), "k1", {"a": 1}, ttl=timedelta(minutes=15))
    
    assert len(store.captured_ops) == 1
    op = store.captured_ops[0]
    assert op.ttl == 15.0

@pytest.mark.asyncio
async def test_store_aput_datetime():
    store = MockStore()
    future = datetime.now(timezone.utc) + timedelta(minutes=60)
    await store.aput(("test",), "k2", {"a": 1}, ttl=future)
    
    assert len(store.captured_ops) == 1
    op = store.captured_ops[0]
    assert 59.9 <= op.ttl <= 60.0
