import sys
from unittest.mock import MagicMock, AsyncMock, patch

# --------------------------------------------------------------------------
# 彻底拦截 psycopg 导入，支持泛型下标语法和 isinstance 检查
# --------------------------------------------------------------------------
class SubscriptableMeta(type):
    def __getitem__(cls, item):
        return cls

class MockConnection(metaclass=SubscriptableMeta):
    connect = MagicMock()

class MockAsyncConnection(metaclass=SubscriptableMeta):
    connect = AsyncMock() # 必须是 AsyncMock 才能被 await

class MockConnectionPool(metaclass=SubscriptableMeta): pass
class MockAsyncConnectionPool(metaclass=SubscriptableMeta): pass

mock_psycopg = MagicMock()
mock_psycopg.Connection = MockConnection
mock_psycopg.AsyncConnection = MockAsyncConnection

mock_psycopg_pool = MagicMock()
mock_psycopg_pool.ConnectionPool = MockConnectionPool
mock_psycopg_pool.AsyncConnectionPool = MockAsyncConnectionPool

sys.modules["psycopg"] = mock_psycopg
sys.modules["psycopg.rows"] = MagicMock()
sys.modules["psycopg.types.json"] = MagicMock()
sys.modules["psycopg_pool"] = mock_psycopg_pool

# 导入要测试的模块
import unittest
from langgraph.checkpoint.postgres import PostgresSaver, ShallowPostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres.shallow import AsyncShallowPostgresSaver

class TestCheckpointPrepareThresholdRegressions(unittest.TestCase):
    def test_postgres_saver_mock(self) -> None:
        mock_conn = MagicMock()
        MockConnection.connect.return_value.__enter__.return_value = mock_conn
        
        with PostgresSaver.from_conn_string("dsn", prepare_threshold=42):
            pass
        
        _, kwargs = MockConnection.connect.call_args
        self.assertEqual(kwargs["prepare_threshold"], 42)

    def test_shallow_postgres_saver_mock(self) -> None:
        mock_conn = MagicMock()
        MockConnection.connect.return_value.__enter__.return_value = mock_conn
        
        with ShallowPostgresSaver.from_conn_string("dsn", prepare_threshold=None):
            pass
        
        _, kwargs = MockConnection.connect.call_args
        self.assertIsNone(kwargs["prepare_threshold"])

class TestAsyncCheckpointPrepareThresholdRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_async_postgres_saver_mock(self) -> None:
        # 代码逻辑是：async with await AsyncConnection.connect(...) as conn:
        # 1. AsyncConnection.connect(...) 返回一个 awaitable
        # 2. await 该 awaitable 得到一个上下文管理器对象
        # 3. 该对象支持 __aenter__
        mock_conn = AsyncMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        
        # 模拟 connect(...) 返回一个可以通过 await 得到 mock_cm 的对象
        MockAsyncConnection.connect.return_value = AsyncMock(return_value=mock_cm)
        
        async with AsyncPostgresSaver.from_conn_string("dsn", prepare_threshold=None):
            pass
        
        _, kwargs = MockAsyncConnection.connect.call_args
        self.assertIsNone(kwargs["prepare_threshold"])

    async def test_async_shallow_postgres_saver_mock(self) -> None:
        mock_conn = AsyncMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        
        MockAsyncConnection.connect.return_value = AsyncMock(return_value=mock_cm)
        
        async with AsyncShallowPostgresSaver.from_conn_string("dsn", prepare_threshold=7):
            pass
        
        _, kwargs = MockAsyncConnection.connect.call_args
        self.assertEqual(kwargs["prepare_threshold"], 7)
