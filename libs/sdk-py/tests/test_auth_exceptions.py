import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies globally before any imports happen
sys.modules["orjson"] = MagicMock()
sys.modules["httpx"] = MagicMock()
sys.modules["langgraph_sdk.client"] = MagicMock()
sys.modules["langgraph_sdk.encryption"] = MagicMock()
sys.modules["langgraph_sdk.encryption.types"] = MagicMock()

class TestAuthExceptions(unittest.TestCase):
    
    def setUp(self):
        # Un-import the module so we can re-import it with different mocks
        if "langgraph_sdk.auth.exceptions" in sys.modules:
            del sys.modules["langgraph_sdk.auth.exceptions"]
        if "langgraph_sdk.auth" in sys.modules:
            del sys.modules["langgraph_sdk.auth"]

    def test_starlette_available(self):
        """Test that HTTPException inherits from starlette.exceptions.HTTPException when available."""
        
        # Create a mock Starlette exception
        class MockStarletteHTTPException(Exception):
            def __init__(self, status_code: int = 401, detail: str = None, **kwargs):
                self.status_code = status_code
                self.detail = detail
                
        mock_starlette = MagicMock()
        mock_starlette.exceptions.HTTPException = MockStarletteHTTPException

        with patch.dict(sys.modules, {
            "starlette": mock_starlette,
            "starlette.exceptions": mock_starlette.exceptions
        }):
            from langgraph_sdk.auth import exceptions
            
            # Verify inheritance
            self.assertTrue(issubclass(exceptions.HTTPException, MockStarletteHTTPException))
            
            # Verify usage
            exc = exceptions.HTTPException(status_code=403, detail="Forbidden")
            self.assertIsInstance(exc, MockStarletteHTTPException)
            self.assertEqual(exc.status_code, 403)
            self.assertEqual(exc.detail, "Forbidden")

    def test_starlette_missing(self):
        """Test that HTTPException inherits from Exception when starlette is missing."""
        
        # Simulate ImportError by setting starlette to None (or just ensuring it's not there)
        # Using a patch that side_effects ImportError on import is harder with sys.modules dict.
        # But we can just make sure it's NOT in sys.modules, and if it tries to import, we need to make it fail.
        # 'builtins.__import__' patching is complex.
        # Simpler approach: In our code we check `try: from starlette... except ImportError`.
        
        # We can simulate this by setting sys.modules['starlette'] = None which causes ImportError in some py versions,
        # or we just assume it's not installed in this test env.
        # If it IS installed, we need to hide it.
        
        with patch.dict(sys.modules):
            # Remove starlette if it exists
            if "starlette" in sys.modules:
                del sys.modules["starlette"]
            if "starlette.exceptions" in sys.modules:
                del sys.modules["starlette.exceptions"]
            
            # We also need to prevent it from being found. 
            # A straightforward way to force ImportError is mocking sys.meta_path or wrapping import,
            # but for this specific "from x import y" pattern:
            
            with patch('builtins.__import__', side_effect=ImportError("No starlette")):
                 # This might be too aggressive, identifying ONLY starlette import is better.
                 pass

            # Easier way: The code does `from starlette.exceptions import ...`
            # We can put a dummy module there that RAISES ImportError on attribute access? No.
            
            # Let's rely on the fact that if we replace the function `__import__` we can filter.
            original_import = __import__
            def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "starlette.exceptions" or (name == "starlette" and fromlist):
                    raise ImportError("No module named 'starlette'")
                return original_import(name, globals, locals, fromlist, level)
            
            with patch('builtins.__import__', side_effect=mock_import):
                 from langgraph_sdk.auth import exceptions
                 
                 # Verify inheritance from basic Exception
                 self.assertTrue(issubclass(exceptions.HTTPException, Exception))
                 # And NOT Starlette (if we had a real one, but we don't here)
                 
                 # Verify usage
                 exc = exceptions.HTTPException(status_code=404, detail="Not Found")
                 self.assertIsInstance(exc, Exception)
                 self.assertEqual(exc.status_code, 404)
                 self.assertEqual(exc.detail, "Not Found")

if __name__ == "__main__":
    unittest.main()
