# Override autouse db fixtures for unit tests that don't need a database.
# The parent conftest.py's autouse=True clear_test_db runs for integration
# tests; unit tests in tests/unit/ are isolated from that requirement.

import pytest


# Override the autouse function-scoped clear_test_db so it does nothing.
# Unit tests should never need this fixture.
@pytest.fixture(scope="function", autouse=True)
def clear_test_db():
    # No-op override — prevents parent conftest from creating a DB connection
    pass
