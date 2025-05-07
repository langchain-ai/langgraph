import logging
import subprocess
import time

import pytest
import requests

logger = logging.getLogger(__name__)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def dev_server():
    # Command to start the dev server
    cmd = [
        "uvx",
        "--with-requirements",
        "integration_tests/example_app/requirements.txt",
        "--from",
        "langgraph-cli[inmem]",
        "langgraph",
        "dev",
        "--config",
        "integration_tests/example_app/langgraph.json",
        "--no-browser",
    ]
    server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    url = "http://localhost:2024/ok"
    timeout = 30
    start = time.time()
    while True:
        if server.poll() is not None:
            out, err = server.communicate()
            raise RuntimeError(
                f"Dev server exited early.\nstdout: {out.decode()}\nstderr: {err.decode()}"
            )
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        if time.time() - start > timeout:
            server.terminate()
            out, err = server.communicate()
            raise TimeoutError(
                f"Timed out waiting for dev server to start.\nstdout: {out.decode()}\nstderr: {err.decode()}"
            )
        time.sleep(0.25)
    yield
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
